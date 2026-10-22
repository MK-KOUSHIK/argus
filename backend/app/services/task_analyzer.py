"""Task Analysis and Intent Extraction Module (Phase 3.1).

Performs lightweight deterministic inspection of raw user prompts to detect:
- Task category
- Explicit constraints (prohibitions, filenames, exact keys, requested format)
- Quality signals (redundancy, underspecification, existing clarity)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class RequiredOperation:
    """A distinct functional operation or requirement within the prompt."""

    raw_text: str
    action_verb: str
    target: str
    normalized_phrase: str
    operation_family: str
    key_terms: List[str] = field(default_factory=list)
    associated_entities: List[str] = field(default_factory=list)


@dataclass
class TaskAnalysisResult:
    """Structured analysis of prompt intent, domain, and constraint guardrails."""

    task_category: str
    primary_action: Optional[str] = None
    primary_deliverable: Optional[str] = None
    action_family: Optional[str] = None
    deliverable_family: Optional[str] = None
    required_operations: List[RequiredOperation] = field(default_factory=list)
    explicit_constraints: List[str] = field(default_factory=list)
    prohibitions: List[str] = field(default_factory=list)
    filenames: List[str] = field(default_factory=list)
    specific_keys: List[str] = field(default_factory=list)
    format_requested: Optional[str] = None
    is_already_effective: bool = False
    is_underspecified: bool = False
    has_redundancy: bool = False
    has_conversational_filler: bool = False
    complexity_score: int = 1


class TaskAnalyzer:
    """Inspects prompt characteristics to inform strategy selection and candidate validation."""

    # Category keyword patterns
    CODING_KEYWORDS = {
        "python", "function", "code", "script", "class", "method", "bug", "sql",
        "api", "algorithm", "refactor", "typescript", "javascript", "react",
        "java", "c++", "rust", "golang", "test", "unittest", "pytest", "endpoint",
        "database", "schema", "query", "debug", "compile", "memory leak",
    }

    DATA_KEYWORDS = {
        "csv", "revenue", "pandas", "dataframe", "dataset", "calculate", "metrics",
        "monthly", "sales", "statistics", "distribution", "analytics", "table",
        "aggregate", "numpy", "plot", "chart", "trend", "averages", "median",
    }

    EXTRACTION_KEYWORDS = {
        "extract", "parse", "json", "entities", "entity", "receipt", "invoice",
        "regex", "fields", "unstructured", "structure into", "key-value", "schema",
    }

    SUMMARIZATION_KEYWORDS = {
        "summarize", "summary", "tl;dr", "tldr", "brief", "condense", "overview",
        "digest", "shorten", "abstract", "bulleted summary",
    }

    WRITING_KEYWORDS = {
        "write an essay", "article", "blog", "draft", "story", "copywriting",
        "tone", "proofread", "rewrite the text", "paraphrase", "email to",
    }

    RESEARCH_KEYWORDS = {
        "research", "what is", "explain how", "history of", "compare and contrast",
        "literature", "deep dive", "mechanism of", "why does",
    }

    PLANNING_KEYWORDS = {
        "itinerary", "schedule", "roadmap", "plan", "recommend", "steps to",
        "strategy for", "milestones", "checklist", "prioritize",
    }

    # Action Taxonomy: verb -> (normalized action, action_family)
    ACTION_TAXONOMY = {
        # Software & Code creation
        "build": ("build", "software_development"),
        "create": ("create", "software_development"),
        "write": ("write", "software_development"),
        "implement": ("implement", "software_development"),
        "develop": ("develop", "software_development"),
        "code": ("code", "software_development"),
        "generate": ("generate", "software_development"),
        "program": ("program", "software_development"),
        "refactor": ("refactor", "software_development"),
        "debug": ("debug", "software_development"),
        # Explanation & Teaching
        "explain": ("explain", "explanation"),
        "describe": ("describe", "explanation"),
        "teach": ("teach", "explanation"),
        "clarify": ("clarify", "explanation"),
        "illustrate": ("illustrate", "explanation"),
        # Analysis & Computation
        "analyze": ("analyze", "data_analysis"),
        "calculate": ("calculate", "data_analysis"),
        "compute": ("compute", "data_analysis"),
        "evaluate": ("evaluate", "data_analysis"),
        "audit": ("audit", "data_analysis"),
        # Summarization
        "summarize": ("summarize", "summarization"),
        "condense": ("condense", "summarization"),
        # Extraction
        "extract": ("extract", "extraction"),
        "parse": ("parse", "extraction"),
        # Planning & Architecture
        "plan": ("plan", "planning"),
        "recommend": ("recommend", "planning"),
        "outline": ("outline", "planning"),
        "design": ("design", "planning"),
        # Return / Data Retrieval
        "return": ("return", "data_analysis"),
    }

    # Deliverable Taxonomy: noun -> (normalized deliverable, deliverable_family)
    DELIVERABLE_TAXONOMY = {
        # Software deliverables
        "program": ("program", "software"),
        "script": ("script", "software"),
        "function": ("function", "software"),
        "class": ("class", "software"),
        "method": ("method", "software"),
        "code": ("code", "software"),
        "application": ("application", "software"),
        "app": ("app", "software"),
        "algorithm": ("algorithm", "software"),
        "module": ("module", "software"),
        "endpoint": ("endpoint", "software"),
        "api": ("api", "software"),
        "parser": ("parser", "software"),
        "query": ("query", "software"),
        # Explanation deliverables
        "explanation": ("explanation", "explanation"),
        "overview": ("overview", "explanation"),
        "guide": ("guide", "explanation"),
        "tutorial": ("tutorial", "explanation"),
        "walkthrough": ("walkthrough", "explanation"),
        "example": ("example", "explanation"),
        # Document deliverables
        "summary": ("summary", "document"),
        "report": ("report", "document"),
        "essay": ("essay", "document"),
        "article": ("article", "document"),
        "draft": ("draft", "document"),
        "email": ("email", "document"),
        # Plan deliverables
        "plan": ("plan", "plan"),
        "itinerary": ("itinerary", "plan"),
        "roadmap": ("roadmap", "plan"),
        "schedule": ("schedule", "plan"),
        "checklist": ("checklist", "plan"),
        # Structured Data deliverables
        "table": ("table", "data"),
        "json": ("json", "data"),
        "csv": ("csv", "data"),
    }

    # Operation Verb Families: canonical_verb -> (canonical_verb, op_family, set_of_inflections)
    OPERATION_VERB_FAMILIES = {
        # Ingestion / Input
        "read": (
            "read",
            "ingest",
            {
                "read", "reads", "reading", "load", "loads", "loading", "import", "imports",
                "importing", "fetch", "fetches", "fetching", "ingest", "ingests", "ingesting",
                "open", "opens", "opening", "parse", "parses", "parsing", "scan", "scans", "scanning",
            },
        ),
        # Computation / Calculation / Aggregation
        "calculate": (
            "calculate",
            "computation",
            {
                "calculate", "calculates", "calculating", "compute", "computes", "computing",
                "determine", "determines", "determining", "find", "finds", "finding", "evaluate",
                "evaluates", "evaluating", "measure", "measures", "measuring", "estimate",
                "estimates", "estimating", "quantify", "quantifies", "quantifying", "aggregate",
                "aggregates", "aggregating", "sum", "sums", "summing", "total", "totals",
                "totaling", "count", "counts", "counting", "average", "averages", "averaging",
                "score", "scores", "scoring",
            },
        ),
        # Transformation / Grouping / Sorting / Filtering / Cleaning
        "group": (
            "group",
            "transformation",
            {
                "group", "groups", "grouping", "cluster", "clusters", "clustering", "categorize",
                "categorizes", "categorizing", "segment", "segments", "segmenting", "bucket",
                "buckets", "aggregate", "aggregates", "aggregating", "partition", "partitions",
                "partitioning",
            },
        ),
        "sort": (
            "sort",
            "transformation",
            {
                "sort", "sorts", "sorting", "order", "orders", "ordering", "rank", "ranks",
                "ranking", "arrange", "arranges", "arranging",
            },
        ),
        "filter": (
            "filter",
            "transformation",
            {
                "filter", "filters", "filtering", "select", "selects", "selecting", "exclude",
                "excludes", "excluding", "isolate", "isolates", "isolating", "drop", "drops",
                "dropping", "remove", "removes", "removing", "keep", "keeps", "keeping",
                "screen", "screens", "screening",
            },
        ),
        "clean": (
            "clean",
            "transformation",
            {
                "clean", "cleans", "cleaning", "normalize", "normalizes", "normalizing",
                "sanitize", "sanitizes", "sanitizing", "format", "formats", "formatting",
                "standardize", "standardizes", "standardizing", "convert", "converts",
                "converting", "transform", "transforms", "transforming", "deduplicate",
                "deduplicates", "deduplicating",
            },
        ),
        "merge": (
            "merge",
            "transformation",
            {
                "merge", "merges", "merging", "join", "joins", "joining", "combine", "combines",
                "combining", "integrate", "integrates", "integrating",
            },
        ),
        # Identification / Search / Extreme finding
        "identify": (
            "identify",
            "identification",
            {
                "identify", "identifies", "identifying", "find", "finds", "finding", "detect",
                "detects", "detecting", "locate", "locates", "locating", "discover", "discovers",
                "discovering", "spot", "spots", "spotting", "pinpoint", "pinpoints", "pinpointing",
                "pick", "picks", "picking", "select", "selects", "selecting", "highlight",
                "highlights", "highlighting",
            },
        ),
        # Extraction
        "extract": (
            "extract",
            "extraction",
            {
                "extract", "extracts", "extracting", "pull", "pulls", "pulling", "scrape",
                "scrapes", "scraping", "parse", "parses", "parsing", "retrieve", "retrieves",
                "retrieving", "collect", "collects", "collecting",
            },
        ),
        # Storage / Output / Export
        "save": (
            "save",
            "output",
            {
                "save", "saves", "saving", "write", "writes", "writing", "export", "exports",
                "exporting", "store", "stores", "storing", "dump", "dumps", "dumping",
                "output", "outputs", "outputting", "record", "records", "recording",
                "persist", "persists", "persisting",
            },
        ),
        # Generation / Display / Return
        "display": (
            "display",
            "output",
            {
                "display", "displays", "displaying", "print", "prints", "printing", "show",
                "shows", "showing", "render", "renders", "rendering", "log", "logs", "logging",
                "return", "returns", "returning",
            },
        ),
    }

    # Canonical concept synonyms and semantic clusters
    CONCEPT_SYNONYMS = {
        "average": {"average", "mean", "avg"},
        "mean": {"average", "mean", "avg"},
        "highest": {"highest", "top", "max", "maximum", "best", "greatest", "peak", "highest-scoring", "top-performing"},
        "highest-scoring": {"highest", "top", "max", "maximum", "best", "greatest", "peak", "highest-scoring", "top-performing"},
        "lowest": {"lowest", "bottom", "min", "minimum", "least", "worst", "lowest-scoring", "bottom-performing"},
        "lowest-scoring": {"lowest", "bottom", "min", "minimum", "least", "worst", "lowest-scoring", "bottom-performing"},
        "total": {"total", "sum", "aggregate", "overall", "cumulative", "gross", "summation"},
        "sum": {"total", "sum", "aggregate", "summation"},
        "count": {"count", "number", "quantity", "frequency", "tally"},
        "median": {"median", "midpoint", "percentile"},
        "spending": {"spending", "spent", "spend", "expenditure", "expense", "expenses", "cost", "costs"},
        "marks": {"marks", "mark", "score", "scores", "scoring", "grade", "grades", "points", "result", "results"},
        "student": {"student", "students", "pupil", "pupils", "learner", "learners"},
        "summary": {"summary", "overview", "breakdown", "digest", "report"},
        "results": {"results", "result", "output", "outputs", "findings", "outcome"},
        "customer": {"customer", "customers", "client", "clients", "user", "users", "buyer", "buyers"},
        "order": {"order", "orders", "purchase", "purchases", "transaction", "transactions"},
    }

    STOPWORDS = {
        "the", "a", "an", "all", "each", "every", "any", "some",
        "of", "in", "to", "for", "from", "by", "with", "into", "onto", "on", "at",
        "and", "or", "that", "which", "this", "these", "those",
        "their", "its", "your", "our", "my", "his", "her",
        "is", "are", "was", "were", "be", "been", "being",
        "it", "them", "then", "also", "just", "please", "make", "sure"
    }

    # Filler words and pleasantries
    FILLER_PATTERNS = [
        r"\b(?:could you please|can you please|please|kindly)\b",
        r"\bi would (?:really )?(?:appreciate|like|love) it if you (?:could )?\b",
        r"\bi need you to (?:make sure to )?\b",
        r"\bmake sure that you\b",
        r"\bbe sure to\b",
        r"\bas a matter of fact\b",
        r"\bat this point in time\b",
        r"\bin order to\b",
    ]

    # Vague / underspecified indicators
    VAGUE_PATTERNS = [
        r"\b(?:make it good|do it well|make it great|make it cool)\b",
        r"\b(?:tell me useful things|give me useful info|say something interesting)\b",
        r"\b(?:don't miss anything important|include everything important|dont miss anything)\b",
        r"\b(?:do whatever you think is best|you decide|any way you want)\b",
        r"\b(?:build me something|make something)\b",
    ]

    @classmethod
    def analyze(cls, prompt: str) -> TaskAnalysisResult:
        """Perform comprehensive analysis on the prompt."""
        text = prompt.strip()
        lower = text.lower()

        # 1. Detect Category
        category = cls._detect_category(lower)

        # 2. Detect Action & Deliverable Intent (Phase 3.2 Semantic Preservation)
        action, deliverable, action_fam, deliv_fam = cls._detect_action_and_deliverable(lower)

        # 3. Detect Prohibitions (e.g. "Do not use pandas", "no numpy", "without X")
        prohibitions = cls._extract_prohibitions(text)

        # 4. Detect Filenames and specific file references
        filenames = cls._extract_filenames(text)

        # 5. Detect Specific Keys / Field names
        specific_keys = cls._extract_keys(text)

        # 6. Detect Requested Format
        format_requested = cls._extract_format(lower)

        # 7. Detect Required Operations (Phase 3.2.1 Operation Preservation)
        required_operations = cls._extract_required_operations(
            text=text,
            filenames=filenames,
            prohibitions=prohibitions,
            primary_action=action,
            primary_deliverable=deliverable,
        )

        # 8. Detect Filler & Redundancy
        has_filler = any(re.search(pat, text, re.IGNORECASE) for pat in cls.FILLER_PATTERNS)
        has_redundancy = cls._detect_redundancy(lower) or has_filler

        # 9. Detect Underspecification
        is_underspecified = any(re.search(pat, text, re.IGNORECASE) for pat in cls.VAGUE_PATTERNS)

        # 10. Detect if prompt is already effective & concise
        is_already_effective = cls._check_already_effective(
            text=text,
            has_filler=has_filler,
            is_underspecified=is_underspecified,
            format_requested=format_requested,
            specific_keys=specific_keys,
        )

        # Build list of user-facing explicit constraints
        explicit_constraints: List[str] = []
        if deliverable and deliv_fam in {"software", "document", "plan"}:
            explicit_constraints.append(f"Deliverable: {deliverable.capitalize()}")
        for p in prohibitions:
            explicit_constraints.append(f"Prohibition: No {p}")
        for op in required_operations:
            explicit_constraints.append(f"Operation: {op.normalized_phrase.capitalize()}")
        for fn in filenames:
            explicit_constraints.append(f"File reference: {fn}")
        for k in specific_keys:
            explicit_constraints.append(f"Output key: '{k}'")
        if format_requested:
            explicit_constraints.append(f"Format: {format_requested}")

        # Complexity score based on sentence count & distinct clauses
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]
        complexity_score = max(1, len(sentences))

        return TaskAnalysisResult(
            task_category=category,
            primary_action=action,
            primary_deliverable=deliverable,
            action_family=action_fam,
            deliverable_family=deliv_fam,
            required_operations=required_operations,
            explicit_constraints=explicit_constraints,
            prohibitions=prohibitions,
            filenames=filenames,
            specific_keys=specific_keys,
            format_requested=format_requested,
            is_already_effective=is_already_effective,
            is_underspecified=is_underspecified,
            has_redundancy=has_redundancy,
            has_conversational_filler=has_filler,
            complexity_score=complexity_score,
        )

    @classmethod
    def _detect_action_and_deliverable(
        cls, lower: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Identify primary action verb and requested deliverable artifact."""
        action = None
        action_family = None
        deliverable = None
        deliverable_family = None

        words = re.findall(r"\b[a-z_0-9\-]+\b", lower)

        # Find primary action verb in prompt sequence
        for word in words:
            if word in cls.ACTION_TAXONOMY:
                action, action_family = cls.ACTION_TAXONOMY[word]
                break

        # Find primary deliverable noun in prompt sequence
        for word in words:
            if word in cls.DELIVERABLE_TAXONOMY:
                deliverable, deliverable_family = cls.DELIVERABLE_TAXONOMY[word]
                break

        # Software intent alignment: if prompt specifies software artifact (e.g. program, script, function)
        # or has coding verbs, tie together
        if deliverable_family == "software":
            action_family = "software_development"

        # Explanation intent alignment
        if action == "explain" and not deliverable:
            deliverable = "explanation"
            deliverable_family = "explanation"

        # Summarization intent alignment
        if action == "summarize" and not deliverable:
            deliverable = "summary"
            deliverable_family = "document"

        return action, deliverable, action_family, deliverable_family

    @classmethod
    def _detect_category(cls, lower: str) -> str:
        """Detect task domain using weighted keyword analysis."""
        words = set(re.findall(r"\b[a-z_0-9\-]+\b", lower))

        scores = {
            "coding": len(words & cls.CODING_KEYWORDS),
            "data_analysis": len(words & cls.DATA_KEYWORDS),
            "extraction": len(words & cls.EXTRACTION_KEYWORDS),
            "summarization": len(words & cls.SUMMARIZATION_KEYWORDS),
            "writing": sum(1 for phrase in cls.WRITING_KEYWORDS if phrase in lower),
            "research": sum(1 for phrase in cls.RESEARCH_KEYWORDS if phrase in lower),
            "planning": sum(1 for phrase in cls.PLANNING_KEYWORDS if phrase in lower),
        }

        # Handle specific combinations
        if scores["coding"] > 0 and scores["data_analysis"] > 0:
            if any(term in lower for term in ["csv", "revenue", "dataset", "monthly_report"]):
                return "data_analysis"
            return "coding"

        best_category = max(scores, key=scores.get)
        if scores[best_category] > 0:
            return best_category

        return "general"

    @classmethod
    def _extract_prohibitions(cls, text: str) -> List[str]:
        """Extract explicit negative constraints (e.g. 'do not use pandas', 'does not use X', 'no numpy')."""
        prohibitions: Set[str] = set()

        ignored_words = {
            "the", "a", "an", "any", "miss", "anything", "forget", "worry",
            "hesitate", "matter", "stop", "fail", "leave", "drop"
        }

        # Pattern: "do not / does not / don't / never / avoid / without [using/importing] X [or Y]"
        pat1 = (
            r"\b(?:do not|does not|don't|dont|doesn't|never|avoid|without)\s+"
            r"(?:use|using|include|including|import|importing|rely on|relying on)?\s*"
            r"([a-zA-Z0-9_\-]+)(?:\s+or\s+([a-zA-Z0-9_\-]+))?"
        )
        for m in re.finditer(pat1, text, re.IGNORECASE):
            item1 = m.group(1).lower().strip()
            if item1 and item1 not in ignored_words and len(item1) > 1:
                prohibitions.add(item1)
            if m.group(2):
                item2 = m.group(2).lower().strip()
                if item2 and item2 not in ignored_words and len(item2) > 1:
                    prohibitions.add(item2)

        # Pattern: "no pandas", "no numpy"
        pat2 = r"\bno\s+([a-zA-Z0-9_\-]+)\b"
        for m in re.finditer(pat2, text, re.IGNORECASE):
            item = m.group(1).lower().strip()
            if item in {
                "pandas", "numpy", "matplotlib", "external", "third-party",
                "dependencies", "libraries", "scipy", "sklearn", "tensorflow", "torch"
            }:
                prohibitions.add(item)

        return sorted(list(prohibitions))

    @classmethod
    def _extract_filenames(cls, text: str) -> List[str]:
        """Extract exact file names mentioned (e.g. sales.csv, monthly_report.csv, app.log)."""
        pat = r"\b([a-zA-Z0-9_\-]+\.(?:csv|json|py|txt|html|yaml|yml|md|parquet|sql|tsv|xml|pdf|log))\b"
        matches = re.findall(pat, text, re.IGNORECASE)
        # Deduplicate preserving order
        seen = set()
        filenames = []
        for fn in matches:
            if fn.lower() not in seen:
                seen.add(fn.lower())
                filenames.append(fn)
        return filenames

    @classmethod
    def _extract_keys(cls, text: str) -> List[str]:
        """Extract specific key or field names (e.g. key 'result')."""
        keys: Set[str] = set()
        # key 'result' or key "result"
        pat1 = r"\b(?:key|field|attribute)\s+['\"]([a-zA-Z0-9_\-]+)['\"]"
        for m in re.finditer(pat1, text, re.IGNORECASE):
            keys.add(m.group(1))

        # keys: vendor, total_amount, ...
        pat2 = r"\bkeys:\s*([a-zA-Z0-9_\-,\s]+?)(?:\.|\n|$)"
        m2 = re.search(pat2, text, re.IGNORECASE)
        if m2:
            raw_keys = m2.group(1).split(",")
            for k in raw_keys:
                cleaned = k.strip()
                if cleaned and len(cleaned) < 30:
                    keys.add(cleaned)

        return sorted(list(keys))

    @classmethod
    def _extract_format(cls, lower: str) -> Optional[str]:
        """Detect explicit output format if requested."""
        if "json object" in lower or "valid json" in lower or "raw json" in lower:
            return "JSON"
        if "markdown" in lower:
            return "Markdown"
        if "bullet points" in lower or "bulleted list" in lower:
            return "Bullet points"
        if "table format" in lower or "markdown table" in lower:
            return "Table"
        if "csv format" in lower:
            return "CSV"
        return None

    @classmethod
    def _detect_redundancy(cls, lower: str) -> bool:
        """Detect circular or repeated instructions in prompt."""
        # Check repeated phrases or sentences
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", lower) if len(s.strip()) > 15]
        if len(sentences) >= 2:
            # Check for semantic duplicate pairs like "adds two numbers" and "returns their sum"
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    words_i = set(re.findall(r"\b\w+\b", sentences[i]))
                    words_j = set(re.findall(r"\b\w+\b", sentences[j]))
                    if len(words_i & words_j) >= 4:
                        return True

        if "in addition to that" in lower and ("make sure" in lower or "i would like you to" in lower):
            return True

        return False

    @classmethod
    def _check_already_effective(
        cls,
        text: str,
        has_filler: bool,
        is_underspecified: bool,
        format_requested: Optional[str],
        specific_keys: List[str],
    ) -> bool:
        """Identify if a prompt is already concise, unambiguous, and effective."""
        if has_filler or is_underspecified:
            return False

        word_count = len(text.split())
        # If under 25 words, has explicit directive (e.g. "Return the square of...")
        # and has clear format or keys, it is already effective.
        if word_count <= 22 and (format_requested is not None or len(specific_keys) > 0):
            # Check for clear action verb at start
            first_words = text.lower().split()[:2]
            action_verbs = {"return", "calculate", "find", "extract", "get", "compute", "format", "convert"}
            if any(w in action_verbs for w in first_words):
                return True

        return False

    @classmethod
    def _extract_required_operations(
        cls,
        text: str,
        filenames: List[str],
        prohibitions: List[str],
        primary_action: Optional[str] = None,
        primary_deliverable: Optional[str] = None,
    ) -> List[RequiredOperation]:
        """Extract generic required operations, calculations, transformations, and data flows."""
        # 1. Mask out negative prohibitions so their verbs aren't parsed as active operations
        cleaned = text
        for pat in [
            r"\b(?:do not|does not|don't|dont|doesn't|never|avoid|without)\s+(?:use|using|include|including|import|importing)?\s*[a-zA-Z0-9_\-\s]+(?:\.|\n|$|,)",
            r"\bno\s+[a-zA-Z0-9_\-]+\b",
        ]:
            cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

        # Build reverse verb mapping
        verb_map = {}
        for root_key, (can_verb, op_fam, inflections) in cls.OPERATION_VERB_FAMILIES.items():
            for infl in inflections:
                verb_map[infl.lower()] = (can_verb, op_fam)

        verbs_pat = "|".join(re.escape(v) for v in sorted(verb_map.keys(), key=len, reverse=True))

        # 2. Split into candidate segments using conjunctions, relative clauses, commas, semicolons, and list markers
        raw_segments = re.split(
            rf"(?:,\s*(?:and\s+|then\s+)?|\band\s+(?=(?:{verbs_pat})\b)|\bthat\s+|\bwhich\s+|\bto\s+(?=(?:{verbs_pat})\b)|\b(?:then|next|after that|finally)\s+|;|\.\s+|\n+)",
            cleaned,
            flags=re.IGNORECASE,
        )

        operations: List[RequiredOperation] = []
        seen_normalized = set()

        for raw_seg in raw_segments:
            seg = raw_seg.strip()
            # Strip list prefixes like "1.", "-", "*"
            seg = re.sub(r"^(?:\d+[\.\)]|[-*•])\s*", "", seg).strip()
            if not seg or len(seg) < 3:
                continue

            words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", seg)
            if not words:
                continue

            # Identify if the segment begins with or contains an action verb in its first 3 words
            found_verb_idx = -1
            canonical_verb = None
            op_family = None

            for i, w in enumerate(words[:3]):
                w_lower = w.lower()
                if w_lower in verb_map:
                    canonical_verb, op_family = verb_map[w_lower]
                    found_verb_idx = i
                    break

            if found_verb_idx == -1 or canonical_verb is None:
                continue

            # Extract target phrase: everything after the verb word in this segment
            verb_word = words[found_verb_idx]
            match = re.search(rf"\b{re.escape(verb_word)}\b", seg, re.IGNORECASE)
            target_str = seg[match.end():].strip() if match else " ".join(words[found_verb_idx + 1:])

            # Check if this clause is purely the overarching deliverable creation declaration
            # (e.g. "Create a Python program", "Write a Python script")
            if (
                canonical_verb in {"create", "write", "build", "develop", "implement", "code", "generate"}
                and primary_deliverable
                and primary_deliverable.lower() in target_str.lower()
                and not any(fn.lower() in target_str.lower() for fn in filenames)
            ):
                continue

            # Identify associated entities (e.g. filenames) in target
            associated_entities = [fn for fn in filenames if fn.lower() in seg.lower()]

            # Extract key concept terms from target (excluding stopwords)
            target_tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", target_str.lower())
            key_terms = [t for t in target_tokens if t not in cls.STOPWORDS and len(t) > 1]

            if not key_terms and not associated_entities:
                continue

            # Build normalized phrase
            display_terms = [t for t in key_terms if t not in [fn.lower() for fn in associated_entities]]
            if associated_entities and display_terms:
                norm_phrase = f"{canonical_verb} {' '.join(display_terms)} ({', '.join(associated_entities)})"
            elif associated_entities:
                norm_phrase = f"{canonical_verb} {', '.join(associated_entities)}"
            else:
                norm_phrase = f"{canonical_verb} {' '.join(display_terms)}"

            if norm_phrase.lower() in seen_normalized:
                continue
            seen_normalized.add(norm_phrase.lower())

            operations.append(
                RequiredOperation(
                    raw_text=seg,
                    action_verb=canonical_verb,
                    target=target_str,
                    normalized_phrase=norm_phrase,
                    operation_family=op_family,
                    key_terms=[canonical_verb] + key_terms,
                    associated_entities=associated_entities,
                )
            )

        return operations
