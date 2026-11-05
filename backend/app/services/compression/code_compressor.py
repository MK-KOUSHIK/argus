"""Safe, parser-driven code compressor preserving executable behavior and semantics."""

import ast
import re
from typing import List, Optional, Tuple


class CodeCompressor:
    """Compresses source code via AST analysis and safe tokenization without regex-only corruption."""

    ALGORITHMIC_KEYWORDS = re.compile(
        r"\b(lru|ttl|fifo|lifo|eviction|algorithm|policy|complexity|thread-safe|invariant|concurrency|mutex|semaphore|amortized|cache|caching|strategy|heuristic|time complexity|space complexity)\b",
        re.IGNORECASE,
    )
    METADATA_IGNORE = re.compile(
        r"^[#\s=*\-_/]+$|^\s*#\s*(author|updated|last updated|created|date|version|copyright|license)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _extract_algorithmic_comments(cls, code: str) -> List[str]:
        """Extract algorithmically important comments and class/module-level context."""
        preserved = []
        for line in code.splitlines():
            s = line.strip()
            if s.startswith("#"):
                if not cls.METADATA_IGNORE.search(s) and cls.ALGORITHMIC_KEYWORDS.search(s):
                    clean_comment = "# " + s.lstrip("# \t")
                    if clean_comment not in preserved:
                        preserved.append(clean_comment)
        return preserved

    @classmethod
    def compress(cls, code: str, language: Optional[str] = None) -> Tuple[str, bool, str]:
        """Compress code by stripping non-executable comments and formatting while preserving algorithmic context.

        Returns:
            Tuple of (compressed_code, validation_passed, reason).
        """
        trimmed = code.strip()
        if not trimmed:
            return code, True, "Empty input"

        lang = (language or "").lower()

        # 1. Python Code Compression (AST-based)
        if lang in {"py", "python", ""} or cls._is_python(trimmed):
            compressed, ok, reason = cls._compress_python(trimmed)
            if ok:
                return compressed, True, reason
            # If specified as python but failed, fallback to original safely
            if lang in {"py", "python"}:
                return code, False, reason

        # 2. SQL Code Compression
        if lang in {"sql"}:
            return cls._compress_sql(trimmed)

        # 3. JavaScript / TypeScript Compression
        if lang in {"js", "javascript", "ts", "typescript"}:
            return cls._compress_js(trimmed)

        # 4. Unsupported or generic language: preserve source rather than unsafe regex
        return code, True, f"Preserved source verbatim (no safe AST parser for language '{language}')"

    @classmethod
    def _is_python(cls, code: str) -> bool:
        """Check if code can be successfully parsed into a Python AST."""
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    @classmethod
    def _compress_python(cls, code: str) -> Tuple[str, bool, str]:
        """Compress Python code using native AST parse and unparse while preserving algorithmic header comments."""
        try:
            tree_original = ast.parse(code)
        except SyntaxError as exc:
            return code, False, f"Python syntax error on input: {str(exc)}"

        try:
            # ast.unparse removes comments and normalizes whitespace while preserving all AST nodes
            unparsed = ast.unparse(tree_original).strip()
        except Exception as exc:
            return code, False, f"AST unparse failed: {str(exc)}"

        # Preserve class-level architectural headers and algorithmically important comments
        algo_comments = cls._extract_algorithmic_comments(code)
        if algo_comments:
            header_block = "\n".join(algo_comments)
            compressed = f"{header_block}\n\n{unparsed}"
        else:
            compressed = unparsed

        # Validate by parsing the compressed code back into AST
        try:
            tree_compressed = ast.parse(compressed)
            if not cls._compare_ast_structures(tree_original, tree_compressed):
                return code, False, "AST functional structure mismatch after compression"
        except SyntaxError as exc:
            return code, False, f"Compressed Python code failed syntax re-parse: {str(exc)}"

        # Only return compressed if it actually saved space
        if len(compressed) >= len(code):
            return code, True, "Python code already compact"

        reason = "Python AST-based compression passed"
        if algo_comments:
            reason += f" (preserved {len(algo_comments)} algorithmic comment(s))"
        return compressed, True, reason

    @classmethod
    def _compare_ast_structures(cls, tree_a: ast.AST, tree_b: ast.AST) -> bool:
        """Verify essential executable AST nodes (functions, classes, calls, returns) match."""
        nodes_a = [type(n).__name__ for n in ast.walk(tree_a)]
        nodes_b = [type(n).__name__ for n in ast.walk(tree_b)]

        # Counts of functional nodes must match exactly
        for node_type in ["FunctionDef", "AsyncFunctionDef", "ClassDef", "Return", "Yield", "Call", "If", "For"]:
            if nodes_a.count(node_type) != nodes_b.count(node_type):
                return False

        return True

    @classmethod
    def _compress_sql(cls, code: str) -> Tuple[str, bool, str]:
        """Compress SQL statements by stripping comments and normalizing whitespace outside literals."""
        # Split on string literals to avoid corrupting data inside quotes
        parts = re.split(r"('(?:''|[^'])*')", code)
        cleaned_parts = []

        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                # Inside string literal: preserve exactly
                cleaned_parts.append(part)
            else:
                # Outside literal: remove single-line comments (-- ...) and multi-line comments (/* ... */)
                s = re.sub(r"--[^\n]*", "", part)
                s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
                s = re.sub(r"\s+", " ", s)
                cleaned_parts.append(s)

        compressed = "".join(cleaned_parts).strip()
        if not compressed:
            return code, False, "SQL compression produced empty output"

        # Validate basic balance of parens and quotes
        if compressed.count("(") != compressed.count(")") or compressed.count("'") % 2 != 0:
            return code, False, "SQL syntax balance validation failed"

        if len(compressed) >= len(code):
            return code, True, "SQL already compact"

        return compressed, True, "SQL safe formatting passed"

    @classmethod
    def _compress_js(cls, code: str) -> Tuple[str, bool, str]:
        """Compress JavaScript/TypeScript by stripping safe comments and excess blank lines."""
        # Split by string literals and template literals
        pattern = re.compile(r"('(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\"|`(?:\\`|[^`])*`)")
        parts = pattern.split(code)
        cleaned_parts = []

        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                # String literal: preserve verbatim
                cleaned_parts.append(part)
            else:
                # Remove single line comments // ... and multi-line /* ... */
                s = re.sub(r"//[^\n]*", "", part)
                s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
                # Collapse multiple blank lines to at most one
                s = re.sub(r"\n\s*\n+", "\n", s)
                cleaned_parts.append(s)

        compressed = "".join(cleaned_parts).strip()
        # Validate balanced braces and parens
        if (
            compressed.count("{") != compressed.count("}")
            or compressed.count("(") != compressed.count(")")
            or compressed.count("[") != compressed.count("]")
        ):
            return code, False, "JS/TS bracket balance validation failed"

        if len(compressed) >= len(code):
            return code, True, "JS/TS already compact"

        return compressed, True, "JS/TS comment stripping passed"
