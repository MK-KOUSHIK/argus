"""Thread-safe, TTL-bounded original content registry for reversible compression."""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RegistryEntry:
    """Entry stored in the compression registry."""

    ref_key: str
    content: str
    created_at: float
    expires_at: float
    size_bytes: int


class CompressionRegistry:
    """In-memory LRU/TTL original content store for exact round-trip retrieval."""

    _lock = threading.Lock()
    _entries: OrderedDict[str, RegistryEntry] = OrderedDict()

    @classmethod
    def register(cls, content: str, ttl_seconds: Optional[int] = None) -> str:
        """Store original content and return a deterministic, unique reference key."""
        ttl = ttl_seconds if ttl_seconds is not None else settings.COMPRESSION_TTL_SECONDS
        max_entries = settings.COMPRESSION_REGISTRY_MAX_ENTRIES

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        ref_key = f"argus:ref:{content_hash}"

        now = time.time()
        expires_at = now + ttl
        size_bytes = len(content.encode("utf-8"))

        with cls._lock:
            cls._evict_expired_locked(now)

            # If entry already exists, update expiration and move to end (MRU)
            if ref_key in cls._entries:
                cls._entries.move_to_end(ref_key)
                cls._entries[ref_key].expires_at = expires_at
            else:
                # Enforce max entries bound by evicting oldest (LRU)
                while len(cls._entries) >= max_entries and cls._entries:
                    evicted_key, evicted_entry = cls._entries.popitem(last=False)
                    logger.debug(
                        "Evicted LRU compression registry entry: %s (%d bytes)",
                        evicted_key,
                        evicted_entry.size_bytes,
                    )

                entry = RegistryEntry(
                    ref_key=ref_key,
                    content=content,
                    created_at=now,
                    expires_at=expires_at,
                    size_bytes=size_bytes,
                )
                cls._entries[ref_key] = entry
                cls._entries.move_to_end(ref_key)

        logger.info(
            "Registered original content: ref=%s, bytes=%d, ttl=%ds",
            ref_key,
            size_bytes,
            ttl,
        )
        return ref_key

    @classmethod
    def retrieve(cls, ref_key: str) -> Optional[str]:
        """Retrieve exact original content by reference key, or None if missing or expired."""
        now = time.time()
        with cls._lock:
            cls._evict_expired_locked(now)
            entry = cls._entries.get(ref_key)
            if entry is None:
                logger.warning("Compression registry reference not found or expired: %s", ref_key)
                return None

            if entry.expires_at <= now:
                cls._entries.pop(ref_key, None)
                logger.warning("Compression registry reference expired: %s", ref_key)
                return None

            # Mark as recently used
            cls._entries.move_to_end(ref_key)
            return entry.content

    @classmethod
    def contains(cls, ref_key: str) -> bool:
        """Check if an active reference key exists in the registry."""
        return cls.retrieve(ref_key) is not None

    @classmethod
    def verify_restoration(cls, ref_key: str, expected_content: str) -> Dict[str, Any]:
        """Verify that retrieved content exactly matches expected content byte-for-byte and by SHA-256 hash."""
        expected_bytes = expected_content.encode("utf-8")
        expected_hash = hashlib.sha256(expected_bytes).hexdigest()

        retrieved = cls.retrieve(ref_key)
        if retrieved is None:
            return {
                "verified": False,
                "status": "missing_or_expired",
                "expected_hash": expected_hash,
                "actual_hash": None,
                "exact_match": False,
                "expected_bytes": len(expected_bytes),
                "actual_bytes": None,
            }

        retrieved_bytes = retrieved.encode("utf-8")
        actual_hash = hashlib.sha256(retrieved_bytes).hexdigest()
        exact_match = (retrieved_bytes == expected_bytes)

        status = "matched" if exact_match and (actual_hash == expected_hash) else "mismatch"
        return {
            "verified": exact_match and (actual_hash == expected_hash),
            "status": status,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "exact_match": exact_match,
            "expected_bytes": len(expected_bytes),
            "actual_bytes": len(retrieved_bytes),
        }

    @classmethod
    def get_stats(cls) -> Dict[str, int]:
        """Return current registry utilization statistics."""
        with cls._lock:
            cls._evict_expired_locked(time.time())
            total_bytes = sum(e.size_bytes for e in cls._entries.values())
            return {
                "active_entries": len(cls._entries),
                "total_stored_bytes": total_bytes,
                "max_entries": settings.COMPRESSION_REGISTRY_MAX_ENTRIES,
            }

    @classmethod
    def clear(cls) -> None:
        """Flush all registered entries (primarily for test isolation)."""
        with cls._lock:
            cls._entries.clear()

    @classmethod
    def _evict_expired_locked(cls, current_time: float) -> None:
        """Evict all expired entries (must be called with _lock acquired)."""
        expired_keys = [k for k, v in cls._entries.items() if v.expires_at <= current_time]
        for k in expired_keys:
            cls._entries.pop(k, None)

