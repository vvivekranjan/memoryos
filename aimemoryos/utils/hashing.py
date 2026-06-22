from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

def normalize_text(
    text: str,
) -> str:
    """
    Stable text normalization.

    Used before:
    - fingerprinting
    - deduplication
    - embedding cache lookup
    """

    return " ".join(
        text.strip()
        .lower()
        .split()
    )


def sha256_text(
    text: str,
) -> str:
    """
    Stable SHA-256 fingerprint
    for textual content.
    """

    normalized = normalize_text(text).encode("utf-8")

    return hashlib.sha256(
        normalized
    ).hexdigest()

def sha256_json(
    payload: dict[str, Any],
) -> str:
    """
    Deterministic JSON fingerprint.

    Used for:
    - replay integrity
    - metadata fingerprints
    - deterministic cache keys
    """

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        serialized
    ).hexdigest()

def content_fingerprint(
    *,
    content: str,
    metadata: (
        dict[str, Any]
        | None
    ) = None,
) -> str:
    """
    Stable deduplication fingerprint.

    Combines:
    - normalized content
    - deterministic metadata
    """

    normalized_content = normalize_text(content)

    metadata_hash = (
        sha256_json(
            metadata or {}
        )
    )

    combined = (
        f"{normalized_content}"
        f"::{metadata_hash}"
    )

    return hashlib.sha256(
        combined.encode("utf-8")
    ).hexdigest()

def sha256_bytes(
    data: bytes,
) -> str:
    """
    Raw byte fingerprint.
    """

    return hashlib.sha256(
        data
    ).hexdigest()

def verify_hash(
    *,
    value: str,
    expected_hash: str,
) -> bool:
    """
    Constant deterministic hash verification.
    """

    actual = sha256_text(value)

    # compare_digest avoids timing side-channels from string comparison
    return hmac.compare_digest(
        actual,
        expected_hash,
    )

def short_hash(
    value: str,
    *,
    length: int = 12,
) -> str:
    """
    Human-readable short fingerprint.
    """

    if length <= 0:
        raise ValueError(
            "length must be greater than 0"
        )

    if length > 64:
        raise ValueError(
            "length must be <= 64"
        )

    return sha256_text(
        value
    )[:length]

