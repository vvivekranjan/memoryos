from __future__ import annotations

import math
from datetime import (
    datetime,
    timezone,
)

DEFAULT_DECAY_RATE = 0.015
DEFAULT_IMPORTANCE_WEIGHT = 0.15
DEFAULT_RECENCY_WEIGHT = 0.10
DEFAULT_SALIENCE_WEIGHT = 0.20
MIN_SCORE = 0.0
MAX_SCORE = 1.0

def clamp_score(
    value: float,
) -> float:
    """
    Bounds score into valid range.
    """

    return max(
        MIN_SCORE,
        min(
            MAX_SCORE,
            value,
        ),
    )

def exponential_decay(
    *,
    age_seconds: float,
    decay_rate: float = DEFAULT_DECAY_RATE,
) -> float:
    """
    Exponential temporal decay.

    Returns:
    1.0 -> fresh
    approaching 0.0 -> stale
    """

    age_hours = max(
        age_seconds / 3600,
        0.0,
    )

    return math.exp(
        -decay_rate
        * age_hours
    )

def recency_score(
    *,
    timestamp: datetime,
    now: datetime | None = None,
    weight: float = DEFAULT_RECENCY_WEIGHT,
) -> float:
    """
    Time-aware retrieval boost.
    """

    now = (
        now
        or datetime.now(
            timezone.utc
        )
    )

    # avoid naive/aware datetime subtraction errors
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(
            tzinfo=timezone.utc
        )

    if now.tzinfo is None:
        now = now.replace(
            tzinfo=timezone.utc
        )

    age_seconds = (
        now - timestamp
    ).total_seconds()

    decay = exponential_decay(age_seconds=age_seconds)

    return decay * weight

def importance_score(
    *,
    importance: float,
    weight: float = DEFAULT_IMPORTANCE_WEIGHT,
) -> float:
    """
    Importance weighting helper.
    """

    return (
        clamp_score(
            importance
        )
        * weight
    )

def salience_score(
    *,
    semantic_similarity: float,
    importance: float,
    recency: float,
    semantic_weight: float = 0.65,
    importance_weight: float = DEFAULT_IMPORTANCE_WEIGHT,
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
) -> float:
    """
    Composite retrieval salience.

    Formula:
    semantic
    + importance
    + recency
    """

    score = (
        clamp_score(
            semantic_similarity
        )
        * semantic_weight
    )

    score += (
        clamp_score(importance)
        * importance_weight
    )

    score += (
        clamp_score(recency)
        * recency_weight
    )

    return clamp_score(score)

def reinforce_importance(
    *,
    current_score: float,
    reinforcement: float = 0.05,
) -> float:
    """
    Retrieval reinforcement update.
    """

    updated = current_score + reinforcement

    return clamp_score(updated)

def decay_importance(
    *,
    current_score: float,
    age_seconds: float,
    decay_rate: float = (
        DEFAULT_DECAY_RATE
    ),
) -> float:
    """
    Time-aware importance decay.
    """

    decay = exponential_decay(
        age_seconds=age_seconds,
        decay_rate=decay_rate,
    )

    return clamp_score(
        current_score
        * decay
    )

def token_overlap_score(
    *,
    query: str,
    content: str,
    token_weight: float = 0.03,
) -> float:
    """
    Lightweight lexical overlap helper.

    Supplemental only.
    Never primary ranking signal.
    """

    query_tokens = {
        token
        for token
        in query.lower().split()
        if len(token) > 2
    }

    content_tokens = set(
        content.lower().split()
    )

    overlap = len(
        query_tokens
        & content_tokens
    )

    return overlap * token_weight

def normalize_similarity(
    similarity: float,
) -> float:
    """
    Converts cosine range:
    [-1, 1] -> [0, 1]
    """

    normalized = (
        similarity + 1.0
    ) / 2.0

    return clamp_score(normalized)

def reciprocal_rank_fusion(
    *,
    rank: int,
    k: int = 60,
) -> float:
    """
    Reciprocal Rank Fusion helper.

    Used in M2 hybrid retrieval.
    """

    if rank <= 0:
        raise ValueError(
            "rank must be greater than 0"
        )

    if k < 0:
        raise ValueError(
            "k must be >= 0"
        )

    return 1.0 / (
        k + rank
    )

