from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from memoryos.memory.models import BaseMemory

DEFAULT_IMPORTANCE = 0.5
MIN_IMPORTANCE = 0.0
MAX_IMPORTANCE = 1.0
DEFAULT_REINFORCEMENT = 0.05
DEFAULT_DECAY_RATE = 0.015

@dataclass(slots=True)
class ImportanceUpdate:
    """
    Importance score transition result.
    """

    previous_score: float
    updated_score: float
    delta: float
    updated_at: datetime

class ImportanceScorer:
    """
    Deterministic importance scoring engine.

    Responsibilities:
    - reinforcement scoring
    - temporal decay
    - retrieval weighting
    - bounded normalization

    Does NOT:
    - perform retrieval
    - access storage
    - mutate persistence directly
    """

    def __init__(
        self,
        *,
        reinforcement_factor: float = DEFAULT_REINFORCEMENT,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ):
        
        self.reinforcement_factor = reinforcement_factor
        self.decay_rate = decay_rate

    @staticmethod
    def clamp(
        value: float,
    ) -> float:
        """
        Constrains importance score
        to valid range.
        """

        return max(
            MIN_IMPORTANCE,
            min(
                MAX_IMPORTANCE,
                value,
            ),
        )

    def reinforce(
        self,
        memory: BaseMemory,
        *,
        reinforcement: float | None = None,
    ) -> ImportanceUpdate:
        """
        Reinforces memory importance.

        Called after:
        - retrieval
        - explicit feedback
        - repeated access
        """

        previous = memory.importance_score

        delta = (
            reinforcement
            if reinforcement
            is not None
            else self.reinforcement_factor
        )

        updated = self.clamp(previous + delta)

        memory.importance_score = updated
        memory.last_accessed_at = datetime.now(timezone.utc)

        return ImportanceUpdate(
            previous_score=previous,
            updated_score=updated,
            delta=updated - previous,
            updated_at=memory.last_accessed_at,
        )

    def decay(
        self,
        memory: BaseMemory,
        *,
        now: (
            datetime | None
        ) = None,
    ) -> ImportanceUpdate:
        """
        Applies exponential temporal decay.
        """

        now = (
            now
            or datetime.now(
                timezone.utc
            )
        )

        previous = memory.importance_score

        last_access = (
            memory.last_accessed_at
            or memory.created_at
        )

        elapsed_hours = max(
            (
                now - last_access
            ).total_seconds()
            / 3600,
            0.0,
        )

        decay_multiplier = math.exp(
            -self.decay_rate
            * elapsed_hours
        )

        updated = self.clamp(previous * decay_multiplier)

        memory.importance_score = updated

        return ImportanceUpdate(
            previous_score=previous,
            updated_score=updated,
            delta=updated - previous,
            updated_at=now,
        )

    @staticmethod
    def retrieval_weight(
        memory: BaseMemory,
    ) -> float:
        """
        Converts importance into
        retrieval weighting factor.
        """

        return 1.0 + memory.importance_score

    @staticmethod
    def initial_importance(
        *,
        explicit_score: (
            float | None
        ) = None,
    ) -> float:
        """
        Initial memory importance.
        """

        if explicit_score is None:
            return DEFAULT_IMPORTANCE

        return max(
            MIN_IMPORTANCE,
            min(
                MAX_IMPORTANCE,
                explicit_score,
            ),
        )

