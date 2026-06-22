from __future__ import annotations
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import Field, field_validator

from aimemoryos.memory.models import BaseMemory, MemoryTypeEnum, SpeakerRoleEnum

class EpisodicMemory(BaseMemory):
    """
    Canonical replay memory.

    Used for:
    - reconstruct_state_at()
    - session replay
    - conversational chronology
    - reflection clustering
    """

    memory_type: Literal[MemoryTypeEnum.EPISODIC] = (MemoryTypeEnum.EPISODIC)
    session_id: UUID # groups turns in one session
    turn_index: int # 0-based; ordered within session
    speaker_role: SpeakerRoleEnum
    referenced_memory_ids: list[UUID] = Field(default_factory=list) # explicit cross-refs in this turn
    emotional_snapshot: Optional[dict[str, Any]] = None
    is_system_message: bool = False
    tool_call_id: Optional[str] = None # if speaker_role=TOOL

    @field_validator("turn_index")
    @classmethod
    def validate_turn_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("turn_index cannot be negative")
        
        return value

