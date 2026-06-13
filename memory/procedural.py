from __future__ import annotations
from typing import Literal, Optional
from uuid import UUID

from pydantic import Field, field_validator

from memory.models import BaseMemory, MemoryTypeEnum

class ProceduralMemory(BaseMemory):
    """
    Procedure stored Memory
    """

    memory_type: Literal[MemoryTypeEnum.PROCEDURAL] = (MemoryTypeEnum.PROCEDURAL)
    trigger_condition: str # natural language activation condition
    steps: list[str] # ordered execution steps; min 1
    success_count: int = 0 # CONFIRMED feedback signals
    failure_count: int = 0 # CORRECTION feedback signals
    avg_execution_time_ms: Optional[float] = None
    abstracted_from: list[UUID] = Field(default_factory=list) # source episodic/compressed IDs
    domain: Optional[str] = None # e.g. 'code', 'research'

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, value: list[str]) -> list[str]:

        if not value:
            raise ValueError("steps must contain at least one step")
        cleaned = [s.strip() for s in value]
        if any(not s for s in cleaned):
            raise ValueError("steps cannot contain empty strings")
        return cleaned
    
    @field_validator("success_count")
    @classmethod
    def validate_success_count(cls, value: int) -> int:

        if value < 0:
            raise ValueError("Success count should be non negative number")
        
        return value
    
    @field_validator("failure_count")
    @classmethod
    def validate_failure_count(cls, value: int) -> int:

        if value < 0:
            raise ValueError("Failure count should be non negative number")
        
        return value
    
    @field_validator("avg_execution_time_ms")
    @classmethod
    def validate_avg_execution_time_ms(cls, value: Optional[float]) -> Optional[float]:

        if value < 0:
            raise ValueError("Average execution time should be non negative number")
        
        return value

