from __future__ import annotations
from typing import Literal, Optional
from uuid import UUID

from pydantic import Field, field_validator

from memoryos.memory.models import BaseMemory, MemoryTypeEnum

class SemanticMemory(BaseMemory):
    """
    Memory stored with semantic relation
    """

    memory_type: Literal[MemoryTypeEnum.SEMANTIC] = (MemoryTypeEnum.SEMANTIC)
    entity: str # normalised entity label (subject)
    relation: str # predicate string (e.g. 'works_at')
    object_value: str # object of SPO triple
    confidence: float # extraction confidence [0.0, 1.0]
    entity_type: Optional[str] = None # PERSON|ORG|PLACE|CONCEPT|EVENT
    object_type: Optional[str] = None
    source_url: Optional[str] = None # origin document URL or path
    contradicted_by: list[UUID] = Field(default_factory=list) # IDs of conflicting SemanticMemories
    promoted_from: Optional[UUID] = None # if promoted from hypothesisMemory

    @field_validator("entity")
    @classmethod
    def validate_entity(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Entity cannot be empty")
        
        return value
    
    @field_validator("relation")
    @classmethod
    def validate_relation(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Relation cannot be empty")
        
        return value
    
    @field_validator("object_value")
    @classmethod
    def validate_object_value(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Object value cannot be empty")
        
        return value
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value is None:
            return value

        if value < -1.0:
            return -1.0

        if value > 1.0:
            return 1.0

        return value
