from __future__ import annotations
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid import UUID, uuid4
from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Any, Literal

import re

class MemoryTypeEnum(Enum):
    EPISODIC = "EPISODIC"
    WORKING = "WORKING"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"

class LifecycleStateEnum(str, Enum):
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    ARCHIVED = "ARCHIVED"
    PRUNED = "PRUNED"

class ModalityEnum(str, Enum):
    TEXT = "TEXT"
    DOC = "DOC"

class ProvenanceEnum(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    HYPOTHESISED = "HYPOTHESISED"
    IMAGINED = "IMAGINED"
    RECEIVED_TRUSTED = "RECEIVED_TRUSTED"
    RECEIVED_UNTRUSTED = "RECEIVED_UNTRUSTED"

class SpeakerRoleEnum(str, Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"
    TOOL = "TOOL"

class EventTypeEnum(str, Enum):
    MEMORY_INGESTED = "MEMORY_INGESTED"
    MEMORY_RETRIEVED = "MEMORY_RETRIEVED"
    FEEDBACK_RECEIVED = "FEEDBACK_RECEIVED"
    LIFECYCLE_TRANSITION = "LIFECYCLE_TRANSITION"

class FeedbackTypeEnum(str, Enum):
    CONFIRMED = "CONFIRMED"
    CORRECTION = "CORRECTION"
    IGNORED = "IGNORED"

class TriggerEnum(str, Enum):
    DECAY_SCHEDULAR = "DECAY_SCHEDULAR"
    REFLECTION_MERGE = "REFLECTION_MERGE"
    MANUAL = "MANUAL"
    COMPRESSION = "COMPRESSION"

class BaseMemory(BaseModel):
    """
    Root schema for all memory types.

    MemoryOS v1.0
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )

    # Identity
    memory_id: UUID = Field(default_factory=uuid4)
    schema_version: str = "1.0.0"
    memory_type: MemoryTypeEnum
    agent_id: str

    # Content
    content: str # normalised UTF-8 text
    sha256: str # SHA-256 of normalised content
    modality: ModalityEnum = ModalityEnum.TEXT

    # Lifecycle
    lifecycle_state: LifecycleStateEnum = LifecycleStateEnum.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) # updated on every retrieval
    access_count: int = 0 # monotonically increasing
    decay_anchor: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) # reset on access; used by decay functions
    decay_multiplier: float = 1.0 # 1.0=normal; >1.0=interference-accelerated
    forward_ref: Optional[UUID] = None

    # Importance
    importance_score: float # [0.0, 1.0]; recomputed on retrieval
    salience_score: float = 0.5 # static salience from ingestion context

    # Affect (Stub until M6)
    vad_v: Optional[float] = None # valence   [-1, 1]
    vad_a: Optional[float] = None # arousal   [-1, 1]
    vad_d: Optional[float] = None # dominance [-1, 1]

    emotional_class: Optional[str] = None # GoEmotions 28-class label

    # Provenance
    provenance: ProvenanceEnum = ProvenanceEnum.OBSERVED
    provenance_confidence: float = 1.0 # [0.0, 1.0]

    source_ids: list[UUID] = Field(default_factory=list) # parent memory IDs

    # Graph
    graph_node_id: Optional[str] = None # KuzuDB node identifier

    # Extensible
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict) # schema_version-gated bag

    # Validators
    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("agent_id cannot be empty")

        return value
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("content cannot be empty")

        return value
    
    @field_validator('sha256')
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        value = value.lower().strip()

        if not re.fullmatch(r"^[a-f0-9]{64}$", value):
            raise ValueError("sha256 must be exactly 64 lowercase hex characters")
        
        return value
    
    @field_validator("importance_score")
    @classmethod
    def validate_importance_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "importance_score must be between 0.0 and 1.0"
            )

        return value
    
    @field_validator("salience_score")
    @classmethod
    def validate_salience_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "salience_score must be between 0.0 and 1.0"
            )

        return value
    
    @field_validator("provenance_confidence")
    @classmethod
    def validate_provenance_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "provenance_confidence must be between 0.0 and 1.0"
            )

        return value
    
    @field_validator("decay_multiplier")
    @classmethod
    def validate_decay_multiplier(cls, value: float) -> float:
        if value < 1.0:
            raise ValueError(
                "decay_multiplier must be >= 1.0"
            )

        return value
    
    @field_validator("access_count")
    @classmethod
    def validate_access_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                "access_count cannot be negative"
            )

        return value
    
    @field_validator("vad_v", "vad_a", "vad_d")
    @classmethod
    def validate_vad_range(
        cls,
        value: Optional[float],
    ) -> Optional[float]:
        if value is None:
            return value

        if value < -1.0:
            return -1.0

        if value > 1.0:
            return 1.0

        return value

    @field_validator("modality")
    @classmethod
    def validate_modality(cls, value: ModalityEnum) -> ModalityEnum:
        return value

    @field_validator("lifecycle_state")
    @classmethod
    def validate_lifecycle_state(cls, value: LifecycleStateEnum) -> LifecycleStateEnum:
        return value

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, value: ProvenanceEnum) -> ProvenanceEnum:
        return value

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        import re
        if not re.fullmatch(r"^\d+\.\d+\.\d+$", value):
            raise ValueError("schema_version must follow semantic versioning (e.g. 1.0.0)")
        return value

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

class WorkingMemory(BaseMemory):
    """
    Session-scoped scratch memory.

    WorkingMemory MUST NEVER be written to FAISS
    unless promoted_to is set.

    Enforcement occurs in:
        storage/orchestrator.py

    Not in the schema layer.
    """

    memory_type: Literal[MemoryTypeEnum.WORKING] = (MemoryTypeEnum.WORKING)
    session_id: UUID
    ttl_seconds: int = 3600 # cleared at session end or TTL expiry
    promoted_to: Optional[UUID] = None # set on explicit promotion
    scratch_data: dict[str, Any] = Field(default_factory=dict) # unstructured in-session state
    expires_at: datetime | None = None # created_at + ttl_seconds

    @field_validator("ttl_seconds")
    @classmethod
    def validate_ttl_seconds(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("ttl_seconds must be > 0")

        return value
    
    def model_post_init(self, __context: Any) -> None:
        if self.expires_at is None:
            self.expires_at = (
                self.created_at
                + timedelta(seconds=self.ttl_seconds)
            )

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

        if not value or '0' in value:
            raise ValueError("Value cannot be empty or atleast minimum step is 1")
        
        return value
    
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

class BaseEvent(BaseModel):
    """
    Immutable append-only event.

    SQLite is the single source of truth.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

    event_id: UUID = Field(default_factory=uuid4)
    schema_version: str = "1.0.0"
    event_type: EventTypeEnum
    agent_id: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any]
    checksum: str  # SHA-256(event_id + schema_version + event_type + occurred_at + sequence_num + payload)
    sequence_num: int # monotonically increasing per agent_id

    @field_validator("sequence_num")
    @classmethod
    def validate_sequence_num(cls, value: int) -> int:
        if value < 0:
            raise ValueError("sequence_num cannot be negative")
        
        return value

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: UUID) -> UUID:
        if not isinstance(value, UUID):
            raise ValueError("event_id must be a UUID")
        return value

    @field_validator("agent_id")
    @classmethod
    def validate_base_agent_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("agent_id cannot be empty")
        return value

    @field_validator("payload")
    @classmethod
    def validate_base_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("payload must be a dictionary")
        return value

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        import re
        if not re.fullmatch(r"^\d+\.\d+\.\d+$", value):
            raise ValueError("schema_version must follow semantic versioning (e.g. 1.0.0)")
        return value

class IngestionPayload(BaseModel):

    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    memory_type: str
    sha256: str
    chunks_created: int
    modality: str
    pipeline_stages_ms: dict[str, float] # stage_name -> latency
    entity_count: int = 0
    relation_count: int = 0
    emotion_class: Optional[str] = None
    provenance: str = "OBSERVED"

    @field_validator("memory_type")
    @classmethod
    def validate_memory_type(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in MemoryTypeEnum._value2member_map_:
            raise ValueError(f"memory_type must be one of {list(MemoryTypeEnum._value2member_map_.keys())}")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        import re
        value = value.lower().strip()
        if not re.fullmatch(r"^[a-f0-9]{64}$", value):
            raise ValueError("sha256 must be exactly 64 lowercase hex characters")
        return value

    @field_validator("chunks_created")
    @classmethod
    def validate_chunks_created(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunks_created must be >= 0")
        return value

    @field_validator("entity_count", "relation_count")
    @classmethod
    def validate_non_negative_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("count must be >= 0")
        return value

    @field_validator("pipeline_stages_ms")
    @classmethod
    def validate_pipeline_stages(cls, value: dict[str, float]) -> dict[str, float]:
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("pipeline_stages_ms keys must be strings")
            if v < 0:
                raise ValueError("pipeline_stages_ms values must be >= 0")
        return value

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, value: str) -> str:
        return value.strip().upper()

class RetrievalPayload(BaseModel):
    
    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    importance_before: float
    importance_after: float
    access_count_after: int
    retrieval_score: float

class FeedbackPayload(BaseModel):

    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    feedback_type: FeedbackTypeEnum
    feedback_text: Optional[str] = None
    submitted_by: Optional[str] = None

class LifecycleTransitionPayload(BaseModel):

    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    old_state: LifecycleStateEnum
    new_state: LifecycleStateEnum
    trigger: TriggerEnum
    importance_at_transition: float

class SessionScope(BaseModel):
    model_config = ConfigDict(strict=True)
    
    agent_id: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    working_memory_ids: List[str] = Field(default_factory=list)
    spreading_activation_seeds: List[str] = Field(default_factory=list)

class RetrieverContribution(BaseModel):
    model_config = ConfigDict(strict=True)
    
    retriever: str
    rank: int
    raw_score: float
    rrf_contribution: float

class RetrievalTrace(BaseModel):
    model_config = ConfigDict(strict=True)
    
    memory_id: str
    retrieved_by: List[RetrieverContribution]
    final_score: float
    importance_score: float
    recency_boost: float
    activation_boost: float
    graph_path: Optional[List[str]] = None
    provenance_tag: str
    conflict_flag: bool
    timestamp_retrieved: datetime

class RetrievalCandidate(BaseModel):
    model_config = ConfigDict(strict=True)
    
    memory_id: str
    content: str
    memory_type: MemoryTypeEnum
    final_score: float
    trace: RetrievalTrace

class RetrievalResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    
    results: List[RetrievalCandidate]
    query_id: str
    latency_ms: float
    cache_hit: bool
    retrievers_active: List[str]
    semantic_cache_checked: bool

# Utilites
def utc_now() -> datetime:
    return datetime.now(timezone.utc)
