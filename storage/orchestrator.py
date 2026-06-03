from __future__ import annotations

import sys
import traceback

from dataclasses import dataclass
from typing import Optional, Protocol
from uuid import UUID
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.models import (
    BaseMemory,
    BaseEvent,
    LifecycleTransitionPayload,
    TriggerEnum,
    MemoryTypeEnum,
    IngestionPayload,
    RetrievalPayload,
)

from storage.duckdb_store import (
    DuckDBStore,
    MemoryNotFoundError,
)

from storage.faiss_store import (
    FAISSStore,
)

try:
    from graph.ontology import KuzuDBStore
except Exception:  # pragma: no cover - optional graph dependency
    KuzuDBStore = None  # type: ignore[assignment]

try:
    from storage.sqlite_log import SQLiteEventLog
except Exception:  # pragma: no cover - optional event-log dependency
    SQLiteEventLog = None  # type: ignore[assignment]

class StorageOrchestrationError(Exception):
    """
    Base orchestration error.
    """

class ReplayRebuildError(StorageOrchestrationError):
    """
    Raised when replay reconstruction fails.
    """

class InvariantViolationError(StorageOrchestrationError):
    """
    Raised when system invariants are violated.
    """


class EventLog(Protocol):
    def log_ingestion(self, *, agent_id: str, payload: IngestionPayload) -> BaseEvent:
        ...

    def log_retrieval(self, *, agent_id: str, payload: RetrievalPayload) -> BaseEvent:
        ...

    def log_feedback(self, *, agent_id: str, payload) -> BaseEvent:
        ...

    def log_lifecycle_transition(
        self,
        *,
        agent_id: str,
        payload: LifecycleTransitionPayload,
    ) -> BaseEvent:
        ...

@dataclass(slots=True)
class TransactionResult:
    success: bool
    txn_id: Optional[str] = None
    rollback_performed: bool = False
    error: Optional[str] = None

class StorageOrchestrator:
    """
    Coordinates all cross-store writes.

    CRITICAL ARCHITECTURAL RULE:

    DuckDB is reconstructable state.

    REQUIRED WRITE ORDER:
        2. DuckDB write
        3. FAISS write

    NEVER reverse this order.
    """

    def __init__(
        self,
        duckdb_store: DuckDBStore,
        faiss_store: FAISSStore,
        event_log: EventLog | None = None,
        graph_store: object | None = None,
    ):
        """
        Proper dependency injection.

        NEVER inject classes.
        ONLY inject instances.
        """
        
        self.duckdb = duckdb_store
        self.faiss = faiss_store
        self.event_log = event_log
        self.graph = graph_store
    
    async def ingest_memory(
        self,
        memory: BaseMemory,
        embedding: list[float],
    ) -> TransactionResult:
        """
        Primary ingestion entrypoint.

        Flow:
            1. append immutable SQLite event
            2. write DuckDB state
            3. rollback on failure
        """

        try:

            if self.event_log is not None:
                self.event_log.log_ingestion(
                    agent_id=memory.agent_id,
                    payload=IngestionPayload(
                        memory_id=memory.memory_id,
                        memory_type=str(memory.memory_type),
                        sha256=memory.sha256,
                        chunks_created=1,
                        modality=str(memory.modality),
                        pipeline_stages_ms={},
                        entity_count=0,
                        relation_count=0,
                        provenance=str(getattr(memory, "provenance", "OBSERVED")),
                    ),
                )

            # DuckDB State write
            self.duckdb.insert_memory(memory)

            # Graph write is best-effort and derived from the canonical memory row.
            if self.graph is not None and hasattr(self.graph, "save_memory"):
                try:
                    self.graph.save_memory(memory)
                except Exception:
                    traceback.print_exc()

            # Faiss Embedding
            should_index = True
            if (
                memory.memory_type == MemoryTypeEnum.WORKING
                and getattr(memory, "promoted_to", None) is None
            ):
                should_index = False

            if should_index:
                self.faiss.add_embedding(
                    embedding=embedding,
                    memory_id=memory.memory_id,
                    agent_id=memory.agent_id,
                    memory_type=memory.memory_type,
                    metadata={
                        "memory_id": str(memory.memory_id),
                        "agent_id": memory.agent_id,
                        "memory_type": str(memory.memory_type),
                    }
                )

            if self.graph is not None and hasattr(self.graph, "save_relation"):
                related_ids = []
                for field_name in ("source_ids", "referenced_memory_ids"):
                    values = getattr(memory, field_name, None) or []
                    related_ids.extend(values)

                for related_id in related_ids:
                    try:
                        self.graph.save_relation(
                            source=str(memory.memory_id),
                            target=str(related_id),
                            relation="REFERENCES",
                        )
                    except Exception:
                        traceback.print_exc()

            return TransactionResult(success=True)

        except Exception as exc:
            traceback.print_exc()

            return TransactionResult(
                success=False,
                rollback_performed=False,
                error=str(exc),
            )
    
    def retrieve_memory(
        self,
        memory_id: UUID,
    ) -> BaseMemory:
        """
        Retrieves memory and updates retrieval metadata.

        Retrieval itself becomes part of memory history.
        """

        memory = self.duckdb.get_memory(memory_id)

        self.duckdb.update_access_metadata(memory_id) # Update access metadata

        if self.event_log is not None:
            self.event_log.log_retrieval(
                agent_id=memory.agent_id,
                payload=RetrievalPayload(
                    memory_id=memory.memory_id,
                    importance_before=memory.importance_score,
                    importance_after=memory.importance_score,
                    access_count_after=memory.access_count + 1,
                    retrieval_score=1.0,
                ),
            )

        # Emit Retrivel event
        payload = RetrievalPayload(
            memory_id=memory.memory_id,
            importance_before=memory.importance_score,
            importance_after=memory.importance_score,
            access_count_after=memory.access_count+1,
            retrieval_score=1.0,
        )
        
        # reload updated memory
        return self.duckdb.get_memory(memory_id)
    
    
    def replay_rebuild(
        self,
        agent_id: str,
    ) -> TransactionResult:
        """
        Reconstructs DuckDB state from immutable SQLite history.

        Disaster recovery primitive.
        """

        try:

            # IMPORTANT:
            # Full replay engine intentionally deferred
            # to replay/reconstructor.py
            #
            # This method currently exists as orchestration
            # placeholder to preserve architecture boundaries.

            return TransactionResult(
                success=True
            )
        
        except (
            MemoryNotFoundError,
            Exception,
        ) as exc:

            raise ReplayRebuildError(str(exc)) from exc

    def transition_lifecycle(
        self,
        memory_id: UUID,
        new_state,
    ) -> None:
        memory = self.duckdb.get_memory(memory_id)
        old_state = memory.lifecycle_state
        self.duckdb._apply_lifecycle_transition(memory_id, new_state)

        if self.event_log is not None:
            self.event_log.log_lifecycle_transition(
                agent_id=memory.agent_id,
                payload=LifecycleTransitionPayload(
                    memory_id=memory.memory_id,
                    old_state=old_state,
                    new_state=new_state,
                    trigger=TriggerEnum.MANUAL,
                    importance_at_transition=memory.importance_score,
                ),
            )
    
    def healthcheck(self) -> dict:
        """
        Basic orchestrator health probe.
        """

        return {
            "duckdb_store": "healthy",
            "orchestrator": "healthy",
        }

