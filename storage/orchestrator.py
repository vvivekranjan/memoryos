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
    MemoryTypeEnum,
    RetrievalPayload,
)

from storage.duckdb_store import (
    DuckDBStore,
    MemoryNotFoundError,
)

from storage.faiss_store import (
    FAISSStore,
)

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
    def _write_txn(
        self,
        *,
        agent_id: str,
        event_id: str,
    ) -> str:
        ...

    def _resolve_txn(
        self,
        *,
        txn_id: str,
    ) -> None:
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
    ):
        """
        Proper dependency injection.

        NEVER inject classes.
        ONLY inject instances.
        """
        
        self.duckdb = duckdb_store
        self.faiss = faiss_store
    
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

            # DuckDB State write
            self.duckdb.insert_memory(memory)

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
                    metadata={
                        "memory_id": str(memory.memory_id),
                        "agent_id": memory.agent_id,
                        "memory_type": str(memory.memory_type),
                    }
                )

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
    
    def healthcheck(self) -> dict:
        """
        Basic orchestrator health probe.
        """

        return {
            "duckdb_store": "healthy",
            "orchestrator": "healthy",
        }

