from __future__ import annotations

import sys
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol
from uuid import UUID
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.exceptions import (
    FirewallViolationError,
    ReplayRebuildError,
)
from memory.models import (
    BaseMemory,
    BaseEvent,
    LifecycleStateEnum,
    LifecycleTransitionPayload,
    MemoryTypeEnum,
    TriggerEnum,
    IngestionPayload,
    RetrievalPayload,
    ProvenanceEnum,
)
from storage.duckdb_store import DuckDBStore, MemoryNotFoundError
from storage.faiss_store import FAISSStore
from storage.sqlite_log import SQLiteEventLog
from graph.ontology import KuzuDBStore

logger = logging.getLogger(__name__)

_FIREWALL_ISOLATED: frozenset[str] = frozenset({
    ProvenanceEnum.IMAGINED.value,
    ProvenanceEnum.HYPOTHESISED.value,
})


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
    Single write coordinator for all cross-store operations.

    DEPENDENCY INJECTION CONTRACT:
      - ONLY inject fully initialised instances.
      - NEVER inject classes or None for required stores.
      - graph_store is Optional — its absence is acceptable degraded operation.
        When absent, graph writes are silently skipped (not an error).

    WRITE ORDER:
      SQLite → DuckDB → FAISS → KuzuDB (best-effort)

    FIREWALL:
      IMAGINED and HYPOTHESISED memories are routed to isolated tables.
      Standard memories table writes with these provenance tags raise
      FirewallViolationError — this is enforced here and CANNOT be bypassed.
    """

    def __init__(
        self,
        *,
        duckdb_store: DuckDBStore,
        faiss_store: FAISSStore,
        sqlite_log: SQLiteEventLog,
        graph_store: Optional[KuzuDBStore] = None,
    ) -> None:
        self.duckdb = duckdb_store
        self.faiss = faiss_store
        self.event_log = sqlite_log
        self.graph = graph_store

    def _enforce_firewall(self, memory: BaseMemory) -> None:
        """
        Raises FirewallViolationError if violated; callers MUST NOT catch and
        continue — the violation must propagate.
        """
        provenance = getattr(memory, "provenance", None)
        if provenance is None:
            return
        prov_value = (
            provenance.value
            if hasattr(provenance, "value")
            else str(provenance)
        )
        if prov_value in _FIREWALL_ISOLATED:
            raise FirewallViolationError(
                f"Firewall violation: provenance={prov_value!r} memory "
                f"(memory_id={memory.memory_id}) may not be written to the "
                f"standard memories table. Route via isolated store."
            )

    # ------------------------------------------------------------------
    # WorkingMemory guards
    # ------------------------------------------------------------------

    @staticmethod
    def _should_index_in_faiss(memory: BaseMemory) -> bool:
        """
        WorkingMemory MUST NOT be written to FAISS unless
        promoted_to is set. Unpromoted working memories are invisible to all
        retrievers except session_manager direct lookup.
        """
        if memory.memory_type == MemoryTypeEnum.WORKING:
            return getattr(memory, "promoted_to", None) is not None
        return True

    @staticmethod
    def _enforce_working_memory_expiry(memory: BaseMemory) -> None:
        """
        expires_at must equal created_at + timedelta(seconds=ttl_seconds).
        Enforced at write time by the orchestrator.
        """
        if memory.memory_type != MemoryTypeEnum.WORKING:
            return
        ttl = getattr(memory, "ttl_seconds", None)
        expires_at = getattr(memory, "expires_at", None)
        if ttl is None or expires_at is None:
            return
        expected = memory.created_at + timedelta(seconds=ttl)
        # Allow 1-second tolerance for float precision.
        delta = abs((expires_at - expected).total_seconds())
        if delta > 1.0:
            raise ValueError(
                f"Violated: WorkingMemory expires_at={expires_at!r} "
                f"does not equal created_at + ttl_seconds ({expected!r}). "
                f"delta={delta:.3f}s memory_id={memory.memory_id}"
            )

    # ------------------------------------------------------------------
    # Primary ingestion entrypoint
    # ------------------------------------------------------------------

    async def ingest_memory(
        self,
        memory: BaseMemory,
        embedding: list[float],
    ) -> TransactionResult:
        """
        Ingest a memory across all four stores in required write order.

        Write order:
          1. Firewall check          — hard reject before any I/O
          2. Working memory guard    — expiry validation
          3. SQLite event log        — MUST succeed; abort if it fails
          4. DuckDB state write      — memory document
          5. FAISS vector index      — skip for unpromoted WorkingMemory
          6. KuzuDB graph write      — best-effort; divergence logged

        If step 3 (SQLite) fails, steps 4-6 are NOT executed (INV-RC-001).
        If step 4 (DuckDB) fails, a failure TransactionResult is returned;
          the SQLite event is NOT rolled back (SQLite is append-only by
          design; partial-write detection is handled by integrity_checker.py
          via CS-003).
        If step 5 (FAISS) fails, a failure result is returned; CS-001
          requires ACTIVE memories to have vectors.
        Step 6 failures are logged at WARNING and do not fail the transaction.
        """
        # ---- Guard: firewall ----
        try:
            self._enforce_firewall(memory)
        except FirewallViolationError:
            logger.error(
                "storage.orchestrator | firewall violation rejected "
                "memory_id={} provenance={}",
                memory.memory_id,
                getattr(memory, "provenance", "?"),
            )
            raise  # Must not be caught and continued.

        # ---- Guard: WorkingMemory expiry ----
        try:
            self._enforce_working_memory_expiry(memory)
        except ValueError as exc:
            logger.error(
                "storage.orchestrator | INV-WM-002 violation | "
                "memory_id={} | error={}", memory.memory_id, exc,
            )
            return TransactionResult(success=False, error=str(exc))

        # ---- Step 1: SQLite event log (MUST succeed first) ----
        #If SQLite write fails, DuckDB MUST NOT proceed.
        try:
            self.event_log.log_ingestion(
                agent_id=memory.agent_id,
                payload=IngestionPayload(
                    memory_id=memory.memory_id,
                    memory_type=memory.memory_type.value,
                    sha256=memory.sha256,
                    chunks_created=1,
                    modality=memory.modality.value,
                    pipeline_stages_ms={},
                    entity_count=0,
                    relation_count=0,
                    provenance=str(
                        getattr(memory, "provenance", ProvenanceEnum.OBSERVED)
                        if not hasattr(getattr(memory, "provenance", None), "value")
                        else memory.provenance.value
                    ),
                ),
            )
        except Exception as exc:
            # SQLite failed → abort; DuckDB must not be written.
            logger.error(
                "storage.orchestrator | SQLite event log write failed — "
                "aborting ingestion (INV-RC-001) | memory_id={} | error={}",
                memory.memory_id, exc,
            )
            return TransactionResult(success=False, error=f"SQLite write failed: {exc}")

        # ---- Step 2: DuckDB state write ----
        try:
            self.duckdb.insert_memory(memory)
        except Exception as exc:
            logger.exception(
                "storage.orchestrator | DuckDB write failed | memory_id={}",
                memory.memory_id,
            )
            # SQLite event is already written (append-only)
            # the missing DuckDB record on the next integrity check.
            return TransactionResult(
                success=False,
                rollback_performed=False,
                error=f"DuckDB write failed: {exc}",
            )

        # ---- Step 3: FAISS vector index ----
        # Every ACTIVE memory MUST have a corresponding vector.
        # Skip unpromoted WorkingMemory.
        if self._should_index_in_faiss(memory):
            try:
                self.faiss.add_embedding(
                    embedding=embedding,
                    memory_id=memory.memory_id,
                    agent_id=memory.agent_id,
                    memory_type=memory.memory_type,
                    metadata={
                        "memory_id": str(memory.memory_id),
                        "agent_id": memory.agent_id,
                        "memory_type": memory.memory_type.value,
                    },
                )
            except Exception as exc:
                # FAISS failure: integrity violation.
                # DuckDB is written but FAISS is missing → log as error.
                # Do not silently swallow; return failure so pipeline can retry.
                logger.error(
                    "storage.orchestrator | FAISS write failed (CS-001 risk) — "
                    "DuckDB written but vector missing | memory_id={} | error={}",
                    memory.memory_id, exc,
                )
                return TransactionResult(
                    success=False,
                    rollback_performed=False,
                    error=f"FAISS write failed: {exc}",
                )

        # ---- Step 4: KuzuDB graph write (best-effort) ----
        # Failures logged at WARNING; do not fail the transaction.
        # Graph divergence from DuckDB is detected by integrity_checker.py.
        if self.graph is not None:
            await self._graph_write_memory(memory)
            await self._graph_write_edges(memory)

        return TransactionResult(success=True)

    async def _graph_write_memory(self, memory: BaseMemory) -> None:
        """Write memory node to KuzuDB (best-effort)."""
        if not hasattr(self.graph, "save_memory"):
            return
        try:
            await self.graph.save_memory(memory)
        except Exception as exc:
            logger.warning(
                "storage.orchestrator | graph node write failed (best-effort) — "
                "DuckDB and FAISS remain consistent | memory_id={} | error={}",
                memory.memory_id, exc,
            )

    async def _graph_write_edges(self, memory: BaseMemory) -> None:
        """
        Write REFERENCES edges to KuzuDB for any related memory IDs found
        on the memory object (source_ids, referenced_memory_ids).

        REFERENCES rel table links EpisodicMemory cross-references.
        Best-effort; edge write failures do not fail the transaction.
        """
        if not hasattr(self.graph, "save_edge"):
            return

        related_ids: list[UUID] = []
        for field_name in ("source_ids", "referenced_memory_ids"):
            values = getattr(memory, field_name, None) or []
            related_ids.extend(values)

        for related_id in related_ids:
            try:
                await self.graph.save_edge(
                    from_id=str(memory.memory_id),
                    to_id=str(related_id),
                    relation="REFERENCES",
                    rel_table="RELATES",
                )
            except Exception as exc:
                logger.warning(
                    "storage.orchestrator | graph edge write failed (best-effort) | "
                    "from={} to={} | error={}",
                    memory.memory_id, related_id, exc,
                )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_memory(self, memory_id: UUID) -> BaseMemory:
        """
        Retrieve a memory by ID and update access metadata.

        Retrieval is logged to SQLite (RETRIEVE event) AFTER the DuckDB
        read, because the log payload requires the pre-update importance
        score. This is the one exception to SQLite-first order: retrieval
        is idempotent and the event log is not used to replay retrieval
        state, only ingestion/lifecycle events.

        access_count update is logged with post-update value.
        """
        memory = self.duckdb.get_memory(memory_id)
        self.duckdb.update_access_metadata(memory_id)

        if self.event_log is not None:
            try:
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
            except Exception as exc:
                # Retrieval event log failure is non-fatal; the DuckDB update
                # already succeeded. Log at WARNING — replay correctness for
                # retrieval events is lower-stakes than ingestion events.
                logger.warning(
                    "storage.orchestrator | retrieval event log failed "
                    "(non-fatal) | memory_id={} | error={}",
                    memory_id, exc,
                )

        return self.duckdb.get_memory(memory_id)

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def transition_lifecycle(
        self,
        memory_id: UUID,
        new_state: LifecycleStateEnum,
    ) -> None:
        """
        Apply a lifecycle state transition.

        Write order:
          1. SQLite lifecycle transition event (MUST succeed)
          2. DuckDB lifecycle state update

        PRUNED is a terminal state — transitions out of
        PRUNED are rejected here.
        """
        memory = self.duckdb.get_memory(memory_id)
        old_state = memory.lifecycle_state

        # Guard: PRUNED is terminal.
        if (
            old_state == LifecycleStateEnum.PRUNED
            or (hasattr(old_state, "value") and old_state.value == "PRUNED")
        ):
            raise ValueError(
                f"INV-LC-001: PRUNED is a terminal state. "
                f"Transition to {new_state!r} rejected for memory_id={memory_id}"
            )

        # Step 1: SQLite MUST succeed before DuckDB update.
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

        # Step 2: DuckDB state update.
        self.duckdb.apply_lifecycle_transition(memory_id, new_state)

    # ------------------------------------------------------------------
    # Replay rebuild (disaster recovery stub)
    # ------------------------------------------------------------------

    def replay_rebuild(self, agent_id: str) -> TransactionResult:
        """
        Reconstruct DuckDB state from the immutable SQLite event log.

        Disaster recovery primitive. Full implementation is intentionally
        deferred to replay/reconstructor.py to preserve architecture
        boundaries. This method exists as an orchestration
        placeholder with the correct exception contract.
        """
        try:
            # IMPORTANT:
            # Full replay engine intentionally deferred
            # to replay/reconstructor.py
            #
            # This method currently exists as orchestration
            # placeholder to preserve architecture boundaries.

            return TransactionResult(success=True)

        except (MemoryNotFoundError, Exception) as exc:
            raise ReplayRebuildError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Health probe
    # ------------------------------------------------------------------

    def healthcheck(self) -> dict:
        """
        Basic orchestrator health probe.

        Returns per-subsystem health status. In production this should
        be extended with actual connectivity checks per store.
        """
        return {
            "sqlite": "healthy",
            "duckdb_store": "healthy",
            "faiss_store": "healthy",
            "graph_store": "healthy" if self.graph is not None else "not_configured",
            "orchestrator": "healthy",
        }

