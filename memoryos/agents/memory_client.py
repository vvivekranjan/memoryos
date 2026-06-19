from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID, uuid4

from memoryos.agents.context_builder import BuiltContext, ContextBuilder
from memoryos.agents.session_manager import AgentSession, SessionManager
from memoryos.ingestion.pipeline import IngestionRequest, Pipeline
from memoryos.memory.models import MemoryTypeEnum, SpeakerRoleEnum
from memoryos.retrieval.context_assembler import ContextBlock
from memoryos.retrieval.engine import RetrievalEngine
from memoryos.vector.embedder import Embedder

from memoryos.core.runtime import MemoryRuntime, build_runtime

import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestResult:
    """
    SDK ingestion result.

    Fields
    ------
    ingestion_id       : request-level UUID (not a memory_id)
    memory_ids         : UUIDs of created memory records
    chunks_created     : total chunks produced by the chunker
    duplicate_detected : True when the entire content was a SHA-256 duplicate
    """

    ingestion_id: UUID
    memory_ids: list[UUID]
    chunks_created: int
    duplicate_detected: bool


@dataclass(slots=True)
class RetrieveResult:
    """
    SDK retrieval result.

    Fields
    ------
    context    : prompt-ready BuiltContext from ContextBuilder
    raw_result : raw RetrievalResult from RetrievalEngine for callers
                 that need RetrievalTrace per memory
    """

    context: BuiltContext
    raw_result: Any


class MemoryClient:
    """
    Primary MemoryOS SDK interface (TRD §16: agents/memory_client.py, M1).

    Application developers interact with THIS class only. All storage
    backends, retrieval internals, and pipeline stages are hidden behind
    the ingest() / retrieve() / feedback() / session() surface.

    Construction modes
    ------------------
    1. Runtime mode (recommended for local deployment):
       Pass `runtime` or leave it None → build_runtime() is called.

    2. Component mode (for testing or custom wiring):
       Pass `ingestion_pipeline` + `retrieval_engine` explicitly.
       `runtime` is not used in this mode.

    Both modes require an `embedder` — either explicit or embedded in the
    runtime object.
    """

    def __init__(
        self,
        *,
        agent_id: str = "default",
        endpoint: str = "local",
        ingestion_pipeline: Pipeline | None = None,
        retrieval_engine: RetrievalEngine | None = None,
        context_builder: ContextBuilder | None = None,
        session_manager: SessionManager | None = None,
        embedder: Embedder | None = None,
        runtime: MemoryRuntime | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.endpoint = endpoint

        # Shared components — constructed with defaults if not supplied.
        self.embedder = embedder or Embedder()
        self.context_builder = context_builder or ContextBuilder()
        self.session_manager = session_manager or SessionManager()
        self.runtime = runtime

        # Auto-build runtime when neither runtime nor explicit components given.
        if self.runtime is None and (
            ingestion_pipeline is None or retrieval_engine is None
        ):
            if build_runtime is None:
                raise ImportError(
                    "core.runtime is not available. Supply ingestion_pipeline "
                    "and retrieval_engine explicitly, or install the full runtime."
                )
            self.runtime = build_runtime(embedder=self.embedder)

        # Unpack runtime stores so they are accessible for health / future use.
        if self.runtime is not None:
            self.db_store = self.runtime.duckdb_store
            self.faiss_store = self.runtime.faiss_store
            self.event_log = self.runtime.event_log
            self.graph_store = self.runtime.graph_store
            self.orchestrator = self.runtime.orchestrator
            self.ingestion: Pipeline = ingestion_pipeline or self.runtime.pipeline
            self.retrieval_engine: RetrievalEngine = (
                retrieval_engine or self.runtime.retrieval_engine
            )
        else:
            self.db_store = None
            self.faiss_store = None
            self.event_log = None
            self.graph_store = None
            self.orchestrator = None
            self.ingestion = ingestion_pipeline  # type: ignore[assignment]
            self.retrieval_engine = retrieval_engine  # type: ignore[assignment]

        if self.ingestion is None or self.retrieval_engine is None:
            raise ValueError(
                "MemoryClient requires either a runtime or both "
                "ingestion_pipeline and retrieval_engine."
            )

    async def ingest(
        self,
        *,
        agent_id: Optional[str] = None,
        content: str,
        memory_type: MemoryTypeEnum | str = MemoryTypeEnum.EPISODIC,
        importance_score: float = 0.5,
        session_id: UUID | None = None,
        turn_index: int = 0,
        speaker_role: SpeakerRoleEnum = SpeakerRoleEnum.USER,
        referenced_memory_ids: list[UUID] | None = None,
        is_system_message: bool = False,
        tool_call_id: str | None = None,
        ttl_seconds: int = 3600,
        promoted_to: UUID | None = None,
        scratch_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """
        Ingest content through the full 5-stage pipeline.

        Stages:
          1. Preprocess  — normalise, detect language, strip PII
          2. Deduplicate — SHA-256 gate before any embedding compute
          3. Chunk       — semantic sentence-boundary split
          4. Embed       — sentence-transformers inference
          5. Store       — SQLite → DuckDB → FAISS → KuzuDB (orchestrator)

        Duplicate content (SHA-256 match) is silently dropped and returns
        IngestResult(duplicate_detected=True, memory_ids=[], chunks_created=0).
        This is not an error.

        Parameters mirror the /ingest endpoint contract exactly.
        """
        resolved_agent = agent_id or self.agent_id
        request_id = uuid4()

        pipeline_result = await self.ingestion.ingest_document(
            request=IngestionRequest(
                document_id=str(request_id),
                content=content,
                agent_id=resolved_agent,
                memory_type=self._coerce_memory_type(memory_type),
                importance_score=importance_score,
                session_id=session_id,
                turn_index=turn_index,
                speaker_role=speaker_role,
                referenced_memory_ids=referenced_memory_ids,
                is_system_message=is_system_message,
                tool_call_id=tool_call_id,
                ttl_seconds=ttl_seconds,
                promoted_to=promoted_to,
                scratch_data=scratch_data,
                metadata=metadata or {},
            )
        )

        if not pipeline_result or pipeline_result.get("duplicate_detected"):
            return IngestResult(
                ingestion_id=request_id,
                memory_ids=[],
                chunks_created=0,
                duplicate_detected=True,
            )

        vector_ids: list[str] = pipeline_result.get("vector_ids", [])
        chunks_count: int = pipeline_result.get(
            "total_chunks",
            len(pipeline_result.get("chunks", [])),
        )

        logger.debug(
            "memory_client | ingest | agent_id={} chunks={} memory_ids={}",
            resolved_agent, chunks_count, vector_ids,
        )

        return IngestResult(
            ingestion_id=request_id,
            memory_ids=[UUID(v) for v in vector_ids],
            chunks_created=chunks_count,
            duplicate_detected=False,
        )

    async def retrieve(
        self,
        *,
        agent_id: Optional[str] = None,
        query: str,
        top_k: int = 10,
        min_score: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrieveResult:
        """
        Retrieve prompt-ready context for a natural language query.

        Embeds the query, runs the retrieval engine, assembles a ContextBlock,
        and returns a BuiltContext ready for LLM injection.

        Every returned memory carries a full RetrievalTrace.
        IMAGINED and HYPOTHESISED memories are excluded unconditionally.
        """
        resolved_agent = agent_id or self.agent_id

        embeddings = await self.embedder.generate_embeddings([query])
        query_embedding: list[float] = (
            embeddings[0].tolist() if len(embeddings) > 0 else []
        )

        result = await self.retrieval_engine.retrieve(
            agent_id=resolved_agent,
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            min_importance=min_score if min_score is not None else 0.0,
            memory_types=self._coerce_memory_types(
                (filters or {}).get("memory_types")
            ),
        )

        context_block: ContextBlock = (
            self.retrieval_engine.context_assembler.build_context_block(
                query=query,
                results=result.memories,
                max_memories=self.context_builder.max_memories,
                include_trace=True,
            )
        )

        built_context = self.context_builder.build(context_block)

        logger.debug(
            "memory_client | retrieve | agent_id={} query={!r} "
            "memories_returned={} cache_hit={}",
            resolved_agent, query[:60],
            built_context.memory_count,
            getattr(result, "cache_hit", False),
        )

        return RetrieveResult(context=built_context, raw_result=result)

    async def feedback(
        self,
        *,
        memory_id: UUID,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Submit a feedback signal for a retrieved memory.

        At M1 this is a no-op placeholder. At M4A it routes to
        autonomous/feedback_collector.py (USED / IGNORED / CORRECTION /
        CONFIRMED signal dispatch → SelfTuner Bayesian update).

        The parameter signature matches the ACS /feedback contract so
        callers are not broken when the body is wired at M4A.
        """
        _ = (memory_id, score, metadata)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def session(
        self,
        *,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentSession:
        """
        Create a managed AgentSession for the given agent.

        The returned AgentSession carries a session_id for use with
        advance_turn(), working_memories(), and close_session().
        """
        return self.session_manager.create_session(
            agent_id=agent_id,
            metadata=metadata or {},
        )

    def working_memories(self, *, session_id: UUID):
        """Return active (non-expired) WorkingMemory objects for the session."""
        return self.session_manager.working_memories(session_id=session_id)

    def advance_turn(self, *, session_id: UUID) -> int:
        """Advance the conversation turn counter and return the new index."""
        return self.session_manager.advance_turn(session_id=session_id)

    def close_session(self, *, session_id: UUID) -> None:
        """Terminate the session and release its working memory list."""
        self.session_manager.close_session(session_id=session_id)

    # ------------------------------------------------------------------
    # Event API stubs
    # ------------------------------------------------------------------

    async def replay(
        self,
        *,
        agent_id: Optional[str] = None,
        until: Any,                        # datetime (UTC)
        verify_checksums: bool = True,
        halt_on_gap: bool = True,
    ) -> Any:
        """
        Reconstruct agent memory state at a historical timestamp.

        Full implementation lives in replay/reconstructor.py.
        This stub preserves the ACS §4.1 contract so callers can be
        written against it before the reconstructor is wired.

        ACS invariants that the full implementation must honour:
          EVT-001: read-only — zero writes or state changes.
          EVT-002: checksum verified before each event is applied.
          EVT-003: deterministic — same params always return same result.
        """
        raise NotImplementedError(
            "replay() full implementation lives in replay/reconstructor.py. "
        )

    async def snapshot(
        self,
        *,
        agent_id: Optional[str] = None,
        output_path: Any,                  # pathlib.Path
        since: Any | None = None,          # datetime | None
        verify: bool = True,
    ) -> Any:
        """
        Export the SQLite event log to a portable snapshot file.
        
        Full implementation deferred to M3 / storage/sqlite_log.py.
        EVT-004: read-only — no events written, no state modified.
        EVT-005: if verify=True and any checksum fails, no file is written.
        """
        if self.event_log is None:
            raise RuntimeError("Cannot snapshot: event_log is not initialised in runtime.")
        
        return self.event_log.snapshot(
            output_path=output_path,
            since=since,
            verify=verify,
        )

    async def forget(self, *, memory_id: str) -> None:
        """
        Remove a memory across DuckDB, FAISS, and KuzuDB.

        Deferred until coordinated multi-store deletion is implemented.
        Must write a MEMORY_PRUNED SQLite event before any store delete
        (SSD INV-RC-001). Terminal PRUNED lifecycle state means this is
        irreversible (SSD INV-LC-001).
        """
        if self.orchestrator is None:
            raise RuntimeError("Cannot forget: orchestrator is not initialised in runtime.")
            
        result = await self.orchestrator.forget_memory(memory_id=UUID(memory_id))
        if not result.success:
            raise RuntimeError(f"forget() failed: {result.error}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_memory_type(value: MemoryTypeEnum | str) -> MemoryTypeEnum:
        """Coerce str to MemoryTypeEnum; pass-through if already an enum."""
        if isinstance(value, MemoryTypeEnum):
            return value
        return MemoryTypeEnum(str(value))

    @staticmethod
    def _coerce_memory_types(
        values: Any,
    ) -> list[MemoryTypeEnum] | None:
        """Coerce a single str or list of str/enum to list[MemoryTypeEnum]."""
        if values is None:
            return None
        if isinstance(values, str):
            values = [values]
        return [
            v if isinstance(v, MemoryTypeEnum) else MemoryTypeEnum(str(v))
            for v in values
        ]

    @staticmethod
    def _build_context_block(*, query: str, result: Any) -> ContextBlock:
        """
        Backward-compatibility shim.

        Deprecated: prefer retrieval_engine.context_assembler
        .build_context_block() directly. Kept so existing callers that
        reference this static method are not broken during migration.
        """
        return ContextBlock(
            query=query,
            memories=result.memories,
            combined_context="",
            token_estimate=0,
            retrieval_summary={},
        )

