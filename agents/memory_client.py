from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID, uuid4

from agents.context_builder import (
    BuiltContext,
    ContextBuilder,
)

from agents.session_manager import (
    AgentSession,
    SessionManager,
)

from ingestion.pipeline import (
    IngestionRequest,
    Pipeline,
)

from memory.models import (
    MemoryTypeEnum,
    SpeakerRoleEnum,
)

from retrieval.engine import (
    RetrievalEngine,
)

from retrieval.vector_retriever import VectorRetriever
from vector.embedder import (
    Embedder,
)

from storage.duckdb_store import DuckDBStore
from storage.faiss_store import FAISSStore
from storage.orchestrator import StorageOrchestrator

from retrieval.context_assembler import ContextBlock
from retrieval.context_assembler import ContextAssembler

@dataclass(slots=True)
class IngestResult:
    """
    SDK ingestion result.
    """

    ingestion_id: UUID
    memory_ids: list[UUID]
    chunks_created: int
    duplicate_detected: bool


@dataclass(slots=True)
class RetrieveResult:
    """
    SDK retrieval result.
    """

    context: BuiltContext
    raw_result: Any

class MemoryClient:
    """
    Primary MemoryOS SDK interface.

    Responsibilities:
    - memory ingestion
    - memory retrieval
    - session coordination
    - prompt-ready context assembly

    Application developers should interact
    with THIS layer only.

    Does NOT expose:
    - FAISS
    - replay internals
    - orchestrators
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
    ):
        self.agent_id = agent_id
        self.endpoint = endpoint

        self.embedder = embedder or Embedder()
        self.context_builder = context_builder or ContextBuilder()
        self.session_manager = session_manager or SessionManager()

        self.db_store = DuckDBStore()
        self.faiss_store = FAISSStore()
        self.db_store.initialise()
        self.faiss_store.load()

        self.orchestrator = StorageOrchestrator(
            duckdb_store=self.db_store,
            faiss_store=self.faiss_store,
        )

        self.ingestion = ingestion_pipeline or Pipeline(orchestrator=self.orchestrator)
        self.retrieval_engine = retrieval_engine or RetrievalEngine(
            vector_retriever=VectorRetriever(
                faiss_store=self.faiss_store,
                orchestrator=self.orchestrator,
                embedder=self.embedder,
            ),
            memory_store=self.db_store,
            context_assembler=ContextAssembler(),
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
        metadata: (
            dict[str, Any]
            | None
        ) = None,
    ) -> IngestResult:
        """
        Canonical SDK ingestion entrypoint.
        """

        # Run the ingestion pipeline. Current Pipeline exposes `ingest_document` and
        # returns a dict on success or `None` when a duplicate is detected.

        agent = agent_id or self.agent_id
        request_document_id = uuid4()

        pipeline_result = await self.ingestion.ingest_document(
            request=IngestionRequest(
                document_id=str(request_document_id),
                content=content,
                agent_id=agent,
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
                metadata=(metadata or {}),
            )
        )

        if pipeline_result is None:
            return IngestResult(
                ingestion_id=request_document_id,
                memory_ids=[],
                chunks_created=0,
                duplicate_detected=True,
            )

        vector_ids = pipeline_result.get("vector_ids", [])
        memory_ids = [UUID(v) for v in vector_ids]

        return IngestResult(
            ingestion_id=request_document_id,
            memory_ids=memory_ids,
            chunks_created=pipeline_result.get("total_chunks", len(pipeline_result.get("chunks", []))),
            duplicate_detected=False,
        )

    async def retrieve(
        self,
        *,
        agent_id: Optional[str] = None,
        query: str,
        top_k: int = 10,
        min_score: (
            float | None
        ) = None,
        filters: (
            dict[str, Any]
            | None
        ) = None,
    ) -> RetrieveResult:
        """
        Retrieves prompt-ready context.
        """

        # Generate query embedding using the provided embedder
        embeddings = await self.embedder.generate_embeddings([query])
        query_embedding = embeddings[0].tolist() if len(embeddings) > 0 else []

        result = await self.retrieval_engine.retrieve(
            agent_id=agent_id,
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            min_importance=(min_score if min_score is not None else 0.0),
            memory_types=self._coerce_memory_types((filters or {}).get("memory_types")),
        )

        # Use the engine's ContextAssembler to build a ContextBlock from MemoryResults
        context_block = self.retrieval_engine.context_assembler.build_context_block(
            query=query,
            results=result.memories,
            max_memories=self.context_builder.max_memories,
            include_trace=True,
        )

        built_context = self.context_builder.build(context_block)

        return RetrieveResult(
            context=built_context,
            raw_result=result,
        )

    @staticmethod
    def _coerce_memory_types(
        values: Any,
    ) -> list[MemoryTypeEnum] | None:
        if values is None:
            return None

        if isinstance(values, str):
            values = [values]

        return [MemoryTypeEnum(value) for value in values]

    @staticmethod
    def _coerce_memory_type(value: MemoryTypeEnum | str) -> MemoryTypeEnum:
        if isinstance(value, MemoryTypeEnum):
            return value

        return MemoryTypeEnum(str(value))

    @staticmethod
    def _build_context_block(
        *,
        query: str,
        result: Any,
    ):
        # Deprecated: kept for backward-compatibility; prefer using
        # `retrieval_engine.context_assembler.build_context_block(...)` directly.
        return ContextBlock(
            query=query,
            memories=result.memories,
            combined_context="",
            token_estimate=0,
            retrieval_summary={},
        )

    async def feedback(
        self,
        *,
        memory_id: UUID,
        score: float,
        metadata: (
            dict[str, Any]
            | None
        ) = None,
    ) -> None:
        """
        Feedback hook placeholder.

        Scope:
        currently no-op.

        Future:
        - reinforcement
        - decay adjustment
        - reflection influence
        """

        _ = (
            memory_id,
            score,
            metadata,
        )

    def session(
        self,
        *,
        agent_id: str,
        metadata: (
            dict[str, Any]
            | None
        ) = None,
    ) -> AgentSession:
        """
        Creates managed session.
        """

        return (
            self.session_manager
            .create_session(
                agent_id=agent_id,
                metadata=(
                    metadata
                    or {}
                ),
            )
        )

    def working_memories(
        self,
        *,
        session_id: UUID,
    ):
        """
        Returns active working memories.
        """

        return (
            self.session_manager
            .working_memories(
                session_id=session_id
            )
        )

    def advance_turn(
        self,
        *,
        session_id: UUID,
    ) -> int:
        """
        Advances conversation turn.
        """

        return (
            self.session_manager
            .advance_turn(
                session_id=session_id
            )
        )


    def close_session(
        self,
        *,
        session_id: UUID,
    ) -> None:
        """
        Terminates session.
        """

        self.session_manager.close_session(session_id=session_id)
    
    async def forget(
        self,
        *,
        memory_id: str,
    ) -> None:
        """Remove a memory."""
        pass

