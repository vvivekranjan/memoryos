from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID, uuid4

from aimemoryos.ingestion.pdf_loader import PDFLoader
from aimemoryos.ingestion.preprocessor import Preprocessor
from aimemoryos.ingestion.multimodal_router import MultimodalRouter
from aimemoryos.memory.models import (
    BaseMemory,
    MemoryTypeEnum,
    SpeakerRoleEnum,
)
from aimemoryos.memory.episodic import EpisodicMemory
from aimemoryos.memory.working import WorkingMemory
from aimemoryos.vector.embedder import Embedder
from aimemoryos.ingestion.chunker import Chunker, Chunk
from aimemoryos.ingestion.deduplicator import DedupStore, Deduplicator
from aimemoryos.storage.orchestrator import StorageOrchestrator

@dataclass(slots=True)
class IngestionRequest:
    """
    Public ingestion API request.
    """

    document_id: str
    content: str
    agent_id: str = "default_agent"
    memory_type: MemoryTypeEnum | str = MemoryTypeEnum.EPISODIC
    importance_score: float = 0.5
    session_id: UUID | str | None = None
    turn_index: int = 0
    speaker_role: SpeakerRoleEnum | str = SpeakerRoleEnum.USER
    referenced_memory_ids: list[UUID | str] | None = None
    is_system_message: bool = False
    tool_call_id: str | None = None
    ttl_seconds: int = 3600
    promoted_to: UUID | str | None = None
    scratch_data: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

class Pipeline:
    """
    Ingestion pipeline orchestrator.

    Responsibilities:
    - stage sequencing
    - async orchestration
    - DLQ handling
    - telemetry
    - final storage dispatch

    Pipeline:
    preprocess
        ↓
    deduplicate
        ↓
    chunk
        ↓
    embed
        ↓
    persist
        ↓
    FAISS index
    """

    def __init__(
        self,
        *,
        orchestrator: StorageOrchestrator,
        preprocessor: Preprocessor | None = None,
        deduplicator: Deduplicator | None = None,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
    ):
        
        self.pdf_loader = PDFLoader()
        self.router = MultimodalRouter(self.pdf_loader)
        self.preprocessor = preprocessor or Preprocessor()
        self.chunker = chunker or Chunker()
        self.embedder = embedder or Embedder()
        self.orchestrator = orchestrator
        self.deduplicator = deduplicator or Deduplicator(store=orchestrator.duckdb)
    
    async def ingest_document(
        self,
        *,
        request: IngestionRequest,
    ) -> dict[str, Any]:
        """Chunks a document, embeds chunks, and inserts them into the vector store."""

        text = await self.router.route_and_extract(request.content)

        clean_text = await self.preprocessor.clean(text)

        dedup_result = await self.deduplicator.check_content(
            agent_id=request.agent_id,
            content=clean_text,
        )
        
        if dedup_result.is_duplicate:
            return {
                "document_id": request.document_id,
                "chunks": [],
                "vector_ids": [],
                "total_chunks": 0,
                "total_tokens": 0,
                "duplicate_detected": True,
            }

        chunking_result = self.chunker.chunk(clean_text)
        chunk_texts = [chunk.content for chunk in chunking_result.chunks]
        # Generate embeddings asynchronously (may offload to thread inside Embedder)
        embeddings = await self.embedder.generate_embeddings(chunk_texts)

        # Persist Memories
        vector_ids = []
        for chunk, embedding in zip(chunking_result.chunks, embeddings):

            memory = self._build_memory(
                request=request,
                chunk=chunk,
                # we don't pass content_hash because memory creation handles hashing,
                # but we can pass it as a param if we want, but build_memory expects it.
            )

            txn = await self.orchestrator.ingest_memory(
                memory=memory,
                embedding=embedding.tolist(),
            )
            if txn.success:
                vector_ids.append(str(memory.memory_id))

        return {
            "document_id": request.document_id,
            "chunks": [asdict(chunk) for chunk in chunking_result.chunks],
            "vector_ids": vector_ids,
            "total_chunks": chunking_result.total_chunks,
            "total_tokens": chunking_result.total_tokens,
        }
    
    @staticmethod
    def _build_memory(
        *,
        request: IngestionRequest,
        chunk: Chunk,
    ) -> BaseMemory:
        """
        Converts chunk into canonical memory.
        """

        metadata = {
            **(request.metadata or {}),
            "document_id": request.document_id,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
        }

        memory_id_val = uuid4()
        session_id_val = _coerce_uuid(request.session_id) if request.session_id else uuid4()
        memory_type = _coerce_memory_type(request.memory_type)
        speaker_role = _coerce_speaker_role(request.speaker_role)

        if memory_type == MemoryTypeEnum.WORKING:
            return WorkingMemory(
                memory_id=memory_id_val,
                agent_id=request.agent_id,
                content=chunk.content,
                sha256=_content_sha256(chunk.content),
                importance_score=request.importance_score,
                session_id=session_id_val,
                metadata=metadata,
                ttl_seconds=request.ttl_seconds,
                promoted_to=_coerce_uuid(request.promoted_to) if request.promoted_to else None,
                scratch_data=request.scratch_data or {},
            )
        else:
            return EpisodicMemory(
                memory_id=memory_id_val,
                agent_id=request.agent_id,
                content=chunk.content,
                sha256=_content_sha256(chunk.content),
                importance_score=request.importance_score,
                session_id=session_id_val,
                turn_index=request.turn_index,
                speaker_role=speaker_role,
                metadata=metadata,
                is_system_message=request.is_system_message,
                referenced_memory_ids=[
                    _coerce_uuid(memory_id)
                    for memory_id in (request.referenced_memory_ids or [])
                ],
                tool_call_id=request.tool_call_id,
            )


def _coerce_uuid(value: UUID | str) -> UUID:
    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def _coerce_memory_type(value: MemoryTypeEnum | str) -> MemoryTypeEnum:
    if isinstance(value, MemoryTypeEnum):
        return value

    return MemoryTypeEnum(str(value))


def _coerce_speaker_role(value: SpeakerRoleEnum | str) -> SpeakerRoleEnum:
    if isinstance(value, SpeakerRoleEnum):
        return value

    return SpeakerRoleEnum(str(value))


def ingest_document(
    *,
    document_id: str,
    content: str,
    deduplicator: Deduplicator,
    chunker: Chunker,
    embedder: Embedder,
    orchestrator: StorageOrchestrator,
    importance_score: float = 0.5,
) -> dict[str, Any]:
    """Synchronous helper that runs the async pipeline.

    This convenience wrapper keeps existing callers working by running the
    pipeline on the current thread with `asyncio.run`.
    """

    request = IngestionRequest(document_id=document_id, content=content, importance_score=importance_score)
    pipeline = Pipeline(
        orchestrator=orchestrator,
        deduplicator=deduplicator,
        chunker=chunker,
        embedder=embedder,
    )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(pipeline.ingest_document(request=request))

    raise RuntimeError("Event loop already running; call 'Pipeline.ingest_document' and await it directly")