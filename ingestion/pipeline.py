from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from memory.models import BaseMemory, MemoryTypeEnum, EpisodicMemory, WorkingMemory, SpeakerRoleEnum
from vector.embedder import Embedder
from ingestion.chunker import Chunker, Chunk
from ingestion.deduplicator import Deduplicator
from storage.orchestrator import StorageOrchestrator

import asyncio
import hashlib
from uuid import UUID, uuid4

@dataclass(slots=True)
class IngestionRequest:
    """
    Public ingestion API request.
    """

    document_id: str
    content: str
    agent_id: str = "default_agent"
    memory_type: MemoryTypeEnum = MemoryTypeEnum.EPISODIC
    importance_score: float = 0.5
    session_id: str | None = None
    turn_index: int = 0
    speaker_role: str = "USER"
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
        deduplicator: Deduplicator,
        chunker: Chunker,
        embedder: Embedder,
        orchestrator: StorageOrchestrator,
    ):
        
        self.deduplicator = deduplicator
        self.chunker = chunker
        self.embedder = embedder
        self.orchestrator = orchestrator
    
    async def ingest_document(
        self,
        *,
        request: IngestionRequest,
    ) -> dict[str, Any]:
        """Chunks a document, embeds chunks, and inserts them into the vector store."""

        dedup_result = (
                await self.deduplicator.check_content(
                    agent_id=request.agent_id,
                    content=request.content,
                )
            )
        
        if dedup_result.is_duplicate:
            return

        chunking_result = self.chunker.chunk(request.content)
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
        session_id_val = uuid4() if not request.session_id else UUID(request.session_id)

        if request.memory_type == MemoryTypeEnum.WORKING:
            return WorkingMemory(
                memory_id=memory_id_val,
                agent_id=request.agent_id,
                content=chunk.content,
                sha256=_content_sha256(chunk.content),
                importance_score=request.importance_score,
                session_id=session_id_val,
                metadata=metadata,
                ttl_seconds=3600,
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
                speaker_role=request.speaker_role,
                metadata=metadata,
                is_system_message=False,
                referenced_memory_ids=[],
            )


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
    pipeline = Pipeline(deduplicator=deduplicator, chunker=chunker, embedder=embedder, orchestrator=orchestrator)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(pipeline.ingest_document(request=request))

    raise RuntimeError("Event loop already running; call 'Pipeline.ingest_document' and await it directly")