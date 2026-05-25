from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vector.embedder import Embedder
from ingestion.chunker import Chunker
from storage.faiss_store import FAISSStore
import asyncio

@dataclass(slots=True)
class IngestionRequest:
    """
    Public ingestion API request.
    """

    document_id: str
    content: str
    importance_score: float = 0.5

# (Removed unused IngestionResult dataclass)

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
        chunker: Chunker,
        embedder: Embedder,
        store: FAISSStore,
    ):
        
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
    
    async def ingest_document(
        self,
        *,
        request: IngestionRequest,
    ) -> dict[str, Any]:
        """Chunks a document, embeds chunks, and inserts them into the vector store."""

        chunking_result = self.chunker.chunk(request.content)
        chunk_texts = [chunk.content for chunk in chunking_result.chunks]
        # Generate embeddings asynchronously (may offload to thread inside Embedder)
        embeddings = await self.embedder.generate_embeddings(chunk_texts)

        metadatas = [
            {
                "document_id": request.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in chunking_result.chunks
        ]

        vector_ids = await self.store.add_embeddings_async(
            embeddings=[embedding.tolist() for embedding in embeddings],
            metadatas=metadatas,
        )

        return {
            "document_id": request.document_id,
            "chunks": [asdict(chunk) for chunk in chunking_result.chunks],
            "vector_ids": vector_ids,
            "total_chunks": chunking_result.total_chunks,
            "total_tokens": chunking_result.total_tokens,
        }


def ingest_document(
    *,
    document_id: str,
    content: str,
    chunker: Chunker,
    embedder: Embedder,
    store: FAISSStore,
    importance_score: float = 0.5,
) -> dict[str, Any]:
    """Synchronous helper that runs the async pipeline.

    This convenience wrapper keeps existing callers working by running the
    pipeline on the current thread with `asyncio.run`.
    """

    request = IngestionRequest(document_id=document_id, content=content, importance_score=importance_score)
    pipeline = Pipeline(chunker=chunker, embedder=embedder, store=store)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(pipeline.ingest_document(request=request))

    raise RuntimeError("Event loop already running; call 'Pipeline.ingest_document' and await it directly")