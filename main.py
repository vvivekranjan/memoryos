from __future__ import annotations

import asyncio
from typing import Any

from core.runtime import build_runtime
from ingestion.pipeline import IngestionRequest, Pipeline
from retrieval.engine import RetrievalEngine
from retrieval.vector_retriever import VectorRetriever
from storage.faiss_store import FAISSStore
from vector.embedder import Embedder


class Memory:
    """High-level convenience facade for ingestion and retrieval."""

    def __init__(self, *, chunk_size: int = 300, overlap: int = 50, min_chunk_size: int = 10):
        (
            self.pipeline,
            self.retriever,
            self.faiss_store,
            self.retrieval_engine,
            self.embedder,
        ) = self._build_pipeline(
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

    async def save(
        self,
        *,
        document_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Asynchronously ingest a document without blocking the event loop."""

        request = IngestionRequest(document_id=document_id, content=content)
        ingest_report = await self.pipeline.ingest_document(request=request)
        await asyncio.to_thread(self.faiss_store.persist)
        return ingest_report

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Synchronously retrieve the most relevant chunks for a query."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            embeddings = asyncio.run(self.embedder.generate_embeddings([query]))
            query_embedding = embeddings[0].tolist() if len(embeddings) > 0 else []

            result = asyncio.run(
                self.retrieval_engine.retrieve(
                    agent_id="default_agent",
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    min_importance=score_threshold,
                )
            )

            legacy: list[dict[str, Any]] = []
            for mem_res in result.memories:
                score = mem_res.trace.final_score if mem_res.trace is not None else 0.0
                metadata = {
                    "memory_id": str(mem_res.memory.memory_id),
                    "content": mem_res.memory.content,
                    "agent_id": mem_res.memory.agent_id,
                    "memory_type": str(mem_res.memory.memory_type),
                }
                legacy.append({"score": score, "metadata": metadata})

            return legacy

        raise RuntimeError("Event loop already running; call 'retrieval_engine.retrieve' awaitably")

    @staticmethod
    def _build_pipeline(
        *,
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
    ) -> tuple[Pipeline, VectorRetriever, FAISSStore, RetrievalEngine, Embedder]:
        """Construct and initialize ingestion and retrieval components."""

        runtime = build_runtime(
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

        return (
            runtime.pipeline,
            runtime.retriever,
            runtime.faiss_store,
            runtime.retrieval_engine,
            runtime.embedder,
        )


if __name__ == "__main__":
    Memory
