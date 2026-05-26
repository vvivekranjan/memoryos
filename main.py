from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from vector.embedder import Embedder
from ingestion.chunker import Chunker
from ingestion.deduplicator import Deduplicator
from ingestion.pipeline import IngestionRequest, Pipeline
from retrieval.vector_retriever import VectorRetriever
from storage.duckdb_store import DuckDBStore
from storage.faiss_store import FAISSStore
from storage.orchestrator import StorageOrchestrator


class Memory:
    """High-level convenience façade for ingestion and retrieval."""

    def __init__(self, *, chunk_size: int = 300, overlap: int = 50, min_chunk_size: int = 10):
        (
            self.pipeline,
            self.retriever,
            self.faiss_store,
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
    ) -> List[Dict[str, Any]]:
        """Synchronously retrieve the most relevant chunks for a query."""

        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    @staticmethod
    def _build_pipeline(
        *,
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
    ) -> tuple[Pipeline, VectorRetriever, FAISSStore]:
        """Constructs and initializes all ingestion and retrieval components."""

        chunker = Chunker(chunk_size=chunk_size, overlap=overlap, min_chunk_size=min_chunk_size)
        embedder = Embedder()
        
        duckdb_store = DuckDBStore()
        duckdb_store.initialise()
        
        faiss_store = FAISSStore()
        faiss_store.load()
        
        orchestrator = StorageOrchestrator(duckdb_store=duckdb_store, faiss_store=faiss_store)
        deduplicator = Deduplicator(store=duckdb_store)
        retriever = VectorRetriever(faiss_store=faiss_store, orchestrator=orchestrator, embedder=embedder)
        
        pipeline = Pipeline(
            deduplicator=deduplicator,
            chunker=chunker,
            embedder=embedder,
            orchestrator=orchestrator,
        )

        return pipeline, retriever, faiss_store


if __name__ == "__main__":
    Memory
