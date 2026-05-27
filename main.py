from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from vector.embedder import Embedder
from ingestion.chunker import Chunker
from ingestion.deduplicator import Deduplicator
from ingestion.pipeline import IngestionRequest, Pipeline
from retrieval.vector_retriever import VectorRetriever
from retrieval.context_assembler import ContextAssembler
from retrieval.engine import RetrievalEngine
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
    ) -> List[Dict[str, Any]]:
        """Synchronously retrieve the most relevant chunks for a query."""
        # Use the RetrievalEngine for coordinated retrieval and map to legacy format
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Generate embedding synchronously using the Embedder
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

            # Map RetrieveResult -> legacy list[dict]
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

        # If an event loop is running, instruct caller to use async path directly
        raise RuntimeError("Event loop already running; call 'retrieval_engine.retrieve' awaitably")

    @staticmethod
    def _build_pipeline(
        *,
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
    ) -> tuple[Pipeline, VectorRetriever, FAISSStore, RetrievalEngine, Embedder]:
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
        context_assembler = ContextAssembler()
        retrieval_engine = RetrievalEngine(
            vector_retriever=retriever,
            memory_store=duckdb_store,
            context_assembler=context_assembler,
        )
        pipeline = Pipeline(
            deduplicator=deduplicator,
            chunker=chunker,
            embedder=embedder,
            orchestrator=orchestrator,
        )

        return pipeline, retriever, faiss_store, retrieval_engine, embedder


if __name__ == "__main__":
    Memory
