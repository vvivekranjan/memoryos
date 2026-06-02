from __future__ import annotations

from typing import Tuple

from ingestion.chunker import Chunker
from ingestion.deduplicator import Deduplicator
from ingestion.pipeline import Pipeline
from retrieval.vector_retriever import VectorRetriever
from retrieval.context_assembler import ContextAssembler
from retrieval.engine import RetrievalEngine
from storage.duckdb_store import DuckDBStore
from storage.faiss_store import FAISSStore
from storage.orchestrator import StorageOrchestrator
from vector.embedder import Embedder

from .memory_client import MemoryClient
from .session_manager import SessionManager


def compose(
    *,
    chunk_size: int = 300,
    overlap: int = 50,
    min_chunk_size: int = 10,
    model_name: str | None = None,
) -> MemoryClient:
    """Create a fully-wired :class:`MemoryClient` with sensible defaults.

    This helper centralizes construction for CLIs and programmatic use.
    It returns a `MemoryClient` instance with a ready ingestion pipeline,
    retrieval engine, and embedder. Callers may still override components
    by constructing their own objects and passing them into `MemoryClient`.
    """

    chunker = Chunker(chunk_size=chunk_size, overlap=overlap, min_chunk_size=min_chunk_size)
    embedder = Embedder(model_name) if model_name else Embedder()

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
    pipeline = Pipeline(orchestrator=orchestrator, deduplicator=deduplicator, chunker=chunker, embedder=embedder)

    client = MemoryClient(
        ingestion_pipeline=pipeline,
        retrieval_engine=retrieval_engine,
        context_builder=None,
        session_manager=SessionManager(),
        embedder=embedder,
    )

    return client
