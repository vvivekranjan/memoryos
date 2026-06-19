from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from memoryos.core.config import config
from memoryos.graph.ontology import KuzuDBStore
from memoryos.ingestion.chunker import Chunker
from memoryos.ingestion.deduplicator import Deduplicator
from memoryos.ingestion.pipeline import Pipeline
from memoryos.retrieval.context_assembler import ContextAssembler
from memoryos.retrieval.engine import RetrievalEngine
from memoryos.retrieval.vector_retriever import VectorRetriever
from memoryos.storage.duckdb_store import DuckDBStore
from memoryos.storage.faiss_store import FAISSStore
from memoryos.storage.orchestrator import StorageOrchestrator
from memoryos.storage.sqlite_log import SQLiteEventLog
from memoryos.vector.embedder import Embedder


@dataclass(slots=True, frozen=True)
class RuntimePaths:
    data_dir: Path
    duckdb_path: Path
    faiss_dir: Path
    events_path: Path
    graph_dir: Path

    @classmethod
    def from_env(cls) -> "RuntimePaths":
        data_dir = Path(config.data_dir)
        return cls(
            data_dir=data_dir,
            duckdb_path=data_dir / "memory.duckdb",
            faiss_dir=data_dir / "faiss",
            events_path=data_dir / "events.sqlite",
            graph_dir=data_dir / "graph" / "memory_graph.kuzu",
        )


@dataclass(slots=True)
class MemoryRuntime:
    paths: RuntimePaths
    embedder: Embedder
    chunker: Chunker
    duckdb_store: DuckDBStore
    faiss_store: FAISSStore
    event_log: SQLiteEventLog
    graph_store: KuzuDBStore
    orchestrator: StorageOrchestrator
    deduplicator: Deduplicator
    retriever: VectorRetriever
    context_assembler: ContextAssembler
    retrieval_engine: RetrievalEngine
    pipeline: Pipeline


def build_runtime(
    *,
    chunk_size: int = 300,
    overlap: int = 50,
    min_chunk_size: int = 10,
    paths: RuntimePaths | None = None,
    embedder: Embedder | None = None,
) -> MemoryRuntime:
    runtime_paths = paths or RuntimePaths.from_env()

    chunker = Chunker(
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
    )
    runtime_embedder = embedder or Embedder()

    duckdb_store = DuckDBStore(db_path=runtime_paths.duckdb_path)
    duckdb_store.initialise()

    faiss_store = FAISSStore(root_path=runtime_paths.faiss_dir)
    faiss_store.load()

    event_log = SQLiteEventLog(db_path=runtime_paths.events_path)
    graph_store = KuzuDBStore(db_path=str(runtime_paths.graph_dir))
    graph_store.initialise()

    orchestrator = StorageOrchestrator(
        duckdb_store=duckdb_store,
        faiss_store=faiss_store,
        sqlite_log=event_log,
        graph_store=graph_store,
    )
    deduplicator = Deduplicator(store=duckdb_store)
    retriever = VectorRetriever(
        faiss_store=faiss_store,
        orchestrator=orchestrator,
        embedder=runtime_embedder,
    )
    context_assembler = ContextAssembler()
    retrieval_engine = RetrievalEngine(
        vector_retriever=retriever,
        memory_store=duckdb_store,
        context_assembler=context_assembler,
        graph_store=graph_store,
    )
    pipeline = Pipeline(
        deduplicator=deduplicator,
        chunker=chunker,
        embedder=runtime_embedder,
        orchestrator=orchestrator,
    )

    return MemoryRuntime(
        paths=runtime_paths,
        embedder=runtime_embedder,
        chunker=chunker,
        duckdb_store=duckdb_store,
        faiss_store=faiss_store,
        event_log=event_log,
        graph_store=graph_store,
        orchestrator=orchestrator,
        deduplicator=deduplicator,
        retriever=retriever,
        context_assembler=context_assembler,
        retrieval_engine=retrieval_engine,
        pipeline=pipeline,
    )
