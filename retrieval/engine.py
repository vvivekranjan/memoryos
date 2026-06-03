from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from memory.models import (
    BaseMemory,
    MemoryTypeEnum,
)

from retrieval.context_assembler import (
    ContextAssembler,
)

from retrieval.vector_retriever import (
    RetrievalCandidate,
    VectorRetriever,
)


class GraphStore(Protocol):
    async def bfs_traversal_async(
        self,
        seeds: list[str],
        max_hops: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        ...

@dataclass(slots=True)
class RetrievalTrace:
    """
    ACS RetrievalTrace contract.

    Future milestones extend:
    - graph_rank
    - temporal_rank
    - graph_path
    - activation_boost
    """

    final_score: float
    retrieved_by: list[str]
    vector_rank: int | None = None
    graph_rank: int | None = None
    temporal_rank: int | None = None
    importance_score: float = 0.0
    recency_boost: float = 0.0
    activation_boost: float = 0.0
    graph_path: list[str] | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class MemoryResult:

    memory: BaseMemory
    trace: RetrievalTrace

@dataclass(slots=True)
class RetrieveResult:
    memories: list[MemoryResult]
    cache_hit: bool
    latency_ms: float
    retrievers_used: list[str]

class MemoryStore(Protocol):

    async def get_memories_by_ids(
        self,
        *,
        agent_id: str,
        memory_ids: list[UUID],
    ) -> list[BaseMemory]:
        ...

class RetrievalEngineError(Exception):
    """Base retrieval engine error."""


class EmptyQueryError(RetrievalEngineError):
    """Raised when query is empty."""


class RetrievalValidationError(RetrievalEngineError):
    """Raised when retrieval parameters invalid."""

class RetrievalEngine:
    """
    Retrieval Coordinator.

    Responsibilities:
    - orchestrate retrievers
    - dispatch retrieval tasks
    - hydrate memory objects
    - build RetrievalTrace
    - coordinate ContextAssembler
    - enforce retrieval policies

    Does NOT:
    - perform ANN search directly
    - format prompts
    - perform graph traversal
    - embed memories
    """

    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever,
        memory_store: MemoryStore,
        context_assembler: ContextAssembler
        , graph_store: GraphStore | None = None
    ):
        self.vector_retriever = vector_retriever
        self.memory_store = memory_store
        self.context_assembler = context_assembler
        self.graph_store = graph_store
    
    async def retrieve(
        self,
        *,
        agent_id: str,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        memory_types: list[MemoryTypeEnum] | None = None,
        min_importance: float = 0.0,
        include_trace: bool = True,
        session_id: UUID | None = None,
    ) -> RetrieveResult:
        """
        ACS retrieve() implementation.

        - vector retrieval only

        Future:
        - graph retrieval
        - temporal retrieval
        - RRF fusion
        - spreading activation
        """

        started = time.perf_counter()

        self._validate_inputs(
            query=query,
            top_k=top_k,
            min_importance=min_importance,
        )

        # Retrieve a broader vector candidate pool, then rerank and trim.
        candidate_top_k = self._candidate_top_k(top_k)

        retriever_tasks = self._dispatch_retrievers(
            query_embedding=query_embedding,
            top_k=candidate_top_k,
        )

        retriever_names = list(retriever_tasks)
        retriever_results = await asyncio.gather(*retriever_tasks.values())
        vector_candidates = retriever_results[0] if retriever_results else []

        # No Results
        if not vector_candidates:
            latency_ms = (
                time.perf_counter() - started
            ) * 1000

            return RetrieveResult(
                memories=[],
                cache_hit=False,
                latency_ms=latency_ms,
                retrievers_used=retriever_names,
            )

        results = await self._hydrate_candidates(
            agent_id=agent_id,
            candidates=vector_candidates,
            memory_types=memory_types,
            min_importance=min_importance,
        )

        graph_candidates: list[MemoryResult] = []
        if self.graph_store is not None and results:
            graph_candidates = await self._expand_graph_candidates(
                agent_id=agent_id,
                vector_results=results,
                memory_types=memory_types,
                min_importance=min_importance,
                top_k=top_k,
            )

        if graph_candidates:
            retriever_names.append("graph")
            results = self._merge_results(results, graph_candidates)

        if not results:
            latency_ms = (
                time.perf_counter() - started
            ) * 1000

            return RetrieveResult(
                memories=[],
                cache_hit=False,
                latency_ms=latency_ms,
                retrievers_used=retriever_names,
            )

        assembled_results = (
            self.context_assembler.assemble(
                query=query,
                results=results,
                include_trace=include_trace,
            )
        )
        assembled_results = assembled_results[:top_k]

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        return RetrieveResult(
            memories=assembled_results,
            cache_hit=False,
            latency_ms=latency_ms,
            retrievers_used=retriever_names,
        )

    @staticmethod
    def _candidate_top_k(top_k: int) -> int:
        """Use a wider candidate set to improve recall for long-term episodic memories."""

        expanded = max(top_k, top_k * 4)
        return min(100, expanded)

    def _dispatch_retrievers(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
    ) -> dict[str, asyncio.Future[list[RetrievalCandidate]]]:
        """Create the retrieval fan-out that the coordinator runs with asyncio.gather."""

        return {
            "vector": self.vector_retriever.search_async(
                query_embedding=query_embedding,
                top_k=top_k,
            ),
        }

    async def _hydrate_candidates(
        self,
        *,
        agent_id: str,
        candidates: list[RetrievalCandidate],
        memory_types: list[MemoryTypeEnum] | None,
        min_importance: float,
    ) -> list[MemoryResult]:
        """Batch-hydrate memory rows and convert them into trace-bearing results."""

        memory_ids = [candidate.memory_id for candidate in candidates]

        memories = await self.memory_store.get_memories_by_ids(
            agent_id=agent_id,
            memory_ids=memory_ids,
        )

        memory_map = {
            memory.memory_id: memory for memory in memories
        }

        allowed_memory_types = self._normalise_memory_types(memory_types)

        results: list[MemoryResult] = []

        for rank, candidate in enumerate(candidates, start=1):
            memory = memory_map.get(candidate.memory_id)

            if memory is None:
                continue

            memory_type_value = self._memory_type_value(memory.memory_type)
            if allowed_memory_types and memory_type_value not in allowed_memory_types:
                continue

            if memory.importance_score < min_importance:
                continue

            results.append(
                MemoryResult(
                    memory=memory,
                    trace=RetrievalTrace(
                        final_score=candidate.score,
                        retrieved_by=["vector"],
                        vector_rank=rank,
                        importance_score=memory.importance_score,
                        recency_boost=0.0,
                        activation_boost=0.0,
                        trace_metadata={
                            "memory_id": str(candidate.memory_id),
                            "candidate_metadata": candidate.metadata,
                        },
                    ),
                )
            )

        return results

    async def _expand_graph_candidates(
        self,
        *,
        agent_id: str,
        vector_results: list[MemoryResult],
        memory_types: list[MemoryTypeEnum] | None,
        min_importance: float,
        top_k: int,
    ) -> list[MemoryResult]:
        seeds = [str(result.memory.memory_id) for result in vector_results[: max(1, min(3, len(vector_results)))]]
        if not seeds:
            return []

        traversed = await self.graph_store.bfs_traversal_async(
            seeds=seeds,
            max_hops=2,
            limit=max(10, top_k * 4),
        )

        if not traversed:
            return []

        vector_scores = {
            str(result.memory.memory_id): result.trace.final_score
            for result in vector_results
        }

        graph_scores: dict[str, tuple[float, dict[str, Any]]] = {}
        for edge in traversed:
            target = edge.get("target")
            if not target:
                continue

            source = edge.get("source")
            hop = int(edge.get("hop") or 1)
            source_score = vector_scores.get(str(source), 0.0)
            graph_score = source_score * (0.85 ** max(0, hop - 1))

            existing = graph_scores.get(str(target))
            if existing is None or graph_score > existing[0]:
                graph_scores[str(target)] = (
                    graph_score,
                    {
                        "source": str(source) if source is not None else None,
                        "target": str(target),
                        "relation": edge.get("relation"),
                        "hop": hop,
                    },
                )

        graph_memory_ids = [UUID(memory_id) for memory_id in graph_scores.keys()]
        if not graph_memory_ids:
            return []

        graph_memories = await self.memory_store.get_memories_by_ids(
            agent_id=agent_id,
            memory_ids=graph_memory_ids,
        )

        memory_map = {memory.memory_id: memory for memory in graph_memories}
        allowed_memory_types = self._normalise_memory_types(memory_types)

        graph_results: list[MemoryResult] = []
        for rank, memory_id in enumerate(graph_memory_ids, start=1):
            memory = memory_map.get(memory_id)
            if memory is None:
                continue

            memory_type_value = self._memory_type_value(memory.memory_type)
            if allowed_memory_types and memory_type_value not in allowed_memory_types:
                continue

            if memory.importance_score < min_importance:
                continue

            score, graph_path = graph_scores[str(memory_id)]
            graph_results.append(
                MemoryResult(
                    memory=memory,
                    trace=RetrievalTrace(
                        final_score=score,
                        retrieved_by=["graph"],
                        graph_rank=rank,
                        graph_path=[
                            graph_path["source"],
                            graph_path["target"],
                        ] if graph_path.get("source") else [graph_path["target"]],
                        importance_score=memory.importance_score,
                        activation_boost=score,
                        trace_metadata={
                            "graph_path": graph_path,
                        },
                    ),
                )
            )

        return graph_results

    @staticmethod
    def _merge_results(
        primary: list[MemoryResult],
        secondary: list[MemoryResult],
    ) -> list[MemoryResult]:
        merged: dict[UUID, MemoryResult] = {result.memory.memory_id: result for result in primary}

        for result in secondary:
            existing = merged.get(result.memory.memory_id)
            if existing is None:
                merged[result.memory.memory_id] = result
                continue

            if result.trace.final_score <= existing.trace.final_score:
                continue

            merged[result.memory.memory_id] = MemoryResult(
                memory=existing.memory,
                trace=RetrievalTrace(
                    final_score=result.trace.final_score,
                    retrieved_by=sorted(set(existing.trace.retrieved_by + result.trace.retrieved_by)),
                    vector_rank=existing.trace.vector_rank,
                    graph_rank=result.trace.graph_rank,
                    temporal_rank=existing.trace.temporal_rank,
                    importance_score=existing.trace.importance_score,
                    recency_boost=existing.trace.recency_boost,
                    activation_boost=max(existing.trace.activation_boost, result.trace.activation_boost),
                    graph_path=result.trace.graph_path or existing.trace.graph_path,
                    trace_metadata={
                        **existing.trace.trace_metadata,
                        **result.trace.trace_metadata,
                    },
                ),
            )

        return list(merged.values())

    @staticmethod
    def _normalise_memory_types(
        memory_types: list[MemoryTypeEnum] | None,
    ) -> set[str] | None:
        if not memory_types:
            return None

        return {
            RetrievalEngine._memory_type_value(memory_type)
            for memory_type in memory_types
        }

    @staticmethod
    def _memory_type_value(memory_type: Any) -> str:
        if hasattr(memory_type, "value"):
            return str(memory_type.value)
        return str(memory_type)
    
    @staticmethod
    def _validate_inputs(
        *,
        query: str,
        top_k: int,
        min_importance: float,
    ) -> None:
        
        if not query.strip():
            raise EmptyQueryError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise RetrievalValidationError(
                "top_k must be > 0"
            )

        if top_k > 100:
            raise RetrievalValidationError(
                "top_k cannot exceed 100"
            )

        if (
            min_importance < 0.0
            or min_importance > 1.0
        ):
            raise RetrievalValidationError(
                "min_importance must be within [0,1]"
            )

