from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from memoryos.core.exceptions import EmptyQueryError, RetrievalValidationError
from memoryos.graph.ontology import KuzuDBStore
from memoryos.memory.models import (
    BaseMemory,
    LifecycleStateEnum,
    MemoryTypeEnum,
)

from memoryos.retrieval.context_assembler import ContextAssembler
from memoryos.retrieval.vector_retriever import VectorRetriever, VectorCandidate

# ============================================================
# RETRIEVAL TRACE
# ============================================================

@dataclass(slots=True)
class RetrievalTrace:
    """
    ACS RetrievalTrace contract.
    """
    
    memory_id: UUID
    final_score: float
    retrieved_by: list[str]          # retriever names e.g. ["vector"] or ["vector", "graph"]
    vector_rank: int | None = None
    graph_rank: int | None = None 
    temporal_rank: int | None = None 
    importance_score: float = 0.0
    recency_boost: float = 0.0
    activation_boost: float = 0.0
    graph_path: list[str] | None = None
    provenance: str = "OBSERVED"
    provenance_confidence: float = 1.0


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

@dataclass(slots=True)
class RetrieverContribution:
    retriever: str
    rank: int
    raw_score: float
    rrf_contribution: float

@dataclass(slots=True)
class RetrievalCandidate:
    
    memory_id: UUID
    content: str
    memory_type: MemoryTypeEnum
    final_score: float
    trace: RetrievalTrace


class MemoryStore(Protocol):

    async def get_memories_by_ids(
        self,
        *,
        agent_id: str,
        memory_ids: list[UUID],
    ) -> list[BaseMemory]:
        ...


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

    _GRAPH_ENABLED: bool = True

    # Candidate pool multiplier — wider pool improves recall before top_k trim
    _CANDIDATE_POOL_MULTIPLIER: int = 4
    _CANDIDATE_POOL_MAX: int = 100

    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever,
        memory_store: MemoryStore,
        context_assembler: ContextAssembler,
        graph_store: KuzuDBStore | None = None,
    ):
        self.vector_retriever = vector_retriever
        self.memory_store = memory_store
        self.context_assembler = context_assembler
        self.graph_store = graph_store

    # ──────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────

    async def retrieve(
        self,
        *,
        agent_id: str,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        memory_types: list[MemoryTypeEnum] | None = None,
        lifecycle_states: list[LifecycleStateEnum] | None = None,
        min_importance: float = 0.0,
        include_trace: bool = True,
        session_id: UUID | None = None,
    ) -> RetrieveResult:
        """
        Primary retrieval entrypoint.

        Vector-only retrieval.
        Adds graph expansion when _GRAPH_ENABLED=True.
        """

        started = time.perf_counter()

        self._validate_inputs(
            query=query,
            top_k=top_k,
            min_importance=min_importance,
        )

        # ── Vector retrieval ──────────────────────────────
        candidate_top_k = self._expand_top_k(top_k)

        vector_candidates = await self.vector_retriever.search_async(
            agent_id=agent_id,
            query_embedding=query_embedding,
            top_k=candidate_top_k,
            memory_types=memory_types,
            lifecycle_states=lifecycle_states,
            min_importance=min_importance,
        )

        if not vector_candidates:
            return RetrieveResult(
                memories=[],
                cache_hit=False,
                latency_ms=self._elapsed(started),
                retrievers_used=["vector"],
            )

        # ── Hydrate ───────────────────────────────────────
        results = await self._hydrate(
            agent_id=agent_id,
            candidates=vector_candidates,
            memory_types=memory_types,
            min_importance=min_importance,
            retriever_name="vector",
        )

        retrievers_used = ["vector"]

        if self._GRAPH_ENABLED and self.graph_store is not None and results:
            graph_results = await self._expand_graph(
                agent_id=agent_id,
                vector_results=results,
                memory_types=memory_types,
                min_importance=min_importance,
                top_k=top_k,
            )
            if graph_results:
                results = self._merge(results, graph_results)
                retrievers_used.append("graph")

        if not results:
            return RetrieveResult(
                memories=[],
                cache_hit=False,
                latency_ms=self._elapsed(started),
                retrievers_used=retrievers_used,
            )

        # ── Assemble + trim ───────────────────────────────
        assembled = self.context_assembler.assemble(
            query=query,
            results=results,
            include_trace=include_trace,
        )

        return RetrieveResult(
            memories=assembled[:top_k],
            cache_hit=False,
            latency_ms=self._elapsed(started),
            retrievers_used=retrievers_used,
        )

    # ──────────────────────────────────────────────────────
    # HYDRATION
    # ──────────────────────────────────────────────────────

    async def _hydrate(
        self,
        *,
        agent_id: str,
        candidates: list[VectorCandidate],
        memory_types: list[MemoryTypeEnum] | None,
        min_importance: float,
        retriever_name: str,
    ) -> list[MemoryResult]:
        """
        Batch-fetches memory objects from DuckDB
        and attaches RetrievalTrace to each.
        """

        memory_ids = [c.memory_id for c in candidates]

        memories = await self.memory_store.get_memories_by_ids(
            agent_id=agent_id,
            memory_ids=memory_ids,
        )
        memory_map: dict[UUID, BaseMemory] = {
            m.memory_id: m for m in memories
        }

        allowed_types = self._coerce_types(memory_types)

        results: list[MemoryResult] = []

        for rank, candidate in enumerate(candidates, start=1):
            memory = memory_map.get(candidate.memory_id)
            if memory is None:
                continue

            if allowed_types and self._type_value(memory.memory_type) not in allowed_types:
                continue

            if memory.importance_score < min_importance:
                continue

            trace = RetrievalTrace(
                memory_id=candidate.memory_id,
                final_score=candidate.score,
                retrieved_by=[retriever_name],
                vector_rank=rank if retriever_name == "vector" else None,
                importance_score=memory.importance_score,
                recency_boost=0.0,
                activation_boost=0.0,
            )

            results.append(MemoryResult(memory=memory, trace=trace))

        return results

    # ──────────────────────────────────────────────────────
    # GRAPH EXPANSION
    # ──────────────────────────────────────────────────────

    async def _expand_graph(
        self,
        *,
        agent_id: str,
        vector_results: list[MemoryResult],
        memory_types: list[MemoryTypeEnum] | None,
        min_importance: float,
        top_k: int,
    ) -> list[MemoryResult]:
        """
        BFS graph expansion from top vector seeds.
        """

        seeds = [
            str(r.memory.memory_id)
            for r in vector_results[:min(3, len(vector_results))]
        ]

        from graph.traversal import bfs_traversal
        traversed = await bfs_traversal(
            store=self.graph_store,
            seeds=seeds,
            max_hops=2,
            limit=max(10, top_k * 4),
        )

        if not traversed:
            return []

        vector_scores = {
            str(r.memory.memory_id): r.trace.final_score
            for r in vector_results
        }

        # Score each graph neighbour by proximity-decay from nearest seed
        graph_scores: dict[str, tuple[float, dict[str, Any]]] = {}

        for edge in traversed:
            target = edge.get("end_id")
            if not target:
                continue

            source = edge.get("start_id")
            hop = int(edge.get("hop") or 1)
            source_score = vector_scores.get(str(source), 0.0)
            score = source_score * (0.85 ** max(0, hop - 1))

            existing = graph_scores.get(str(target))
            if existing is None or score > existing[0]:
                graph_scores[str(target)] = (score, {
                    "source": str(source) if source else None,
                    "target": str(target),
                    "relation": edge.get("relation"),
                    "hop": hop,
                })

        graph_memory_ids = [UUID(mid) for mid in graph_scores]
        if not graph_memory_ids:
            return []

        graph_memories = await self.memory_store.get_memories_by_ids(
            agent_id=agent_id,
            memory_ids=graph_memory_ids,
        )
        memory_map: dict[UUID, BaseMemory] = {
            m.memory_id: m for m in graph_memories
        }

        allowed_types = self._coerce_types(memory_types)
        results: list[MemoryResult] = []

        for rank, mid in enumerate(graph_memory_ids, start=1):
            memory = memory_map.get(mid)
            if memory is None:
                continue

            if allowed_types and self._type_value(memory.memory_type) not in allowed_types:
                continue

            if memory.importance_score < min_importance:
                continue

            score, path_meta = graph_scores[str(mid)]
            graph_path = (
                [path_meta["source"], path_meta["target"]]
                if path_meta.get("source")
                else [path_meta["target"]]
            )

            results.append(MemoryResult(
                memory=memory,
                trace=RetrievalTrace(
                    final_score=score,
                    retrieved_by=["graph"],
                    graph_rank=rank,
                    graph_path=graph_path,
                    importance_score=memory.importance_score,
                    activation_boost=score,
                    trace_metadata={"graph_path": path_meta},
                ),
            ))

        return results

    # ──────────────────────────────────────────────────────
    # MERGE (RRF-ready stub)
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _merge(
        primary: list[MemoryResult],
        secondary: list[MemoryResult],
    ) -> list[MemoryResult]:
        """
        Merges two ranked result lists.
        Deduplicates by memory_id, keeps highest score.
        Replace with full RRF fusion.
        """

        merged: dict[UUID, MemoryResult] = {
            r.memory.memory_id: r for r in primary
        }

        for result in secondary:
            mid = result.memory.memory_id
            existing = merged.get(mid)

            if existing is None:
                merged[mid] = result
                continue

            if result.trace.final_score <= existing.trace.final_score:
                continue

            # Higher score from graph — merge traces
            merged[mid] = MemoryResult(
                memory=existing.memory,
                trace=RetrievalTrace(
                    memory_id=str(existing.memory.memory_id),
                    final_score=result.trace.final_score,
                    retrieved_by=sorted(
                        set(existing.trace.retrieved_by + result.trace.retrieved_by)
                    ),
                    vector_rank=existing.trace.vector_rank,
                    graph_rank=result.trace.graph_rank,
                    temporal_rank=existing.trace.temporal_rank,
                    importance_score=existing.trace.importance_score,
                    recency_boost=existing.trace.recency_boost,
                    activation_boost=max(
                        existing.trace.activation_boost,
                        result.trace.activation_boost,
                    ),
                    graph_path=result.trace.graph_path or existing.trace.graph_path,
                    provenance=result.trace.provenance or existing.trace.provenance,
                    provenance_confidence=max(
                        existing.trace.provenance_confidence,
                        result.trace.provenance_confidence,
                    ),
                    trace_metadata={
                        **existing.trace.trace_metadata,
                        **result.trace.trace_metadata,
                    },
                ),
            )

        return list(merged.values())

    # ──────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────

    def _expand_top_k(self, top_k: int) -> int:
        """Widens candidate pool to improve recall before final trim."""
        return min(
            self._CANDIDATE_POOL_MAX,
            top_k * self._CANDIDATE_POOL_MULTIPLIER,
        )

    @staticmethod
    def _elapsed(started: float) -> float:
        return (time.perf_counter() - started) * 1000

    @staticmethod
    def _coerce_types(
        memory_types: list[MemoryTypeEnum] | None,
    ) -> set[str] | None:
        if not memory_types:
            return None
        return {RetrievalEngine._type_value(t) for t in memory_types}

    @staticmethod
    def _type_value(memory_type: Any) -> str:
        return memory_type.value if hasattr(memory_type, "value") else str(memory_type)

    @staticmethod
    def _validate_inputs(
        *,
        query: str,
        top_k: int,
        min_importance: float,
    ) -> None:

        if not query.strip():
            raise EmptyQueryError("Query cannot be empty")

        if top_k <= 0:
            raise RetrievalValidationError("top_k must be > 0")

        if top_k > 100:
            raise RetrievalValidationError("top_k cannot exceed 100")

        if not 0.0 <= min_importance <= 1.0:
            raise RetrievalValidationError("min_importance must be within [0.0, 1.0]")

