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
    ):
        self.vector_retriever = vector_retriever
        self.memory_store = memory_store
        self.context_assembler = context_assembler
    
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

