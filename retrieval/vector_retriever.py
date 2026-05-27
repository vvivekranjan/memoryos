from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List
from uuid import UUID

from storage.faiss_store import FAISSStore, SearchResult
from storage.orchestrator import StorageOrchestrator
from vector.embedder import Embedder


@dataclass(slots=True)
class RetrievalCandidate:
    memory_id: UUID
    score: float
    metadata: dict[str, Any]


class VectorRetriever:
    """Vector-store access layer with both low-level search and legacy query retrieval."""

    def __init__(
        self,
        faiss_store: FAISSStore,
        orchestrator: StorageOrchestrator,
        embedder: Embedder,
    ):
        self.faiss_store = faiss_store
        self.orchestrator = orchestrator
        self.embedder = embedder

    def search(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievalCandidate]:
        """Synchronously search the vector index using a precomputed embedding."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.search_async(query_embedding=query_embedding, top_k=top_k)
            )

        raise RuntimeError("Event loop already running; call 'search_async' instead")

    async def search_async(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievalCandidate]:
        """Async vector search that returns lightweight candidates for the coordinator."""

        if top_k <= 0:
            return []

        results: List[SearchResult] = await self.faiss_store.search_async(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        candidates: list[RetrievalCandidate] = []
        for result in results:
            memory_id = result["metadata"].get("memory_id")
            if not memory_id:
                continue

            try:
                candidate_id = UUID(str(memory_id))
            except (TypeError, ValueError):
                continue

            candidates.append(
                RetrievalCandidate(
                    memory_id=candidate_id,
                    score=float(result["score"]),
                    metadata=dict(result["metadata"]),
                )
            )

        return candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Legacy synchronous convenience wrapper for query-string retrieval."""

        if not query or not query.strip():
            return []

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aretrieve(
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
            )

        raise RuntimeError("Event loop already running; call 'aretrieve' awaitably instead")

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Async convenience wrapper for query-string retrieval."""

        if not query or not query.strip():
            return []

        try:
            embeddings = await self.embedder.generate_embeddings([query])
            candidates = await self.search_async(
                query_embedding=embeddings[0].tolist(),
                top_k=top_k,
            )

            if not candidates:
                return []

            filtered_candidates = [
                candidate
                for candidate in candidates
                if candidate.score >= score_threshold
            ]

            if not filtered_candidates:
                return []

            memories = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self.orchestrator.retrieve_memory,
                        candidate.memory_id,
                    )
                    for candidate in filtered_candidates
                )
            )

            filtered_results: List[Dict[str, Any]] = []
            for candidate, memory in zip(filtered_candidates, memories):
                filtered_results.append(
                    {
                        "score": candidate.score,
                        "metadata": {
                            "memory_id": str(memory.memory_id),
                            "content": memory.content,
                            "agent_id": memory.agent_id,
                            "memory_type": str(memory.memory_type),
                        },
                    }
                )

            return filtered_results

        except Exception as exc:
            print(f"Error during retrieval: {exc}")
            return []
