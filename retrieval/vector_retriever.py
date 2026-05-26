from __future__ import annotations

from vector.embedder import Embedder
from storage.faiss_store import FAISSStore, SearchResult
from storage.orchestrator import StorageOrchestrator
from typing import List, Dict, Any
from uuid import UUID
import asyncio

class VectorRetriever:
    """Handles query-based retrieval from the vector store and orchestrator"""

    def __init__(
        self,
        faiss_store: FAISSStore,
        orchestrator: StorageOrchestrator,
        embedder: Embedder,
    ):
        
        self.faiss_store = faiss_store
        self.orchestrator = orchestrator
        self.embedder = embedder
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information for a query
        """

        if not query or not query.strip():
            return []

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aretrieve(query=query, top_k=top_k, score_threshold=score_threshold)
            )

        # If there is an active running loop, instruct caller to use `aretrieve` directly.
        raise RuntimeError("Event loop already running; call 'aretrieve' awaitably instead")

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Async retrieval path that awaits the async embedder and FAISS search."""

        if not query or not query.strip():
            return []

        try:
            embeddings = await self.embedder.generate_embeddings([query])
            query_embedding = embeddings[0]

            results: List[SearchResult] = await self.faiss_store.search_async(
                query_embedding=query_embedding.tolist(), top_k=top_k
            )

            filtered_results: List[Dict[str, Any]] = []
            for result in results:
                if result["score"] >= score_threshold:
                    try:
                        mem_id_str = result["metadata"].get("memory_id")
                        if mem_id_str:
                            mem = self.orchestrator.retrieve_memory(UUID(mem_id_str))
                            filtered_results.append({
                                "score": result["score"],
                                "metadata": {
                                    "memory_id": str(mem.memory_id),
                                    "content": mem.content,
                                    "agent_id": mem.agent_id,
                                    "memory_type": str(mem.memory_type),
                                }
                            })
                    except Exception as exc:
                        print(f"Failed to retrieve {result['metadata'].get('memory_id')}: {exc}")

            return filtered_results

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
