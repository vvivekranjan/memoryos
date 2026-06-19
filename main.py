from __future__ import annotations

import asyncio
from typing import Any

from agents.memory_client import MemoryClient
from core.runtime import build_runtime


class Memory:
    """High-level convenience facade for ingestion and retrieval.
    Wraps the modern MemoryClient interface."""

    def __init__(self, *, chunk_size: int = 300, overlap: int = 50, min_chunk_size: int = 10):
        runtime = build_runtime(
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )
        self.client = MemoryClient(runtime=runtime)

    async def save(
        self,
        *,
        document_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Asynchronously ingest a document without blocking the event loop."""
        
        result = await self.client.ingest(
            content=content,
            agent_id="default_agent",
            metadata={"document_id": document_id}
        )
        
        return {
            "ingestion_id": str(result.ingestion_id),
            "memory_ids": [str(m) for m in result.memory_ids],
            "total_chunks": result.chunks_created,
            "duplicate_detected": result.duplicate_detected,
        }

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
            raise RuntimeError("Event loop already running; call 'client.retrieve' awaitably instead")
        except RuntimeError as e:
            if "Event loop already running" in str(e):
                raise
            
        result = asyncio.run(
            self.client.retrieve(
                query=query,
                top_k=top_k,
                min_score=score_threshold,
                agent_id="default_agent"
            )
        )

        legacy: list[dict[str, Any]] = []
        for mem_res in result.raw_result.memories:
            score = mem_res.trace.final_score if mem_res.trace is not None else 0.0
            metadata = {
                "memory_id": str(mem_res.memory.memory_id),
                "content": mem_res.memory.content,
                "agent_id": mem_res.memory.agent_id,
                "memory_type": str(mem_res.memory.memory_type),
            }
            # Add back the document_id if it exists in the memory's metadata
            if hasattr(mem_res.memory, "metadata") and isinstance(mem_res.memory.metadata, dict):
                if "document_id" in mem_res.memory.metadata:
                    metadata["document_id"] = mem_res.memory.metadata["document_id"]

            legacy.append({"score": score, "metadata": metadata})

        return legacy

    async def snapshot(self, *args: Any, **kwargs: Any) -> Any:
        """Export the SQLite event log to a portable snapshot file."""
        return await self.client.snapshot(*args, **kwargs)

    async def forget(self, *args: Any, **kwargs: Any) -> None:
        """Remove a memory across all backend stores."""
        return await self.client.forget(*args, **kwargs)


if __name__ == "__main__":
    Memory()
