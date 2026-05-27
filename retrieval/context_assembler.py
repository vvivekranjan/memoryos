from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.engine import MemoryResult, RetrievalTrace

@dataclass(slots=True)
class ContextBlock:
    """
    Final prompt-ready retrieval payload.

    This is the structured output passed to:
    - SDK
    - orchestration layer
    - prompt builder
    - LLM adapters

    IMPORTANT:
    ContextBlock is deterministic.

    No retrieval logic.
    No storage calls.
    No embedding calls.
    """

    query: str
    memories: list["ContextMemory"]
    combined_context: str
    token_estimate: int
    retrieval_summary: dict

@dataclass(slots=True)
class ContextMemory:
    """
    Prompt-ready memory representation.
    """

    memory_id: str
    content: str
    memory_type: str
    importance_score: float
    created_at: str
    trace: "RetrievalTrace" | None = None

class ContextAssembler:
    """
    Pure transformation layer.

    Responsibilities:
    - deterministic ordering
    - prompt-safe formatting
    - ContextBlock construction
    - RetrievalTrace attachment
    - token estimation

    Does NOT:
    - retrieve memories
    - perform ANN search
    - perform fusion
    - access storage
    """

    def assemble(
        self,
        *,
        query: str,
        results: list["MemoryResult"],
        include_trace: bool = True,
    ) -> list["MemoryResult"]:

        if not results:
            return []

        query_lower = query.lower().strip()
        scored: list[tuple[float, MemoryResult]] = []
        seen_contents: set[str] = set()

        for result in results:

            memory = result.memory
            content = memory.content.strip()
            normalized = content.lower()

            # Duplicate Suppression
            fingerprint = normalized[:200]

            if fingerprint in seen_contents:
                continue

            seen_contents.add(fingerprint)

            # Base Semantic Code
            semantic_score = (
                result.trace.final_score
                if result.trace
                else 0.0
            )

            importance_boost = memory.importance_score * 0.15
            overlap_boost = 0.0

            if query_lower:

                query_tokens = {
                    token
                    for token
                    in query_lower.split()
                    if len(token) > 2
                }

                content_tokens = set(normalized.split())
                overlap = len(
                    query_tokens
                    & content_tokens
                )

                overlap_boost = overlap * 0.03

            recency_boost = 0.0

            if memory.last_accessed_at is not None:

                last_accessed_at = self._ensure_utc_datetime(
                    memory.last_accessed_at
                )

                age_seconds = (
                    (
                        datetime.now(
                            timezone.utc
                        )
                        - last_accessed_at
                    ).total_seconds()
                )

                recency_boost = max(
                    0.0,
                    0.10
                    - (
                        age_seconds
                        / 86400
                    )
                    * 0.01,
                )

            final_score = (
                semantic_score
                + importance_boost
                + overlap_boost
                + recency_boost
            )

            scored.append(
                (
                    final_score,
                    result,
                )
            )

        ordered = [
            result
            for _, result
            in sorted(
                scored,
                key=lambda x: x[0],
                reverse=True,
            )
        ]

        if not include_trace:

            return [
                replace(
                    result,
                    trace=None,
                )
                for result
                in ordered
            ]

        return ordered

    @staticmethod
    def _enum_value(value) -> str:
        if isinstance(value, Enum):
            return value.value
        return str(value)
    
    def build_context_block(
        self,
        *,
        query: str,
        results: list["MemoryResult"],
        max_memories: int = 10,
        include_trace: bool = True,
    ) -> ContextBlock:
        """
        Converts retrieval results into
        prompt-ready ContextBlock.

        Used by:
        - SDK
        - Prompt builders
        - Agent runtime
        """

        assembled = self.assemble(
            query=query,
            results=results,
            include_trace=include_trace,
        )

        assembled = assembled[:max_memories]
        context_memories: list[ContextMemory] = []
        combined_parts: list[str] = []

        for index, result in enumerate(assembled, start=1):

            memory = result.memory
            block = (
                f"[Memory {index}]\n"
                f"Type: {self._enum_value(memory.memory_type)}\n"
                f"Importance: "
                f"{memory.importance_score:.2f}\n"
                f"Content:\n"
                f"{memory.content.strip()}"
            )

            combined_parts.append(block)
            context_memory = ContextMemory(
                memory_id=str(memory.memory_id),
                content=memory.content,
                memory_type=self._enum_value(memory.memory_type),
                importance_score=memory.importance_score,
                created_at=memory.created_at.isoformat(),
                trace=result.trace
                if include_trace
                else None,
            )

            context_memories.append(context_memory)
        
        combined_context = "\n\n".join(combined_parts)
        token_estimate = (self._estimate_tokens(combined_context))

        retrieval_summary = {
            "query": query,
            "results_count": len(context_memories),
            "token_estimate": token_estimate,
            "retrievers_used": self._extract_retrievers(assembled),
        }

        return ContextBlock(
            query=query,
            memories=context_memories,
            combined_context=combined_context,
            token_estimate=token_estimate,
            retrieval_summary=retrieval_summary,
        )

    @staticmethod
    def _ensure_utc_datetime(value: datetime) -> datetime:
        """Coerce naive timestamps from DuckDB into UTC-aware datetimes."""

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)
    
    @staticmethod
    def _estimate_tokens(
        text: str
    ) -> int:
        """
        Lightweight token estimation.

        Approximation:
        1 token ~= 4 chars
        """

        if not text:
            return 0

        return max(
            1,
            int(len(text) / 4),
        )
    
    @staticmethod
    def _extract_retrievers(
        results: list["MemoryResult"],
    ) -> list[str]:
        
        retrievers: set[str] = set()

        for result in results:

            if result.trace is None:
                continue

            retrievers.update(
                result.trace.retrieved_by
            )

        return sorted(retrievers)

