from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from retrieval.context_assembler import (
    ContextBlock,
)

DEFAULT_HEADER = "Relevant Memory Context"
DEFAULT_SEPARATOR = "\n\n"
DEFAULT_MAX_MEMORIES = 10


@dataclass(slots=True)
class BuiltContext:
    """
    Final prompt-ready context payload.
    """

    text: str
    memory_count: int
    truncated: bool
    estimated_tokens: int

class ContextBuilder:
    """
    Prompt-oriented context assembler.

    Responsibilities:
    - ContextBlock formatting
    - memory ordering
    - prompt-safe rendering
    - token budgeting
    - metadata suppression

    Does NOT:
    - retrieve memories
    - rerank memories
    - access storage
    - mutate memories
    """

    def __init__(
        self,
        *,
        max_memories: int = DEFAULT_MAX_MEMORIES,
        separator: str = DEFAULT_SEPARATOR,
        include_metadata: bool = False,
        include_scores: bool = False,
    ):
        self.max_memories = max_memories
        self.separator = separator
        self.include_metadata = include_metadata
        self.include_scores = include_scores

    def build(
        self,
        context: ContextBlock,
    ) -> BuiltContext:
        """
        Converts ContextBlock into
        prompt-ready context string.
        """

        selected = context.memories[:self.max_memories]
        truncated = len(context.memories) > self.max_memories
        sections: list[str] = [DEFAULT_HEADER]

        for index, memory in enumerate(
            selected,
            start=1,
        ):

            section = (
                self._render_memory(
                    index=index,
                    memory=memory,
                )
            )

            sections.append(section)

        text = self.separator.join(sections)

        estimated_tokens = self._estimate_tokens(text)

        return BuiltContext(
            text=text,
            memory_count=len(selected),
            truncated=truncated,
            estimated_tokens=(estimated_tokens),
        )
    
    def _render_memory(
        self,
        *,
        index: int,
        memory,
    ) -> str:
        """
        Formats individual memory.
        """

        lines: list[str] = []
        lines.append(f"[Memory {index}]")
        lines.append(memory.content)

        if (
            self.include_scores
            and memory.trace
        ):
            lines.append(
                (
                    "[score="
                    f"{memory.trace.final_score:.4f}"
                    "]"
                )
            )

        if self.include_metadata:

            metadata = (
                memory.trace.trace_metadata
                if memory.trace
                else None
            )

            if metadata:

                lines.append(
                    (
                        "[metadata="
                        f"{metadata}"
                        "]"
                    )
                )

        return "\n".join(lines)

    @staticmethod
    def _estimate_tokens(
        text: str,
    ) -> int:
        """
        Rough token estimation.

        Approx:
        1 token ≈ 4 chars
        """

        return max(
            len(text) // 4,
            1,
        )

    def build_raw(
        self,
        memories: Iterable[str],
    ) -> BuiltContext:
        """
        Lightweight raw-string builder.
        """

        selected = list(memories)

        truncated = len(selected) > self.max_memories

        selected = selected[: self.max_memories]

        text = self.separator.join(
            selected
        )

        return BuiltContext(
            text=text,
            memory_count=len(selected),
            truncated=truncated,
            estimated_tokens=(
                self._estimate_tokens(
                    text
                )
            ),
        )

