from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from retrieval.context_assembler import ContextBlock

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HEADER: str = "Relevant Memory Context"
DEFAULT_SEPARATOR: str = "\n\n"
DEFAULT_MAX_MEMORIES: int = 10

_CHARS_PER_TOKEN: int = 4

@dataclass(slots=True)
class BuiltContext:
    """
    Final prompt-ready context payload returned by ContextBuilder.

    Fields
    ------
    text             : the assembled string ready for LLM injection
    memory_count     : number of memories included (≤ max_memories)
    truncated        : True when source list exceeded max_memories
    estimated_tokens : rough token count (len(text) // 4, min 1)
    """

    text: str
    memory_count: int
    truncated: bool
    estimated_tokens: int


class ContextBuilder:
    """
    Prompt-oriented context assembler.

    Responsibilities
    ----------------
    - ContextBlock → prompt string formatting
    - Memory ordering preservation (caller supplies ranked order)
    - Optional score / metadata annotation per memory block
    - Token budget estimation via char-count heuristic
    - Truncation to max_memories

    Does NOT:
    - retrieve memories
    - rerank memories
    - access any storage backend
    - mutate memories
    """

    def __init__(
        self,
        *,
        max_memories: int = DEFAULT_MAX_MEMORIES,
        separator: str = DEFAULT_SEPARATOR,
        include_metadata: bool = False,
        include_scores: bool = False,
    ) -> None:
        self.max_memories = max_memories
        self.separator = separator
        self.include_metadata = include_metadata
        self.include_scores = include_scores

    # ------------------------------------------------------------------
    # Primary build path
    # ------------------------------------------------------------------

    def build(self, context: ContextBlock) -> BuiltContext:
        """
        Convert a ContextBlock into a prompt-ready BuiltContext.

        Memory ordering is preserved from context.memories — the retrieval
        engine has already applied RRF fusion and importance weighting.
        Truncation to max_memories is applied here if the list is longer.

        Each memory is rendered by _render_memory(), which appends optional
        score and metadata lines controlled by include_scores /
        include_metadata. The full output is prefixed with DEFAULT_HEADER.
        """
        selected = context.memories[: self.max_memories]
        truncated = len(context.memories) > self.max_memories

        sections: list[str] = [DEFAULT_HEADER]
        for index, memory in enumerate(selected, start=1):
            sections.append(self._render_memory(index=index, memory=memory))

        text = self.separator.join(sections)
        return BuiltContext(
            text=text,
            memory_count=len(selected),
            truncated=truncated,
            estimated_tokens=self._estimate_tokens(text),
        )

    # ------------------------------------------------------------------
    # Lightweight raw-string path
    # ------------------------------------------------------------------

    def build_raw(self, memories: Iterable[str]) -> BuiltContext:
        """
        Assemble a BuiltContext from an iterable of pre-formatted strings.

        Truncates to max_memories. No header is prepended. Useful when the
        caller has already formatted memory strings and only needs truncation
        and token estimation (e.g. unit tests, custom formatting pipelines).
        """
        all_items = list(memories)
        truncated = len(all_items) > self.max_memories
        selected = all_items[: self.max_memories]
        text = self.separator.join(selected)
        return BuiltContext(
            text=text,
            memory_count=len(selected),
            truncated=truncated,
            estimated_tokens=self._estimate_tokens(text),
        )

    # ------------------------------------------------------------------
    # Memory rendering
    # ------------------------------------------------------------------

    def _render_memory(self, *, index: int, memory: object) -> str:
        """
        Format a single RetrievalCandidate into a prompt block.

        Always includes:
          [Memory N]
          <content>

        Conditionally includes (controlled by constructor flags):
          [score=X.XXXX]          — when include_scores=True and trace present
          [metadata={...}]        — when include_metadata=True and trace metadata present
        """
        lines: list[str] = []
        lines.append(f"[Memory {index}]")
        lines.append(getattr(memory, "content", ""))

        if self.include_scores:
            trace = getattr(memory, "trace", None)
            if trace is not None:
                score = getattr(trace, "final_score", None)
                if score is not None:
                    lines.append(f"[score={score:.4f}]")

        if self.include_metadata:
            trace = getattr(memory, "trace", None)
            metadata = getattr(trace, "trace_metadata", None) if trace else None
            if metadata:
                lines.append(f"[metadata={metadata}]")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough token count: 1 token ≈ 4 characters (min 1).

        This is a heuristic. For budget-critical prompts, run a proper
        tokeniser (e.g. tiktoken) on BuiltContext.text before submission.
        """
        return max(len(text) // _CHARS_PER_TOKEN, 1)

