from __future__ import annotations

from dataclasses import dataclass

import re

class ChunkingError(Exception):
    """Base chunking error."""


class EmptyChunkError(ChunkingError):
    """Raised when chunk generation fails."""

@dataclass(slots=True)
class Chunk:
    """
    Atomic retrieval unit.

    - text only
    - deterministic boundaries

    Future:
    - multimodal spans
    - semantic references
    - graph anchors
    """

    chunk_index: int
    content: str
    token_count: int
    char_count: int
    start_char: int
    end_char: int

@dataclass(slots=True)
class ChunkingResult:
    """
    Final chunking output.
    """

    chunks: list[Chunk]
    total_chunks: int
    total_tokens: int
    overlap_tokens: int

class Chunker:

    SENTENCE_SPLIT_REGEX = r"(?<=[.!?])\s+"

    def __init__(
        self,
        *,
        chunk_size: int = 300,
        overlap: int = 50,
        min_chunk_size: int = 50,
    ):

        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be > 0")

        if overlap < 0:
            raise ChunkingError("overlap must be >= 0")

        if overlap >= chunk_size:
            raise ChunkingError("overlap must be < chunk_size")

        if min_chunk_size <= 0:
            raise ChunkingError("min_chunk_size must be > 0")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk(
        self,
        content: str,
    ) -> ChunkingResult:
        """
        Main chunking entrypoint.

        Strategy:
        1. sentence split
        2. semantic grouping
        3. overlap stitching
        4. fallback split if needed
        """

        self._validate_content(content)

        sentences = self._split_sentences(content)

        if not sentences:
            raise EmptyChunkError("No sentences generated")
        
        chunks = self._build_chunks(sentences)

        total_tokens = sum(
            chunk.token_count
            for chunk in chunks
        )

        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            overlap_tokens=self.overlap,
        )
    
    def _split_sentences(
        self,
        content: str,
    ) -> list[str]:
        """
        Deterministic sentence splitting.

        regex-based
        """

        sentences = re.split(self.SENTENCE_SPLIT_REGEX, content)

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]
    
    def _build_chunks(
        self,
        sentences: list[str],
    ) -> list[Chunk]:
        """
        Builds semantically grouped chunks.
        """

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        char_cursor = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = (
                self._estimate_tokens(
                    sentence
                )
            )
            
            # Chunk Full
            if (
                current_sentences
                and current_tokens
                + sentence_tokens
                > self.chunk_size
            ):
                chunk = self._create_chunk(
                    chunk_index=chunk_index,
                    sentences=current_sentences,
                    start_char=char_cursor,
                )
            
                chunks.append(chunk)

                chunk_index += 1

                # Overlap
                overlap_sentences = (
                    self._build_overlap(
                        current_sentences
                    )
                )

                current_sentences = overlap_sentences

                current_tokens = sum(
                    self._estimate_tokens(s)
                    for s in current_sentences
                )

                char_cursor = (
                    chunk.end_char
                    - len(
                        " ".join(
                            overlap_sentences
                        )
                    )
                )
            
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Final Chunk
        if current_sentences:

            chunk = self._create_chunk(
                chunk_index=chunk_index,
                sentences=current_sentences,
                start_char=char_cursor,
            )

            chunks.append(chunk)
        
        filtered_chunks = [
            chunk for chunk in chunks
            if chunk.token_count >= self.min_chunk_size or len(chunks) == 1
        ]

        return filtered_chunks

    def _create_chunk(
        self,
        *,
        chunk_index: int,
        sentences: list[str],
        start_char: int,
    ) -> Chunk:
        
        content = " ".join(sentences)
        token_count = self._estimate_tokens(content)
        char_count = len(content)
        end_char = start_char + char_count

        return Chunk(
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            char_count=char_count,
            start_char=start_char,
            end_char=end_char,
        )
    
    def _build_overlap(
        self,
        sentences: list[str],
    ) -> list[str]:
        """
        Builds overlap window.

        Keeps trailing sentences
        up to overlap token limit.
        """

        overlap_sentences: list[str] = []
        running_tokens = 0

        for sentence in reversed(sentences):

            tokens = self._estimate_tokens(sentence)

            if (running_tokens + tokens > self.overlap):
                break

            overlap_sentences.insert(
                0,
                sentence,
            )

            running_tokens += tokens

        return overlap_sentences
    
    @staticmethod
    def _estimate_tokens(
        text: str,
    ) -> int:
        """
        Lightweight deterministic estimate.

        Approximation:
        1 token ~= 4 chars
        """

        return max(
            1,
            int(len(text) / 4),
        )
    
    @staticmethod
    def _validate_content(
        content: str,
    ) -> None:

        if not isinstance(content, str):
            raise ChunkingError(
                "Content must be string"
            )

        if not content.strip():
            raise ChunkingError(
                "Content cannot be empty"
            )

