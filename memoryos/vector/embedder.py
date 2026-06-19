from __future__ import annotations

from sentence_transformers import SentenceTransformer
from typing import List

import numpy as np
import asyncio

from memoryos.core.config import MemoryConfig
from memoryos.utils.logger import get_logger

DEFAULT_DIMENSION = MemoryConfig.default_dimension

class EmbeddingError(Exception):
    """Base embedding error."""


class EmptyEmbeddingInputError(
    EmbeddingError
):
    """Raised when input empty."""


class EmbeddingDimensionError(
    EmbeddingError
):
    """Raised when embedding invalid."""

class Embedder:
    """
    Embedding infrastructure.

    Responsibilities:
    - embedding generation
    - embedding normalization
    - embedding batching
    """

    def __init__(self, model_name: str = MemoryConfig.embedding_model):
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = MemoryConfig.default_dimension
        self._logger = get_logger(__name__, subsystem="vector.embedder")
        # defer heavy model loading until needed
    
    def _load_model(self):
        """Load the Sentence Transformer Model"""

        try:
            self._logger.info("loading model", extra={"model_name": self.model_name})
            self.model = SentenceTransformer(self.model_name)
            try:
                dimension = self.model.get_embedding_dimension()
            except Exception:
                dimension = getattr(self.model, "get_embedding_dimension", lambda: MemoryConfig.default_dimension)()
            self.embedding_dimension = int(dimension)
            self._logger.info("model loaded", extra={"embedding_dimension": self.embedding_dimension})
        except Exception as e:
            self._logger.exception("error loading model", extra={"model_name": self.model_name})
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Batch embedding generation (async).

        Returns a numpy array of shape (n_texts, embedding_dim) with dtype float32.
        """

        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        for text in texts:
            self._validate_text(text)

        if not self.model:
            # Lazy-load the model on first use
            try:
                self._load_model()
            except Exception as exc:
                raise ValueError("Model not available") from exc

        # Offload the blocking model.encode call to a thread.
        vectors = await asyncio.to_thread(
            self.model.encode, texts, convert_to_numpy=True
        )

        arr = np.asarray(vectors, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != self.embedding_dimension:
            raise EmbeddingDimensionError(
                f"Expected dimension {self.embedding_dimension}, got {arr.shape}"
            )

        return arr.astype(np.float32)
    
    # _prepare_embedding removed — generate_embeddings returns numpy arrays directly

    @staticmethod
    def _validate_text(
        text: str,
    ) -> None:
        
        if not isinstance(text, str):
            raise EmptyEmbeddingInputError(
                "Text must be string"
            )

        if not text.strip():
            raise EmptyEmbeddingInputError(
                "Text cannot be empty"
            )

