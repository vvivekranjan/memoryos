from __future__ import annotations

from sentence_transformers import SentenceTransformer
from typing import List

import numpy as np
import asyncio

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

DEFAULT_DIMENSION = 768

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

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = DEFAULT_DIMENSION
        self._load_model()
    
    def _load_model(self):
        """Load the Sentence Transformer Model"""

        try:
            print(f"Loading embedding Model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            try:
                dimension = self.model.get_embedding_dimension()
            except Exception:
                dimension = getattr(self.model, "get_embedding_dimension", lambda: DEFAULT_DIMENSION)()
            self.embedding_dimension = int(dimension)
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
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
            raise ValueError("Model not loaded")

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

