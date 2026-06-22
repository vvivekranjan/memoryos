from __future__ import annotations

import os

from pathlib import Path

class MemoryConfig:
    data_dir: str = str(Path(__file__).resolve().parents[2] / ".aimemoryos")
    embedding_model: str = "all-mpnet-base-v2"
    default_dimension: int = 768
    spacy_model: str = "en_core_web_sm"
    rrf_k: int = 60
    chunk_size: int = 512
    chunk_overlap: int = 64
    default_agent_id: str = "default"
    episodic_decay_rate: float = 0.1
    semantic_decay_rate: float = 0.01

    def __init__(self, **kwargs):
        env = os.environ.get
        for key, default in self._defaults().items():
            setattr(self, key, env(f"AIMEMORYOS_{key.upper()}", kwargs.get(key, default)))

    @classmethod
    def _defaults(cls):
        return {
            "data_dir": cls.data_dir,
            "embedding_model": cls.embedding_model,
            "default_dimension": cls.default_dimension,
            "spacy_model": cls.spacy_model,
            "rrf_k": cls.rrf_k,
            "chunk_size": cls.chunk_size,
            "chunk_overlap": cls.chunk_overlap,
            "default_agent_id": cls.default_agent_id,
            "episodic_decay_rate": cls.episodic_decay_rate,
            "semantic_decay_rate": cls.semantic_decay_rate,
        }

config = MemoryConfig()
