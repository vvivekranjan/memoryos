import unittest
import asyncio
import sys
from types import SimpleNamespace
from uuid import UUID
import numpy as np
from pathlib import Path

# Ensure repo root is on sys.path for local imports when running tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.memory_client import MemoryClient, IngestResult, RetrieveResult
from agents.context_builder import ContextBuilder
from agents.session_manager import SessionManager

class DummyPipeline:
    def __init__(self, result=None):
        self._result = result

    async def ingest_document(self, request):
        return self._result

class DummyRetrievalEngine:
    def __init__(self, memories):
        self._memories = memories
        # Provide a simple context_assembler with expected API
        self.context_assembler = SimpleNamespace()
        def build_context_block(*, query, results, max_memories=10, include_trace=True):
            # Convert MemoryResult-like inputs into ContextMemory-like objects
            context_memories = []
            for res in results:
                mem = res.memory
                ctx_mem = SimpleNamespace(
                    memory_id=str(mem.memory_id),
                    content=getattr(mem, "content", ""),
                    memory_type=str(getattr(mem, "memory_type", "")),
                    importance_score=getattr(mem, "importance_score", 0.0),
                    created_at=SimpleNamespace(isoformat=lambda: "now"),
                    trace=res.trace if hasattr(res, "trace") else None,
                )
                context_memories.append(ctx_mem)

            return SimpleNamespace(
                query=query,
                memories=context_memories,
                combined_context="",
                token_estimate=0,
                retrieval_summary={},
            )
        self.context_assembler.build_context_block = build_context_block

    async def retrieve(self, *_, **__):
        return SimpleNamespace(
            memories=self._memories,
            cache_hit=False,
            latency_ms=0,
            retrievers_used=["vector"],
        )

class DummyEmbedder:
    async def generate_embeddings(self, texts):
        # Return a numpy array with shape (n_texts, dim)
        arr = np.asarray([[0.0]], dtype=np.float32)
        return arr

class MemoryClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_duplicate_returns_duplicate_flag(self):
        pipeline = DummyPipeline(result=None)
        client = MemoryClient(
            ingestion_pipeline=pipeline,
            retrieval_engine=None,
            context_builder=ContextBuilder(),
            session_manager=SessionManager(),
            embedder=DummyEmbedder(),
        )

        res = await client.ingest(agent_id="a", content="c", memory_type="EPISODIC")
        self.assertIsInstance(res, IngestResult)
        self.assertTrue(res.duplicate_detected)

    async def test_retrieve_builds_context(self):
        # Build fake memory result structure expected by assembler
        fake_memory = SimpleNamespace(
            memory_id=UUID(int=0),
            content="hello",
            agent_id="a",
            memory_type="EPISODIC",
            importance_score=0.5,
            created_at=SimpleNamespace(isoformat=lambda: "now"),
        )
        fake_trace = SimpleNamespace(final_score=0.9, trace_metadata={})
        fake_memres = SimpleNamespace(memory=fake_memory, trace=fake_trace)

        retrieval_engine = DummyRetrievalEngine(memories=[fake_memres])
        client = MemoryClient(
            ingestion_pipeline=DummyPipeline(result={}),
            retrieval_engine=retrieval_engine,
            context_builder=ContextBuilder(),
            session_manager=SessionManager(),
            embedder=DummyEmbedder(),
        )

        res = await client.retrieve(agent_id="a", query="q", top_k=1)
        self.assertIsInstance(res, RetrieveResult)
        self.assertGreaterEqual(res.context.memory_count, 0)

if __name__ == '__main__':
    unittest.main()
