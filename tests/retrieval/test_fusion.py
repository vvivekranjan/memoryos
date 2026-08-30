from uuid import uuid4
from types import SimpleNamespace
import pytest

from aimemoryos.retrieval.engine import MemoryResult, RetrievalTrace
from aimemoryos.retrieval.fusion import (
    calculate_rrf_score,
    calculate_composite_score,
    fuse_memory_results,
)


def make_fake_memory_result(channel: str, rank: int, score: float = 0.8, mid=None):
    memory_id = mid or uuid4()
    memory = SimpleNamespace(
        memory_id=memory_id,
        content=f"Memory content for {memory_id}",
        importance_score=0.7,
        provenance="OBSERVED",
        provenance_confidence=1.0,
    )
    trace = RetrievalTrace(
        memory_id=memory_id,
        final_score=score,
        retrieved_by=[channel],
        vector_rank=rank if channel == "vector" else None,
        graph_rank=rank if channel == "graph" else None,
        activation_boost=0.1 if channel == "graph" else 0.0,
    )
    return MemoryResult(memory=memory, trace=trace)


def test_rrf_score_calculation():
    score_top1 = calculate_rrf_score({"vector": 1}, k=60)
    assert score_top1 == pytest.approx(0.7 * (1.0 / 61), rel=1e-4)

    # Item present in both vector (rank 1) and graph (rank 2)
    score_dual = calculate_rrf_score({"vector": 1, "graph": 2}, k=60)
    expected = (0.7 * (1.0 / 61)) + (0.3 * (1.0 / 62))
    assert score_dual == pytest.approx(expected, rel=1e-4)


def test_fuse_memory_results_preserves_uuid_and_ranks():
    shared_mid = uuid4()
    vector_only_mid = uuid4()
    graph_only_mid = uuid4()

    vector_results = [
        make_fake_memory_result("vector", rank=1, mid=shared_mid),
        make_fake_memory_result("vector", rank=2, mid=vector_only_mid),
    ]

    graph_results = [
        make_fake_memory_result("graph", rank=1, mid=shared_mid),
        make_fake_memory_result("graph", rank=2, mid=graph_only_mid),
    ]

    fused = fuse_memory_results({"vector": vector_results, "graph": graph_results})

    assert len(fused) == 3
    # Shared item should rank highest because it appeared in both channels
    assert fused[0].memory.memory_id == shared_mid
    assert set(fused[0].trace.retrieved_by) == {"graph", "vector"}
    assert fused[0].trace.vector_rank == 1
    assert fused[0].trace.graph_rank == 1
    # Type check: memory_id must be a UUID instance, not a string
    assert not isinstance(fused[0].trace.memory_id, str)
    assert fused[0].trace.memory_id == shared_mid
