from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from aimemoryos.graph.traversal import (
    GraphStore,
    TraversalResult,
    bfs_traversal,
    DECAY_PER_HOP,
    ACTIVATION_BOOST_CAP,
)


class MockGraphStore:
    def __init__(self, return_records: list[dict] | None = None, raise_error: bool = False):
        self.return_records = return_records or []
        self.raise_error = raise_error
        self.last_call: dict = {}

    async def bfs_traversal_async(
        self,
        seeds: list[str],
        max_hops: int,
        limit: int,
    ) -> list[dict]:
        self.last_call = {"seeds": seeds, "max_hops": max_hops, "limit": limit}
        if self.raise_error:
            raise RuntimeError("Database connection error")
        return self.return_records


def test_traversal_result_attributes():
    res = TraversalResult(start_id="seed1", end_id="nodeA", hop=1, confidence=0.8)
    assert res.start_id == "seed1"
    assert res.end_id == "nodeA"
    assert res.hop == 1
    assert res.activation_boost == DECAY_PER_HOP ** 1
    assert res.traversal_score == pytest.approx(0.8 * 0.5)

    data = res.to_dict()
    assert data["start_id"] == "seed1"
    assert data["end_id"] == "nodeA"
    assert data["hop"] == 1
    assert data["activation_boost"] == 0.5


def test_traversal_result_activation_cap():
    # Hop 0 or artificial values
    res = TraversalResult(start_id="seed1", end_id="nodeA", hop=0, confidence=1.0)
    assert res.activation_boost <= ACTIVATION_BOOST_CAP


import asyncio


def test_bfs_traversal_empty_seeds():
    async def _test():
        store = MockGraphStore()
        results = await bfs_traversal(store=store, seeds=[])
        assert results == []
    asyncio.run(_test())


def test_bfs_traversal_cycle_and_deduplication():
    async def _test():
        raw_records = [
            {"start_id": "seed1", "end_id": "nodeA", "hop": 1, "confidence": 1.0},
            {"start_id": "seed1", "end_id": "seed1", "hop": 1, "confidence": 1.0},
            {"start_id": "seed1", "end_id": "nodeB", "hop": 2, "confidence": 1.0},
            {"start_id": "seed1", "end_id": "nodeA", "hop": 2, "confidence": 0.9},
        ]
        store = MockGraphStore(return_records=raw_records)

        results = await bfs_traversal(store=store, seeds=["seed1"], max_hops=2, limit=10)

        end_ids = [r["end_id"] for r in results]
        assert "seed1" not in end_ids
        assert "nodeA" in end_ids
        assert "nodeB" in end_ids
        assert len(end_ids) == 2

        node_a_res = next(r for r in results if r["end_id"] == "nodeA")
        assert node_a_res["hop"] == 1
        assert node_a_res["activation_boost"] == 0.5
    asyncio.run(_test())


def test_bfs_traversal_limit_and_sorting():
    async def _test():
        raw_records = [
            {"start_id": "seed1", "end_id": f"node_{i}", "hop": 1, "confidence": float(i) / 10.0}
            for i in range(1, 10)
        ]
        store = MockGraphStore(return_records=raw_records)

        results = await bfs_traversal(store=store, seeds=["seed1"], max_hops=2, limit=3)
        assert len(results) == 3
        scores = [r["traversal_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    asyncio.run(_test())


def test_bfs_traversal_backend_error_handling():
    async def _test():
        store = MockGraphStore(raise_error=True)
        results = await bfs_traversal(store=store, seeds=["seed1"])
        assert results == []
    asyncio.run(_test())
