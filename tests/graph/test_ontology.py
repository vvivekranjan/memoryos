from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from aimemoryos.graph.ontology import FalkorDBStore


def test_coerce_datetime():
    store = FalkorDBStore()
    # Aware datetime
    aware_dt = datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc)
    assert store._coerce_datetime(aware_dt) == aware_dt.isoformat()

    # Naive datetime gets converted to UTC aware ISO string
    naive_dt = datetime(2026, 8, 31, 12, 0, 0)
    coerced = store._coerce_datetime(naive_dt)
    assert coerced.endswith("+00:00")

    # String pass-through
    iso_str = "2026-08-31T12:00:00+00:00"
    assert store._coerce_datetime(iso_str) == iso_str

    # None falls back to current UTC timestamp
    assert isinstance(store._coerce_datetime(None), str)


def test_memory_to_node():
    store = FalkorDBStore()
    mem_id = uuid4()
    memory = SimpleNamespace(
        memory_id=mem_id,
        memory_type="EPISODIC",
        content="Alice met Bob in Paris.",
        importance_score=0.85,
        created_at=datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc),
        metadata={"session_id": "s-123", "turn": 1},
    )

    node = store._memory_to_node(memory)
    assert node["node_id"] == str(mem_id)
    assert node["entity_type"] == "EPISODIC"
    assert node["label"] == "Alice met Bob in Paris."
    assert node["importance"] == 0.85
    assert node["created_at"] == "2026-08-31T10:00:00+00:00"
    assert json.loads(node["metadata_json"]) == {"session_id": "s-123", "turn": 1}


import asyncio


def test_uninitialized_store_no_ops():
    async def _test():
        store = FalkorDBStore()
        store._initialised = False

        # All async methods should safely no-op when uninitialized
        await store.save_memory(SimpleNamespace(memory_id=uuid4()))
        await store.delete_memory(str(uuid4()))
        await store.save_edge("a", "b", "RELATES")
        await store.save_edges_batch([{"src": "a", "dst": "b"}])
        await store.increment_cooccurrence("a", "b")
        assert await store.get_node("a") is None
        assert await store.bfs_traversal_async(["a"]) == []
    asyncio.run(_test())


def test_save_and_delete_memory_queries():
    async def _test():
        store = FalkorDBStore()
        store._initialised = True
        store.graph = MagicMock()
        store.graph.query = AsyncMock()

        mem_id = uuid4()
        memory = SimpleNamespace(
            memory_id=mem_id,
            memory_type="SEMANTIC",
            content="Knowledge fact",
            importance_score=0.9,
        )

        await store.save_memory(memory)
        store.graph.query.assert_called_once()
        cypher_call = store.graph.query.call_args[0][0]
        assert "MERGE (n:Entity {node_id: $node_id})" in cypher_call

        # Test delete
        store.graph.query.reset_mock()
        await store.delete_memory(str(mem_id))
        store.graph.query.assert_called_once()
        delete_cypher = store.graph.query.call_args[0][0]
        assert "DETACH DELETE n" in delete_cypher
    asyncio.run(_test())


def test_save_edges_batch_groups_by_type():
    async def _test():
        store = FalkorDBStore()
        store._initialised = True
        store.graph = MagicMock()
        store.graph.query = AsyncMock()

        edges = [
            {"src": "1", "dst": "2", "type": "RELATES", "weight": 0.9, "relation": "LIKES"},
            {"src": "2", "dst": "3", "type": "CAUSES", "weight": 0.7, "relation": "LEADS_TO"},
            {"src": "1", "dst": "3", "type": "UNKNOWN_CUSTOM", "weight": 0.5, "relation": "KNOWS"},
        ]

        await store.save_edges_batch(edges)
        # UNKNOWN_CUSTOM falls back to RELATES, so there are 2 query calls: RELATES and CAUSES
        assert store.graph.query.call_count == 2
    asyncio.run(_test())


def test_bfs_traversal_async_query():
    async def _test():
        store = FalkorDBStore()
        store._initialised = True
        store.graph = MagicMock()

        mock_result = MagicMock()
        mock_result.result_set = [["seed1", "nodeA", 1], ["seed1", "nodeB", 2]]
        store.graph.query = AsyncMock(return_value=mock_result)

        records = await store.bfs_traversal_async(seeds=["seed1"], max_hops=2, limit=10)
        assert len(records) == 2
        assert records[0] == {"start_id": "seed1", "end_id": "nodeA", "hop": 1}
        assert records[1] == {"start_id": "seed1", "end_id": "nodeB", "hop": 2}
    asyncio.run(_test())
