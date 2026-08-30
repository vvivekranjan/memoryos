from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from aimemoryos.graph.contradiction_detector import (
    ContradictionResult,
    _contradiction_score,
    _find_contradictions,
    _is_active,
    _normalise,
    _spo,
    _write_contradiction_event,
    detect_and_flag,
    detect_for_cluster,
    flag_correction,
    mark_resolved,
)


def make_semantic_memory(
    entity: str,
    relation: str,
    object_value: str,
    confidence: float = 0.8,
    lifecycle_state: str = "ACTIVE",
    mid=None,
):
    return SimpleNamespace(
        memory_id=mid or uuid4(),
        entity=entity,
        relation=relation,
        object_value=object_value,
        confidence=confidence,
        lifecycle_state=lifecycle_state,
        importance_score=0.7,
        contradicted_by=[],
    )


def test_normalise_and_spo():
    assert _normalise("  Alice  ") == "alice"
    assert _normalise(None) == ""

    mem = make_semantic_memory(" Alice ", " LIKES ", " Ice Cream ")
    assert _spo(mem) == ("alice", "likes", "ice cream")


def test_is_active():
    assert _is_active(make_semantic_memory("A", "B", "C", lifecycle_state="ACTIVE"))
    assert not _is_active(make_semantic_memory("A", "B", "C", lifecycle_state="ARCHIVED"))
    assert not _is_active(make_semantic_memory("A", "B", "C", lifecycle_state="PRUNED"))


def test_contradiction_score():
    # max confidence 0.9 -> score 0.1
    assert _contradiction_score(0.9, 0.5) == pytest.approx(0.1)
    # max confidence 0.6 -> score 0.4
    assert _contradiction_score(0.6, 0.4) == pytest.approx(0.4)


def test_find_contradictions_pair():
    mem1 = make_semantic_memory("Python", "RELEASED_IN", "1991", confidence=0.9)
    mem2 = make_semantic_memory("Python", "RELEASED_IN", "1995", confidence=0.6)
    mem3 = make_semantic_memory("Python", "CREATOR", "Guido", confidence=0.9)

    contradictions = _find_contradictions([mem1, mem2, mem3])
    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.entity == "python"
    assert c.relation == "released_in"
    assert c.memory_id_a == str(mem1.memory_id)  # Winner (higher conf)
    assert c.memory_id_b == str(mem2.memory_id)  # Flagged loser
    assert c.object_a == "1991"
    assert c.object_b == "1995"


import asyncio


def test_write_contradiction_and_mark_resolved():
    async def _test():
        graph_store = MagicMock()
        graph_store._initialised = True
        graph_store.graph = MagicMock()
        graph_store.graph.query = AsyncMock()

        event = ContradictionResult(
            memory_id_a=str(uuid4()),
            memory_id_b=str(uuid4()),
            entity="user",
            relation="favorite_color",
            object_a="blue",
            object_b="green",
            score=0.3,
        )

        await _write_contradiction_event(graph_store, event)
        graph_store.graph.query.assert_called_once()
        assert "MERGE (e:ContradictionEvent" in graph_store.graph.query.call_args[0][0]

        # Test mark resolved
        graph_store.graph.query.reset_mock()
        await mark_resolved(event.event_id, graph_store)
        graph_store.graph.query.assert_called_once()
        assert "SET e.resolved = true" in graph_store.graph.query.call_args[0][0]
    asyncio.run(_test())


def test_detect_and_flag_orchestration():
    async def _test():
        graph_store = MagicMock()
        graph_store._initialised = True
        graph_store.graph = MagicMock()
        graph_store.graph.query = AsyncMock()

        duckdb_store = MagicMock()
        mem1 = make_semantic_memory("User", "LIVES_IN", "London", confidence=0.9)
        mem2 = make_semantic_memory("User", "LIVES_IN", "Berlin", confidence=0.7)
        duckdb_store.get_memory.return_value = mem2

        results = await detect_and_flag(
            new_memory=mem1,
            existing_memories=[mem2],
            kuzu_store=graph_store,
            duckdb_store=duckdb_store,
        )

        assert len(results) == 1
        assert results[0].entity == "user"
        assert results[0].relation == "lives_in"
        graph_store.graph.query.assert_called()
    asyncio.run(_test())
