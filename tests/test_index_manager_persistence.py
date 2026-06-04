from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from uuid import uuid4

import numpy as np

from vector.index_manager import IndexManager


def make_mem(memory_type: str, days_old: int = 0, tzaware: bool = True):
    mid = uuid4()
    if tzaware:
        created = datetime.now(timezone.utc) - timedelta(days=days_old)
    else:
        created = (datetime.now() - timedelta(days=days_old))

    return SimpleNamespace(memory_id=mid, created_at=created, memory_type=memory_type)


def test_persistence_and_reload(workspace_tmp_path):
    base = workspace_tmp_path / "parts"
    base.mkdir()

    im = IndexManager(dimension=3, base_path=str(base))

    m1 = make_mem("note", days_old=0)
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    im.add(memory=m1, embedding=v1, metadata={"text": "hello"})

    assert im.vector_count() == 1

    # create a fresh manager loading from the same path
    im2 = IndexManager(dimension=3, base_path=str(base))

    results = im2.search(embedding=v1, top_k=1)

    assert len(results) == 1
    assert results[0].memory_id == m1.memory_id


def test_dimension_validation(workspace_tmp_path):
    im = IndexManager(dimension=4, base_path=str(workspace_tmp_path))

    m = make_mem("note")
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    try:
        im.add(memory=m, embedding=v)
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_naive_datetime_handling(workspace_tmp_path):
    im = IndexManager(dimension=3, base_path=str(workspace_tmp_path))

    m = make_mem("note", tzaware=False)
    v = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # should not raise
    im.add(memory=m, embedding=v)

    assert im.vector_count() == 1
