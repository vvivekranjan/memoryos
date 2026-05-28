from __future__ import annotations

import io
import json
import logging
from datetime import (
    datetime,
    timezone,
)

import pytest

from utils.hashing import (
    short_hash,
    verify_hash,
    sha256_text,
)
from utils.logger import (
    JsonFormatter,
    clear_request_id,
    log_exception,
    reset_request_id,
    set_request_id,
)
from utils.metrics import MetricsRegistry
from utils.scoring import (
    reciprocal_rank_fusion,
    recency_score,
    salience_score,
)


def test_hashing_validation_and_verify() -> None:
    value = "Hello world"
    expected = sha256_text(value)

    assert verify_hash(value=value, expected_hash=expected)
    assert not verify_hash(value=value, expected_hash="deadbeef")

    with pytest.raises(ValueError):
        short_hash(value, length=0)

    with pytest.raises(ValueError):
        short_hash(value, length=65)


def test_scoring_timezone_and_bounds() -> None:
    # naive timestamp should be treated safely as UTC
    score = recency_score(
        timestamp=datetime.now(
            timezone.utc
        )
    )
    assert 0.0 <= score <= 1.0

    # inputs over 1.0 are clamped before weighted composition
    salience = salience_score(
        semantic_similarity=2.0,
        importance=5.0,
        recency=3.0,
    )
    assert 0.0 <= salience <= 1.0

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rank=0)


def test_metrics_label_isolation_and_prometheus_labels() -> None:
    registry = MetricsRegistry()

    c_ok = registry.counter(
        "requests_total",
        labels={"status": "ok"},
    )
    c_err = registry.counter(
        "requests_total",
        labels={"status": "error"},
    )

    c_ok.inc()
    c_err.inc(2)

    exported = registry.export()
    assert len(exported["counters"]) == 2

    prom = registry.prometheus()
    assert 'requests_total{status="ok"} 1.0' in prom
    assert 'requests_total{status="error"} 2.0' in prom


def test_logger_request_context_and_explicit_exception() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("test.logger")
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    token = set_request_id("req-1")
    try:
        err = RuntimeError("boom")
        log_exception(
            logger,
            message="failed",
            exc=err,
            component="unit-test",
        )
    finally:
        reset_request_id(token)
        clear_request_id()

    raw = stream.getvalue().strip()
    payload = json.loads(raw)

    assert payload["request_id"] == "req-1"
    assert payload["exception"]["type"] == "RuntimeError"
    assert payload["component"] == "unit-test"
