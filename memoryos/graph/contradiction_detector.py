from __future__ import annotations

import uuid
import logging

from datetime import datetime, timezone
from itertools import combinations
from typing import TYPE_CHECKING, Any, Sequence

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Imported at type-check time only to avoid circular imports at runtime.
    # graph/ontology.py and storage/duckdb_store.py are both M2+ modules.
    from graph.ontology import KuzuDBStore
    from storage.duckdb_store import DuckDBStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum score threshold to emit a ContradictionEvent.
#: Below this threshold the objects are considered close enough variants,
#: not a true contradiction (e.g. minor paraphrase differences).
CONTRADICTION_SCORE_GATE: float = 0.05

#: Lifecycle state that must be present on both memories for contradiction
#: detection to apply. Archived/pruned memories are excluded.
ACTIVE_STATE: str = "ACTIVE"


class ContradictionResult:
    """
    Output record for a single detected contradiction.

    Attributes
    ----------
    event_id      : UUID string — KuzuDB ContradictionEvent primary key
    memory_id_a   : first SemanticMemory (higher confidence = "winner")
    memory_id_b   : second SemanticMemory (lower confidence = flagged)
    entity        : normalised subject entity string
    relation      : SPO predicate
    object_a      : object_value from memory_a
    object_b      : object_value from memory_b
    score         : 1.0 - max(conf_a, conf_b); higher → stronger conflict
    resolved      : always False at creation; set True by consolidator/feedback
    created_at    : ISO-8601 UTC string
    """

    __slots__ = (
        "event_id",
        "memory_id_a",
        "memory_id_b",
        "entity",
        "relation",
        "object_a",
        "object_b",
        "score",
        "resolved",
        "created_at",
    )

    def __init__(
        self,
        memory_id_a: str,
        memory_id_b: str,
        entity: str,
        relation: str,
        object_a: str,
        object_b: str,
        score: float,
    ) -> None:
        self.event_id = str(uuid.uuid4())
        self.memory_id_a = memory_id_a
        self.memory_id_b = memory_id_b
        self.entity = entity
        self.relation = relation
        self.object_a = object_a
        self.object_b = object_b
        self.score = round(score, 6)
        self.resolved = False
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "memory_id_a": self.memory_id_a,
            "memory_id_b": self.memory_id_b,
            "entity": self.entity,
            "relation": self.relation,
            "object_a": self.object_a,
            "object_b": self.object_b,
            "score": self.score,
            "resolved": self.resolved,
            "created_at": self.created_at,
        }


def _normalise(value: str | None) -> str:
    """Lowercase + strip whitespace. Empty string on None."""
    if not value:
        return ""
    return value.strip().lower()


def _is_active(memory: Any) -> bool:
    """True when lifecycle_state is ACTIVE or absent (M1 memories have no state)."""
    state = getattr(memory, "lifecycle_state", ACTIVE_STATE)
    if state is None:
        return True
    return str(state).upper() == ACTIVE_STATE


def _spo(memory: Any) -> tuple[str, str, str]:
    """Return (entity, relation, object_value) normalised triple."""
    return (
        _normalise(getattr(memory, "entity", None)),
        _normalise(getattr(memory, "relation", None)),
        _normalise(getattr(memory, "object_value", None)),
    )


def _confidence(memory: Any) -> float:
    """Extract confidence float; default 1.0 if absent."""
    raw = getattr(memory, "confidence", 1.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def _contradiction_score(conf_a: float, conf_b: float) -> float:
    """
    Contradiction score = 1.0 - max(confidence_a, confidence_b).

    A higher confidence in at least one memory lowers the score because
    the system has a strong prior that one version is correct. Low
    confidence in both raises the score, signalling genuine ambiguity.
    """
    return 1.0 - max(conf_a, conf_b)


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------


def _find_contradictions(
    memories: Sequence[Any],
) -> list[ContradictionResult]:
    """
    Find all pairwise contradictions in a sequence of SemanticMemory objects.

    A contradiction is defined as two memories that share the same
    (entity, relation) pair but have different object_value strings,
    both in ACTIVE lifecycle state, with a score above CONTRADICTION_SCORE_GATE.

    Returns a deduplicated list of ContradictionResult objects.
    Already-seen (memory_id_a, memory_id_b) pairs are skipped to prevent
    duplicate ContradictionEvent nodes when called multiple times across
    reflection cycles.
    """
    results: list[ContradictionResult] = []
    seen_pairs: set[frozenset[str]] = set()

    for mem_a, mem_b in combinations(memories, 2):
        if not (_is_active(mem_a) and _is_active(mem_b)):
            continue

        entity_a, relation_a, object_a = _spo(mem_a)
        entity_b, relation_b, object_b = _spo(mem_b)

        # Must share entity and relation but differ on object_value.
        if not entity_a or not relation_a:
            continue
        if entity_a != entity_b or relation_a != relation_b:
            continue
        if object_a == object_b:
            continue  # Identical objects — no contradiction.

        id_a = str(getattr(mem_a, "memory_id", ""))
        id_b = str(getattr(mem_b, "memory_id", ""))
        pair_key: frozenset[str] = frozenset({id_a, id_b})
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        conf_a = _confidence(mem_a)
        conf_b = _confidence(mem_b)
        score = _contradiction_score(conf_a, conf_b)

        if score < CONTRADICTION_SCORE_GATE:
            logger.debug(
                "graph.contradiction | score below gate — skipping | "
                "pair=({}, {}) score={:.4f}",
                id_a, id_b, score,
            )
            continue

        # memory_id_a = higher-confidence memory (the "winner" side).
        if conf_a >= conf_b:
            winner_id, loser_id = id_a, id_b
            winner_obj, loser_obj = object_a, object_b
        else:
            winner_id, loser_id = id_b, id_a
            winner_obj, loser_obj = object_b, object_a

        results.append(
            ContradictionResult(
                memory_id_a=winner_id,
                memory_id_b=loser_id,
                entity=entity_a,
                relation=relation_a,
                object_a=winner_obj,
                object_b=loser_obj,
                score=score,
            )
        )

    return results


# ---------------------------------------------------------------------------
# KuzuDB + DuckDB persistence helpers
# ---------------------------------------------------------------------------


async def _write_contradiction_event(
    kuzu_store: "KuzuDBStore",
    event: ContradictionResult,
) -> None:
    """
    Write a ContradictionEvent node to KuzuDB.

    Uses MERGE on event_id so reflection cycles that re-run over the same
    memory cluster don't create duplicate nodes.
    """
    cypher = """
        MERGE (e:ContradictionEvent {event_id: $event_id})
        SET e.memory_id_a = $memory_id_a,
            e.memory_id_b = $memory_id_b,
            e.score       = $score,
            e.resolved    = $resolved,
            e.created_at  = $created_at
    """
    conn = getattr(kuzu_store, "_conn", None)
    if conn is None:
        logger.warning(
            "graph.contradiction | KuzuDB unavailable — "
            "ContradictionEvent not persisted | event_id={}",
            event.event_id,
        )
        return
    try:
        conn.execute(
            cypher,
            parameters={
                "event_id": event.event_id,
                "memory_id_a": event.memory_id_a,
                "memory_id_b": event.memory_id_b,
                "score": event.score,
                "resolved": event.resolved,
                "created_at": event.created_at,
            },
        )
        logger.info(
            "graph.contradiction | ContradictionEvent written | "
            "event_id={} memory_a={} memory_b={} score={:.4f}",
            event.event_id, event.memory_id_a, event.memory_id_b, event.score,
        )
    except Exception as exc:
        logger.warning(
            "graph.contradiction | KuzuDB write failed | "
            "event_id={} | error={}", event.event_id, exc,
        )


async def _update_contradicted_by(
    duckdb_store: "DuckDBStore",
    loser_memory_id: str,
    event_id: str,
) -> None:
    """
    Append event_id to the contradicted_by list on the weaker SemanticMemory
    row in DuckDB.

    DuckDB does not have a native list-append; we read, modify, write back.
    All failures are logged at WARNING — a DuckDB update failure is non-fatal
    for the ContradictionEvent itself (already written to KuzuDB).
    """
    try:
        existing = await duckdb_store.get_memory(loser_memory_id)
        if existing is None:
            logger.warning(
                "graph.contradiction | memory not found in DuckDB — "
                "contradicted_by not updated | memory_id={}",
                loser_memory_id,
            )
            return

        current: list[str] = list(getattr(existing, "contradicted_by", None) or [])
        if event_id in current:
            return  # Idempotent — already recorded.
        current.append(event_id)
        await duckdb_store.update_field(
            loser_memory_id, "contradicted_by", current
        )
    except Exception as exc:
        logger.warning(
            "graph.contradiction | DuckDB contradicted_by update failed | "
            "memory_id={} event_id={} | error={}",
            loser_memory_id, event_id, exc,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def detect_and_flag(
    new_memory: Any,
    existing_memories: Sequence[Any],
    kuzu_store: "KuzuDBStore",
    duckdb_store: "DuckDBStore",
) -> list[ContradictionResult]:
    """
    Check a newly ingested SemanticMemory against existing active memories
    for the same (entity, relation) pair.

    Call site: ingestion/relation_extractor.py after writing a RELATES edge.

    Parameters
    ----------
    new_memory        : the freshly ingested SemanticMemory
    existing_memories : active SemanticMemory objects for the same agent,
                        pre-filtered by entity if possible for efficiency
    kuzu_store        : open KuzuDBStore instance
    duckdb_store      : open DuckDBStore instance

    Returns
    -------
    List of ContradictionResult objects persisted during this call.
    Empty list when no contradictions found.
    """
    candidates = [m for m in existing_memories if _is_active(m)]
    candidates_plus_new = [new_memory] + candidates

    contradictions = _find_contradictions(candidates_plus_new)

    for event in contradictions:
        await _write_contradiction_event(kuzu_store, event)
        await _update_contradicted_by(duckdb_store, event.memory_id_b, event.event_id)

    if contradictions:
        logger.info(
            "graph.contradiction | detect_and_flag | "
            "memory_id={} contradictions_found={}",
            str(getattr(new_memory, "memory_id", "?")),
            len(contradictions),
        )

    return contradictions


async def detect_for_cluster(
    cluster_memories: Sequence[Any],
    kuzu_store: "KuzuDBStore",
    duckdb_store: "DuckDBStore",
) -> list[ContradictionResult]:
    """
    Run contradiction detection across all SemanticMemory objects in a
    reflection cluster.

    Call site: reflection/consolidator.py for each HDBSCAN cluster.

    All pairwise combinations within the cluster are checked. Only ACTIVE
    memories with non-empty entity and relation fields are considered.
    ContradictionEvent nodes are written to KuzuDB and contradicted_by
    lists updated in DuckDB.

    Parameters
    ----------
    cluster_memories : all memories in the cluster (may include non-Semantic
                       types — they are silently skipped via _spo() returning
                       empty strings)
    kuzu_store       : open KuzuDBStore instance
    duckdb_store     : open DuckDBStore instance

    Returns
    -------
    List of ContradictionResult objects persisted during this call.
    """
    # Filter to SemanticMemory only — others have no SPO fields.
    semantic = [
        m for m in cluster_memories
        if getattr(m, "memory_type", "") in ("SEMANTIC", "semantic")
        and _is_active(m)
    ]

    if len(semantic) < 2:
        return []

    contradictions = _find_contradictions(semantic)

    for event in contradictions:
        await _write_contradiction_event(kuzu_store, event)
        await _update_contradicted_by(duckdb_store, event.memory_id_b, event.event_id)

    logger.info(
        "graph.contradiction | detect_for_cluster | "
        "cluster_size={} semantic_count={} contradictions_found={}",
        len(cluster_memories),
        len(semantic),
        len(contradictions),
    )

    return contradictions


async def flag_correction(
    superseded_memory: Any,
    corrected_memory: Any,
    kuzu_store: "KuzuDBStore",
    duckdb_store: "DuckDBStore",
) -> ContradictionResult | None:
    """
    Handle CORRECTION feedback signal.

    Creates a ContradictionEvent between the superseded and corrected memory,
    marks the superseded memory's contradicted_by field in DuckDB.
    Archival of the superseded memory is the caller's responsibility
    (autonomous/feedback_collector.py).

    Parameters
    ----------
    superseded_memory : the original memory being corrected (will be archived
                        by feedback_collector after this call)
    corrected_memory  : the new memory ingested from the correction signal
    kuzu_store        : open KuzuDBStore instance
    duckdb_store      : open DuckDBStore instance

    Returns
    -------
    ContradictionResult if an event was created, None otherwise.
    """
    sup_id = str(getattr(superseded_memory, "memory_id", ""))
    cor_id = str(getattr(corrected_memory, "memory_id", ""))

    if not sup_id or not cor_id:
        logger.warning(
            "graph.contradiction | flag_correction called with missing "
            "memory_id | superseded={} corrected={}", sup_id, cor_id,
        )
        return None

    sup_entity, sup_relation, sup_object = _spo(superseded_memory)
    _, _, cor_object = _spo(corrected_memory)

    # Correction always generates a ContradictionEvent regardless of
    # CONTRADICTION_SCORE_GATE — the agent explicitly flagged this as wrong.
    conf_sup = _confidence(superseded_memory)
    conf_cor = _confidence(corrected_memory)
    score = _contradiction_score(conf_sup, conf_cor)

    event = ContradictionResult(
        memory_id_a=cor_id,      # corrected = winner side
        memory_id_b=sup_id,      # superseded = loser side
        entity=sup_entity,
        relation=sup_relation,
        object_a=cor_object,
        object_b=sup_object,
        score=max(score, CONTRADICTION_SCORE_GATE),  # floor at gate for corrections
    )

    await _write_contradiction_event(kuzu_store, event)
    await _update_contradicted_by(duckdb_store, sup_id, event.event_id)

    logger.info(
        "graph.contradiction | flag_correction | "
        "superseded={} corrected={} event_id={}",
        sup_id, cor_id, event.event_id,
    )
    return event


async def mark_resolved(
    event_id: str,
    kuzu_store: "KuzuDBStore",
) -> None:
    """
    Set resolved=True on a ContradictionEvent node in KuzuDB.

    Called by consolidator.py or feedback_collector.py after archival of the
    weaker memory confirms the contradiction has been handled.
    """
    conn = getattr(kuzu_store, "_conn", None)
    if conn is None:
        logger.warning(
            "graph.contradiction | KuzuDB unavailable — "
            "mark_resolved skipped | event_id={}", event_id,
        )
        return
    cypher = """
        MATCH (e:ContradictionEvent {event_id: $event_id})
        SET e.resolved = true
    """
    try:
        conn.execute(cypher, parameters={"event_id": event_id})
        logger.info(
            "graph.contradiction | ContradictionEvent resolved | event_id={}",
            event_id,
        )
    except Exception as exc:
        logger.warning(
            "graph.contradiction | mark_resolved failed | "
            "event_id={} | error={}", event_id, exc,
        )

