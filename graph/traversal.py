from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import logging

logger = logging.getLogger(__name__)

#: Activation boost decay per hop. 1 hop → 0.5, 2 hops → 0.25, …
DECAY_PER_HOP: float = 0.5

#: Hard cap on activation_boost to prevent FM-T2 activation explosion.
ACTIVATION_BOOST_CAP: float = 1.0

#: Default BFS result limit — matches KuzuDBStore.bfs_traversal_async default.
DEFAULT_LIMIT: int = 50


@runtime_checkable
class GraphStore(Protocol):
    """
    Minimal protocol that any graph backend must satisfy.

    bfs_traversal_async MUST accept seeds, max_hops, AND limit.
    graph/traversal.py previously dropped limit, breaking protocol
    conformance with RetrievalEngine._expand_graph. Fixed here.
    """

    async def bfs_traversal_async(
        self,
        seeds: list[str],
        max_hops: int,
        limit: int,
    ) -> list[dict[str, Any]]: ...


class TraversalResult:
    """
    A single BFS result record with computed traversal metadata.

    Attributes
    ----------
    start_id        : seed node the path originated from
    end_id          : reached node (unique per result set after dedup)
    hop             : number of hops from seed (1-indexed)
    activation_boost: hop-decayed activation contribution (TRD §5.4)
    traversal_score : confidence × activation_boost; used by fusion layer
    path            : ordered list of node_ids traversed (when available)
    raw             : original dict from KuzuDBStore for passthrough fields
    """

    __slots__ = (
        "start_id",
        "end_id",
        "hop",
        "activation_boost",
        "traversal_score",
        "path",
        "raw",
    )

    def __init__(
        self,
        start_id: str,
        end_id: str,
        hop: int,
        confidence: float = 1.0,
        path: list[str] | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.start_id = start_id
        self.end_id = end_id
        self.hop = hop
        # boost = DECAY_PER_HOP ** hop, capped at ACTIVATION_BOOST_CAP.
        raw_boost = DECAY_PER_HOP ** hop
        self.activation_boost = min(raw_boost, ACTIVATION_BOOST_CAP)
        self.traversal_score = confidence * self.activation_boost
        self.path = path or [start_id, end_id]
        self.raw = raw or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_id": self.start_id,
            "end_id": self.end_id,
            "hop": self.hop,
            "activation_boost": self.activation_boost,
            "traversal_score": self.traversal_score,
            "path": self.path,
        }


# ---------------------------------------------------------------------------
# Core traversal function
# ---------------------------------------------------------------------------


async def bfs_traversal(
    store: GraphStore,
    seeds: list[str],
    max_hops: int = 2,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """
    BFS expansion from seed node IDs up to max_hops deep.

    This is the public entry point for RetrievalEngine._expand_graph.
    It owns all graph algorithm concerns that KuzuDBStore must not:

      1. Limit propagation — passes limit through to store.bfs_traversal_async.
         Previously this was dropped, causing RetrievalEngine to always get
         the default 50 results regardless of top_k. Fixed.

      2. Cycle detection — FM-T1 guard. Seeds are pre-added to visited so
         BFS never returns a seed as a result. Each end_id is tracked so
         the same node is never added to results twice regardless of how
         many paths reach it (prevents oscillation, FM-T3 adjacency).

      3. Result deduplication — when multiple seeds reach the same end_id
         via different paths, only the shortest-hop (highest-score) result
         is kept. The store may return duplicates for multi-seed queries.

      4. Hop decay — TRD §5.4. Each result carries activation_boost =
         DECAY_PER_HOP ** hop and traversal_score = confidence × boost.

      5. Activation explosion cap — FM-T2. activation_boost is hard-capped
         at ACTIVATION_BOOST_CAP (1.0). The cap is applied in TraversalResult
         before scores leave this layer.

    Parameters
    ----------
    store     : any backend satisfying the GraphStore protocol
    seeds     : memory_id / node_id strings to start BFS from
    max_hops  : maximum hop depth (default 2)
    limit     : maximum total results (caller: max(10, top_k * 4))

    Returns
    -------
    List of dicts (TraversalResult.to_dict()) sorted by traversal_score
    descending. Empty list on error or no results.
    """
    if not seeds:
        return []

    if not isinstance(store, GraphStore):
        logger.warning(
            "graph.traversal | store does not satisfy GraphStore protocol | "
            "type={}", type(store).__name__,
        )

    # Fetch raw BFS results from the graph store.
    # Pass limit through — this was previously dropped causing FM-T2 risk.
    try:
        raw_results: list[dict[str, Any]] = await store.bfs_traversal_async(
            seeds=seeds,
            max_hops=max_hops,
            limit=limit,
        )
    except Exception as exc:
        logger.warning(
            "graph.traversal | bfs_traversal_async failed | "
            "seeds={} max_hops={} | error={}", seeds, max_hops, exc,
        )
        return []

    if not raw_results:
        return []

    # ------------------------------------------------------------------
    # Cycle detection + deduplication
    # ------------------------------------------------------------------
    # visited tracks node_ids that must not appear as results.
    # Seeds are pre-added — a seed must not be returned as a neighbour
    # even if it is reachable from another seed (FM-T1 guard).
    visited: set[str] = set(seeds)

    # best_by_end keeps only the shortest-hop result per end_id.
    # Shorter hop → higher activation_boost → correct to prefer it.
    best_by_end: dict[str, TraversalResult] = {}

    for row in raw_results:
        end_id: str = str(row.get("end_id", ""))
        start_id: str = str(row.get("start_id", ""))
        hop: int = int(row.get("hop", 1))
        confidence: float = float(row.get("confidence", 1.0))

        if not end_id or end_id in visited:
            # Skip seeds, already-visited nodes, and empty IDs.
            continue

        candidate = TraversalResult(
            start_id=start_id,
            end_id=end_id,
            hop=hop,
            confidence=confidence,
            raw=row,
        )

        existing = best_by_end.get(end_id)
        if existing is None or candidate.hop < existing.hop:
            best_by_end[end_id] = candidate

        visited.add(end_id)

    # ------------------------------------------------------------------
    # Sort by traversal_score descending and serialise.
    # ------------------------------------------------------------------
    ranked = sorted(
        best_by_end.values(),
        key=lambda r: r.traversal_score,
        reverse=True,
    )

    return [r.to_dict() for r in ranked[:limit]]

