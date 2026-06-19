from __future__ import annotations

import json
import logging
import kuzu

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, override

logger = logging.getLogger(__name__)

DB_PATH = Path("data/graph/memory_graph.kuzu")

_DDL_STATEMENTS: list[str] = [
    """
    CREATE NODE TABLE IF NOT EXISTS Entity (
        node_id     STRING PRIMARY KEY,
        entity_type STRING,
        label       STRING,
        importance  DOUBLE DEFAULT 0.5,
        created_at  STRING
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS ContradictionEvent (
        event_id    STRING PRIMARY KEY,
        memory_id_a STRING,
        memory_id_b STRING,
        score       DOUBLE,
        resolved    BOOLEAN DEFAULT false,
        created_at  STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS RELATES (
        FROM Entity TO Entity,
        relation    STRING,
        confidence  DOUBLE,
        memory_id   STRING,
        created_at  STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS CAUSES (
        FROM Entity TO Entity,
        causal_strength        DOUBLE,
        counterfactual_weight  DOUBLE,
        memory_id              STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS COOCCURS (
        FROM Entity TO Entity,
        cooccurrence_count  INT64 DEFAULT 1,
        last_seen           STRING
    )
    """,
]


class KuzuDBStore:
    """
    Thin async-compatible wrapper around a KuzuDB embedded database.

    Separation of concerns
    ----------------------
    KuzuDBStore owns graph I/O only:
      - DDL bootstrap
      - MERGE-based node upserts
      - Edge inserts
      - Raw BFS via Cypher (returned as flat dicts)

    Graph algorithms (path scoring, cycle detection, hop decay,
    result deduplication) live in graph/traversal.py, NOT here.
    That separation is intentional and must be preserved as M2 grows.

    Initialisation
    --------------
    Call await store.initialise() before any other method.
    _initialised is set True ONLY after all DDL statements succeed.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self._db_path = str(db_path)
        self._conn: Any | None = None
        self._initialised: bool = False

        # Dev-mode fallback mirrors — process-local, unbounded, NOT KuzuDB state.
        # Used only when self._conn is None. See module docstring.
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Connect to KuzuDB and run DDL bootstrap. Idempotent."""
        if self._initialised:
            return
        self._connect()

    def _connect(self) -> None:
        """
        Open KuzuDB connection and create schema tables.

        _initialised is set True only after ALL DDL statements succeed.
        If KuzuDB is unavailable, _conn remains None and the store falls
        back to the dev-mode in-memory mirrors.
        """
        if kuzu is None:
            logger.warning(
                "graph.ontology | kuzu package not installed; "
                "falling back to dev-mode in-memory mirrors"
            )
            self._initialised = True
            return

        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            db = kuzu.Database(str(self._db_path))
            self._conn = kuzu.Connection(db)
        except Exception as exc:
            logger.warning(
                "graph.ontology | KuzuDB connection failed — "
                "falling back to dev-mode in-memory mirrors | error=%s", exc
            )
            # _initialised stays False; caller will get degraded behaviour.
            return

        # Run DDL. IF NOT EXISTS guards make this idempotent.
        ddl_success = True
        for stmt in _DDL_STATEMENTS:
            try:
                self._conn.execute(stmt)
            except Exception as exc:
                logger.warning(
                    "graph.ontology | DDL failed — partial schema | "
                    "stmt_preview=%r | error=%s",
                    stmt.strip()[:80],
                    exc,
                )
                ddl_success = False
                break

        if ddl_success:
            self._initialised = True
            logger.info("graph.ontology | KuzuDB initialised at {}", self._db_path)
        else:
            # Connection open but schema incomplete — log and keep _conn so
            # reads that don't need the missing table can still work.
            logger.warning(
                "graph.ontology | KuzuDB schema bootstrap incomplete; "
                "some operations may fail"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        """Current UTC timestamp as ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _coerce_datetime(value: Any) -> str:
        """
        Coerce a datetime (or None) to an ISO-8601 UTC string.

        Avoids the fragile double-evaluation pattern of:
          getattr(obj, 'x', fallback).isoformat() if hasattr(...) else fallback
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        if isinstance(value, str) and value:
            return value
        return datetime.now(timezone.utc).isoformat()

    def _memory_to_node(self, memory: Any) -> dict[str, Any]:
        """
        Extract a MemoryNode property dict from a BaseMemory-like object.

        metadata is serialised to JSON when present so the field is never
        silently None (fixes previous issue where metadata_json was always None).
        """
        created_at = self._coerce_datetime(getattr(memory, "created_at", None))

        raw_metadata = getattr(memory, "metadata", None)
        if raw_metadata is not None:
            try:
                metadata_json: str | None = json.dumps(raw_metadata, default=str)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "graph.ontology | metadata serialisation failed for "
                    "memory_id={} | error={}", getattr(memory, "memory_id", "?"), exc
                )
                metadata_json = None
        else:
            metadata_json = None

        return {
            "node_id": str(getattr(memory, "memory_id", "")),
            "entity_type": str(getattr(memory, "memory_type", "CONCEPT")),
            "label": str(getattr(memory, "content", ""))[:256],
            "importance": float(getattr(memory, "importance_score", 0.5)),
            "created_at": created_at,
            "metadata_json": metadata_json,
        }

    async def save_memory(self, memory: Any) -> None:
        """
        Upsert a memory as an Entity node in KuzuDB.

        Uses MERGE on memory_id (node_id) as the primary key so that
        duplicate ingestion (replay rebuild, dedup miss) is idempotent.
        CREATE would produce duplicate nodes — MERGE is mandatory here.

        Falls back to the dev-mode _nodes mirror when KuzuDB is unavailable.
        All KuzuDB exceptions are logged at WARNING; the method does not raise
        so that a graph write failure never blocks the DuckDB write path.
        """
        props = self._memory_to_node(memory)
        node_id = props["node_id"]

        if self._conn is None:
            # Dev-mode fallback — in-memory only, not persisted.
            self._nodes[node_id] = props
            return

        cypher = """
            MERGE (n:Entity {node_id: $node_id})
            SET n.entity_type  = $entity_type,
                n.label        = $label,
                n.importance   = $importance,
                n.created_at   = $created_at
        """
        
        # KuzuDB throws an error if unused parameters are passed
        query_params = {k: v for k, v in props.items() if k != "metadata_json"}

        try:
            self._conn.execute(cypher, parameters=query_params)
        except Exception as exc:
            logger.warning(
                "graph.ontology | KuzuDB node upsert failed — "
                "graph diverges from DuckDB | memory_id=%s | error=%s",
                node_id,
                exc,
            )

    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory (Entity node) and all associated edges from KuzuDB.
        """
        if self._conn is None:
            self._nodes.pop(memory_id, None)
            return

        cypher = "MATCH (n:Entity {node_id: $node_id}) DETACH DELETE n"
        try:
            self._conn.execute(cypher, parameters={"node_id": memory_id})
        except Exception as exc:
            logger.warning(
                "graph.ontology | KuzuDB node deletion failed | memory_id=%s | error=%s",
                memory_id,
                exc,
            )

    async def save_edge(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        confidence: float = 1.0,
        memory_id: str = "",
        rel_table: str = "RELATES",
    ) -> None:
        """
        Insert a directed edge between two Entity nodes.

        rel_table must be one of: RELATES | CAUSES | COOCCURS.
        Falls back to _edges mirror when KuzuDB is unavailable.
        """
        edge = {
            "from_id": from_id,
            "to_id": to_id,
            "relation": relation,
            "confidence": confidence,
            "memory_id": memory_id,
            "created_at": self._now(),
        }

        if self._conn is None:
            self._edges.setdefault(from_id, []).append(edge)
            return

        cypher = f"""
            MATCH (a:Entity {{node_id: $from_id}}),
                  (b:Entity {{node_id: $to_id}})
            CREATE (a)-[:{rel_table} {{
                relation:   $relation,
                confidence: $confidence,
                memory_id:  $memory_id,
                created_at: $created_at
            }}]->(b)
        """
        try:
            self._conn.execute(cypher, parameters=edge)
        except Exception as exc:
            logger.warning(
                "graph.ontology | KuzuDB edge insert failed | "
                "from=%s to=%s rel=%s | error=%s",
                from_id, to_id, rel_table, exc,
            )

    async def increment_cooccurrence(self, from_id: str, to_id: str) -> None:
        """
        Increment COOCCURS edge counter between two entities.
        Creates the edge if it does not exist (MERGE semantics).
        """
        if self._conn is None:
            return  # No meaningful fallback for counters in dev-mode.

        now = self._now()
        cypher = """
            MATCH (a:Entity {node_id: $from_id}),
                  (b:Entity {node_id: $to_id})
            MERGE (a)-[r:COOCCURS]->(b)
            ON CREATE SET r.cooccurrence_count = 1,  r.last_seen = $now
            ON MATCH  SET r.cooccurrence_count = r.cooccurrence_count + 1,
                          r.last_seen = $now
        """
        try:
            self._conn.execute(
                cypher, parameters={"from_id": from_id, "to_id": to_id, "now": now}
            )
        except Exception as exc:
            logger.warning(
                "graph.ontology | COOCCURS increment failed | "
                "from={} to={} | error={}", from_id, to_id, exc,
            )

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """
        Return a node property dict or None.

        Lookup order:
          1. KuzuDB (authoritative, multi-process safe).
          2. Dev-mode _nodes mirror (only when _conn is None).

        Note: _nodes is process-local. During a live session, nodes written
        by a different process are not in _nodes but ARE in KuzuDB, so the
        KuzuDB-first order here is correct for production. In dev-mode
        (conn=None) only the local mirror is available.
        """
        if self._conn is not None:
            cypher = """
                MATCH (n:Entity {node_id: $node_id})
                RETURN n.node_id     AS node_id,
                       n.entity_type AS entity_type,
                       n.label       AS label,
                       n.importance  AS importance,
                       n.created_at  AS created_at
            """
            try:
                result = self._conn.execute(cypher, parameters={"node_id": node_id})
                rows = result.get_as_df() if hasattr(result, "get_as_df") else []
                if len(rows):
                    return dict(rows.iloc[0])
            except Exception as exc:
                logger.warning(
                    "graph.ontology | get_node KuzuDB query failed | "
                    "node_id={} | error={}", node_id, exc,
                )

        # Dev-mode fallback.
        return self._nodes.get(node_id)

    # ------------------------------------------------------------------
    # BFS — raw graph I/O only; algorithms live in graph/traversal.py
    # ------------------------------------------------------------------

    @override
    async def bfs_traversal_async(
        self,
        seeds: list[str],
        max_hops: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Execute a BFS traversal from seed node IDs up to max_hops deep.

        Returns a flat list of dicts:
          {start_id, end_id, hop, relation}

        This method returns raw graph I/O results only — no scoring, no
        cycle detection, no path deduplication. All graph algorithms that
        operate on these results belong in graph/traversal.py.

        Falls back to the dev-mode _edges mirror when KuzuDB is unavailable.
        The in-memory BFS is a best-effort degraded path; it does not support
        hop-level relation metadata.
        """
        if self._conn is not None:
            return await self._bfs_kuzu(seeds, max_hops, limit)
        return self._bfs_in_memory(seeds, max_hops, limit)

    async def _bfs_kuzu(
        self,
        seeds: list[str],
        max_hops: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """BFS via KuzuDB Cypher"""
        cypher = f"""
            MATCH (start:Entity)-[r:RELATES*1..{max_hops}]->(target:Entity)
            WHERE start.node_id IN $seeds
            RETURN start.node_id  AS start_id,
                   target.node_id AS end_id,
                   length(r)      AS hop
            LIMIT {limit}
        """
        try:
            result = self._conn.execute(cypher, parameters={"seeds": seeds})
            cols = result.get_column_names()
            records = []
            while result.has_next():
                records.append(dict(zip(cols, result.get_next())))
            return records
        except Exception as exc:
            logger.warning(
                "graph.ontology | BFS KuzuDB query failed | "
                "seeds=%s max_hops=%s | error=%s", seeds, max_hops, exc,
            )
            return []

    def _bfs_in_memory(
        self,
        seeds: list[str],
        max_hops: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        BFS over the dev-mode _edges mirror.

        Dev-mode fallback only. Does not produce relation metadata.
        Results are not guaranteed to match KuzuDB output.
        """
        visited: set[str] = set(seeds)
        frontier: list[str] = list(seeds)
        results: list[dict[str, Any]] = []

        for hop in range(1, max_hops + 1):
            next_frontier: list[str] = []
            for node_id in frontier:
                for edge in self._edges.get(node_id, []):
                    neighbour = edge.get("to_id", "")
                    if not neighbour or neighbour in visited:
                        continue
                    visited.add(neighbour)
                    next_frontier.append(neighbour)
                    results.append(
                        {"start_id": node_id, "end_id": neighbour, "hop": hop}
                    )
                    if len(results) >= limit:
                        return results
            frontier = next_frontier
            if not frontier:
                break

        return results

