from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    import kuzu
except Exception:  # pragma: no cover - optional dependency
    kuzu = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = Path("data/graph/memory_graph")

CREATE_NODE_TABLE_MEMORY = """
CREATE NODE TABLE MemoryNode (
    memory_id STRING PRIMARY KEY,
    agent_id STRING,
    memory_type STRING,
    content STRING,
    sha256 STRING,
    lifecycle_state STRING,
    created_at STRING,
    last_accessed_at STRING,
    importance_score DOUBLE,
    session_id STRING,
    turn_index INT64,
    speaker_role STRING,
    ttl_seconds INT64,
    promoted_to STRING,
    tool_call_id STRING,
    graph_node_id STRING,
    metadata_json STRING
);
"""

CREATE_REL_TABLE_MEMORY_LINK = """
CREATE REL TABLE MEMORY_LINK (
    FROM MemoryNode TO MemoryNode,
    relation STRING,
    created_at STRING
);
"""


@dataclass(slots=True)
class GraphNode:
    memory_id: str
    agent_id: str
    memory_type: str
    content: str
    sha256: str
    lifecycle_state: str
    created_at: str
    last_accessed_at: str
    importance_score: float
    session_id: str | None = None
    turn_index: int | None = None
    speaker_role: str | None = None
    ttl_seconds: int | None = None
    promoted_to: str | None = None
    tool_call_id: str | None = None
    graph_node_id: str | None = None
    metadata_json: str | None = None


class KuzuDBStore:
    """Graph store for memory nodes and explicit relations.

    Uses Kuzu when available, but keeps an in-memory mirror so the rest of the
    codebase remains functional and fast even when Kuzu is unavailable.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db = None
        self.conn = None
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, list[dict[str, Any]]] = {}
        self._initialised = False
        self._connect()

    def _connect(self) -> None:
        if kuzu is None:
            logger.warning("Kuzu is not installed. Graph persistence will use the in-memory mirror only.")
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)

        try:
            self.conn.execute(CREATE_NODE_TABLE_MEMORY)
            self.conn.execute(CREATE_REL_TABLE_MEMORY_LINK)
        except RuntimeError:
            pass
        finally:
            self._initialised = True

    def initialise(self) -> None:
        if not self._initialised:
            self._connect()
            self._initialised = True

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _coerce_str(value: Any | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, UUID):
            return str(value)
        return str(value)

    def _memory_to_node(self, memory: Any) -> GraphNode:
        return GraphNode(
            memory_id=str(memory.memory_id),
            agent_id=str(memory.agent_id),
            memory_type=str(memory.memory_type),
            content=str(memory.content),
            sha256=str(memory.sha256),
            lifecycle_state=str(getattr(memory, "lifecycle_state", "ACTIVE")),
            created_at=getattr(memory, "created_at", self._now()).isoformat()
            if hasattr(getattr(memory, "created_at", None), "isoformat")
            else self._now(),
            last_accessed_at=getattr(memory, "last_accessed_at", self._now()).isoformat()
            if hasattr(getattr(memory, "last_accessed_at", None), "isoformat")
            else self._now(),
            importance_score=float(getattr(memory, "importance_score", 0.0)),
            session_id=self._coerce_str(getattr(memory, "session_id", None)),
            turn_index=getattr(memory, "turn_index", None),
            speaker_role=self._coerce_str(getattr(memory, "speaker_role", None)),
            ttl_seconds=getattr(memory, "ttl_seconds", None),
            promoted_to=self._coerce_str(getattr(memory, "promoted_to", None)),
            tool_call_id=self._coerce_str(getattr(memory, "tool_call_id", None)),
            graph_node_id=self._coerce_str(getattr(memory, "graph_node_id", None)) or str(memory.memory_id),
            metadata_json=None,
        )

    def save_memory(self, memory: Any) -> str:
        """Store a canonical memory node and return its graph node id."""

        node = self._memory_to_node(memory)
        self._nodes[node.memory_id] = node

        if self.conn is not None:
            try:
                self.conn.execute(
                    "CREATE (m:MemoryNode {memory_id: $memory_id, agent_id: $agent_id, memory_type: $memory_type, content: $content, sha256: $sha256, lifecycle_state: $lifecycle_state, created_at: $created_at, last_accessed_at: $last_accessed_at, importance_score: $importance_score, session_id: $session_id, turn_index: $turn_index, speaker_role: $speaker_role, ttl_seconds: $ttl_seconds, promoted_to: $promoted_to, tool_call_id: $tool_call_id, graph_node_id: $graph_node_id, metadata_json: $metadata_json})",
                    parameters=asdict(node),
                )
            except Exception:
                # Best effort only. The in-memory mirror remains authoritative for app behavior.
                pass

        return node.graph_node_id or node.memory_id

    async def save_memory_async(self, memory: Any) -> str:
        return await asyncio.to_thread(self.save_memory, memory)

    def get_node(self, node_id: str) -> dict[str, Any]:
        node = self._nodes.get(node_id)
        if node is not None:
            return asdict(node)

        if self.conn is None:
            return {}

        try:
            result = self.conn.execute(
                "MATCH (m:MemoryNode {memory_id: $memory_id}) RETURN m",
                parameters={"memory_id": node_id},
            )
            if result.has_next():
                row = result.get_next()[0]
                if isinstance(row, dict):
                    return dict(row)
        except Exception:
            return {}

        return {}

    async def get_node_async(self, node_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.get_node, node_id)

    def save_relation(
        self,
        source: str,
        target: str,
        relation: str,
    ) -> None:
        edge = {
            "source": source,
            "target": target,
            "relation": relation,
            "created_at": self._now(),
        }
        self._edges.setdefault(source, []).append(edge)

        if self.conn is None:
            return

        try:
            self.conn.execute(
                "MATCH (s:MemoryNode {memory_id: $source}), (t:MemoryNode {memory_id: $target}) CREATE (s)-[:MEMORY_LINK {relation: $relation, created_at: $created_at}]->(t)",
                parameters=edge,
            )
        except Exception:
            pass

    async def save_relation_async(self, source: str, target: str, relation: str) -> None:
        await asyncio.to_thread(self.save_relation, source, target, relation)

    def get_connected_nodes(self, node_id: str, limit: int = 25) -> list[dict[str, Any]]:
        """Return outgoing neighbors for a stored memory node."""

        edges = self._edges.get(node_id, [])[:limit]
        return [edge.copy() for edge in edges]

    async def get_connected_nodes_async(self, node_id: str, limit: int = 25) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.get_connected_nodes, node_id, limit)

    def bfs_traversal(self, seeds: list[str], max_hops: int = 2, limit: int = 50) -> list[dict[str, Any]]:
        if not seeds:
            return []

        visited: set[str] = set(seeds)
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
        paths: list[dict[str, Any]] = []

        while queue and len(paths) < limit:
            node_id, depth = queue.popleft()
            if depth >= max_hops:
                continue

            for edge in self._edges.get(node_id, []):
                target = edge["target"]
                path = edge.copy()
                path["hop"] = depth + 1
                path["source"] = node_id
                paths.append(path)
                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))
                if len(paths) >= limit:
                    break

        return paths

    async def bfs_traversal_async(self, seeds: list[str], max_hops: int = 2, limit: int = 50) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.bfs_traversal, seeds, max_hops, limit)
