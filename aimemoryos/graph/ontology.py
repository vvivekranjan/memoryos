from __future__ import annotations

import logging
import json
import redis.asyncio as redis

from datetime import datetime, timezone
from typing import Any
from falkordb import AsyncFalkorDB, FalkorDB

logger = logging.getLogger(__name__)

class FalkorDBStore:
    """
    High-performance GraphBLAS-based database built on Redis (FalkorDB).
    Client-server architecture, enabling true async connection pooling and
    ultra-fast variable-length traversals.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 6379, graph_name: str = "memoryos"):
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self.client = None
        self.graph = None
        self._initialised = False

    def initialise(self) -> None:
        """Connect to FalkorDB and run Indexing bootstrap synchronously."""
        if self._initialised:
            return
        
        if AsyncFalkorDB is None or FalkorDB is None:
            logger.warning("falkordb package not installed; Graph operations will fail.")
            return

        try:
            # Sync initialization for indices to avoid event-loop collisions
            sync_client = FalkorDB(host=self.host, port=self.port)
            sync_graph = sync_client.select_graph(self.graph_name)
            sync_graph.query("CREATE INDEX ON :Entity(node_id)")
            sync_graph.query("CREATE INDEX ON :Entity(entity_type)")
            
            # Setup async client for runtime queries
            self.client = AsyncFalkorDB(host=self.host, port=self.port)
            self.graph = self.client.select_graph(self.graph_name)
            
            self._initialised = True
            logger.info("graph.falkordb_store | FalkorDB initialised at %s:%s", self.host, self.port)
        except Exception as exc:
            logger.error("graph.falkordb_store | FalkorDB connection failed | error=%s", exc)

    @staticmethod
    def _coerce_datetime(value: Any) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        if isinstance(value, str) and value:
            return value
        return datetime.now(timezone.utc).isoformat()

    def _memory_to_node(self, memory: Any) -> dict[str, Any]:
        created_at = self._coerce_datetime(getattr(memory, "created_at", None))
        
        raw_metadata = getattr(memory, "metadata", None)
        metadata_json = None
        if raw_metadata is not None:
            try:
                metadata_json = json.dumps(raw_metadata, default=str)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "graph.falkordb_store | metadata serialisation failed for "
                    "memory_id=%s | error=%s", getattr(memory, "memory_id", "?"), exc
                )

        return {
            "node_id": str(getattr(memory, "memory_id", "")),
            "entity_type": str(getattr(memory, "memory_type", "CONCEPT")),
            "label": str(getattr(memory, "content", ""))[:256],
            "importance": float(getattr(memory, "importance_score", 0.5)),
            "created_at": created_at,
            "metadata_json": metadata_json or "",
        }

    async def save_memory(self, memory: Any) -> None:
        if not self._initialised:
            return

        props = self._memory_to_node(memory)
        cypher = """
            MERGE (n:Entity {node_id: $node_id})
            SET n.entity_type  = $entity_type,
                n.label        = $label,
                n.importance   = $importance,
                n.created_at   = $created_at
        """
        try:
            await self.graph.query(cypher, props)
        except Exception as exc:
            logger.warning("graph.falkordb_store | Node upsert failed | node_id=%s | error=%s", props["node_id"], exc)

    async def delete_memory(self, memory_id: str) -> None:
        if not self._initialised:
            return

        cypher = "MATCH (n:Entity {node_id: $node_id}) DETACH DELETE n"
        try:
            await self.graph.query(cypher, {"node_id": memory_id})
        except Exception as exc:
            logger.warning("graph.falkordb_store | Node deletion failed | memory_id=%s | error=%s", memory_id, exc)

    async def save_edge(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        confidence: float = 1.0,
        memory_id: str = "",
        rel_table: str = "RELATES",
    ) -> None:
        if rel_table not in {"RELATES", "CAUSES", "COOCCURS"}:
            raise ValueError(f"Invalid rel_table: {rel_table}")

        if not self._initialised:
            return

        edge = {
            "from_id": from_id,
            "to_id": to_id,
            "relation": relation,
            "confidence": confidence,
            "memory_id": memory_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        cypher = f"""
            MATCH (a:Entity {{node_id: $from_id}}),
                  (b:Entity {{node_id: $to_id}})
            MERGE (a)-[r:{rel_table}]->(b)
            SET r.relation = $relation,
                r.confidence = $confidence,
                r.memory_id = $memory_id,
                r.created_at = $created_at
        """
        try:
            await self.graph.query(cypher, edge)
        except Exception as exc:
            logger.warning("graph.falkordb_store | Edge insert failed | from=%s to=%s rel=%s | error=%s", from_id, to_id, rel_table, exc)

    async def save_edges_batch(self, edges: list[dict]) -> None:
        """
        Batch insertion of relationship edges into FalkorDB.
        edges = [{"src": "uuid1", "dst": "uuid2", "type": "RELATES", "weight": 0.8}, ...]
        """
        if not self._initialised or not edges:
            return
        
        # Group by relation type for safe Cypher execution in FalkorDB
        by_type: dict[str, list[dict]] = {}
        for edge in edges:
            rel_type = edge.get("type", "RELATES").upper()
            if rel_type not in {"RELATES", "CAUSES", "COOCCURS"}:
                rel_type = "RELATES"
            by_type.setdefault(rel_type, []).append(edge)

        for rel_type, edge_list in by_type.items():
            query = f"""
            UNWIND $edges AS edge
            MATCH (a:Entity {{node_id: edge.src}}), (b:Entity {{node_id: edge.dst}})
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r.confidence = coalesce(edge.weight, 1.0),
                          r.relation = edge.relation,
                          r.memory_id = edge.memory_id,
                          r.created_at = edge.created_at
            """
            try:
                await self.graph.query(query, {"edges": edge_list})
            except Exception as exc:
                logger.warning("graph.falkordb_store | Batch edge insert failed for %s | error=%s", rel_type, exc)

    async def increment_cooccurrence(self, from_id: str, to_id: str) -> None:
        if not self._initialised:
            return

        now = datetime.now(timezone.utc).isoformat()
        cypher = """
            MATCH (a:Entity {node_id: $from_id}),
                  (b:Entity {node_id: $to_id})
            MERGE (a)-[r:COOCCURS]->(b)
            ON CREATE SET r.cooccurrence_count = 1,  r.last_seen = $now
            ON MATCH  SET r.cooccurrence_count = r.cooccurrence_count + 1,
                          r.last_seen = $now
        """
        try:
            await self.graph.query(cypher, {"from_id": from_id, "to_id": to_id, "now": now})
        except Exception as exc:
            logger.warning("graph.falkordb_store | COOCCURS increment failed | from=%s to=%s | error=%s", from_id, to_id, exc)

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        if not self._initialised:
            return None

        cypher = """
            MATCH (n:Entity {node_id: $node_id})
            RETURN n.node_id     AS node_id,
                   n.entity_type AS entity_type,
                   n.label       AS label,
                   n.importance  AS importance,
                   n.created_at  AS created_at
        """
        try:
            result = await self.graph.query(cypher, {"node_id": node_id})
            if result.result_set:
                row = result.result_set[0]
                return {
                    "node_id": row[0],
                    "entity_type": row[1],
                    "label": row[2],
                    "importance": row[3],
                    "created_at": row[4]
                }
        except Exception as exc:
            logger.warning("graph.falkordb_store | get_node query failed | node_id=%s | error=%s", node_id, exc)
        return None

    async def bfs_traversal_async(
        self,
        seeds: list[str],
        max_hops: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Execute a BFS traversal from seed node IDs up to max_hops deep.
        Returns a flat list of dicts: {start_id, end_id, hop}
        """
        if not self._initialised or not seeds:
            return []

        # FalkorDB handles variable-length paths natively and safely
        # Note: FalkorDB openCypher supports IN with list directly
        cypher = f"""
        MATCH p = (start:Entity)-[r:RELATES*1..{int(max_hops)}]-(connected:Entity)
        WHERE start.node_id IN $seeds
        WITH start, connected, min(length(p)) as hop
        RETURN start.node_id AS start_id, connected.node_id AS end_id, hop
        ORDER BY hop ASC
        LIMIT {int(limit)}
        """
        try:
            result = await self.graph.query(cypher, {
                "seeds": seeds
            })
            records = []
            for row in result.result_set:
                records.append({
                    "start_id": row[0],
                    "end_id": row[1],
                    "hop": row[2]
                })
            return records
        except Exception as exc:
            logger.warning(
                "graph.falkordb_store | BFS FalkorDB query failed | "
                "seeds=%s max_hops=%s | error=%s", seeds, max_hops, exc
            )
            return []
