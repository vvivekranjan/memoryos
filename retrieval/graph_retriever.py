from __future__ import annotations

from typing import Any, Optional
from uuid import UUID
from graph.traversal import bfs_traversal, DEFAULT_LIMIT
from memory.models import MemoryTypeEnum
from retrieval.engine import RetrievalTrace, RetrievalCandidate

from core.interfaces import DocStore

import logging

logger = logging.getLogger(__name__)

#: Default BFS hop depth.
DEFAULT_MAX_HOPS: int = 2

#: Limit multiplier applied to k to get graph result budget.
#: Matches RetrievalEngine._expand_graph: limit = max(10, top_k * 4).
LIMIT_MULTIPLIER: int = 4
LIMIT_FLOOR: int = 10

#: Default importance and provenance values for memories where these fields
#: are absent (M1 memories predate provenance enforcement).
_DEFAULT_IMPORTANCE: float = 0.5
_DEFAULT_PROVENANCE: str = "OBSERVED"
_DEFAULT_PROVENANCE_CONFIDENCE: float = 1.0


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------


class GraphRetriever:
    """
    KuzuDB BFS-based retriever.

    Traverses the entity graph from seed nodes derived from the query context,
    resolves neighbouring memory_ids from DuckDB, and returns scored
    RetrievalCandidate objects each with a full RetrievalTrace.

    Separation of concerns
    ----------------------
    GraphRetriever: fetches candidates and assembles RetrievalTrace.
    graph/traversal.py: BFS algorithm, hop decay, cycle detection, dedup.
    graph/ontology.py: KuzuDB I/O.
    retrieval/fusion.py: RRF fusion across all retrievers.
    """

    def __init__(
        self,
        graph_store: Any,   # KuzuDBStore — typed as Any to avoid circular import
        doc_store: DocStore,
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> None:
        self.graph_store = graph_store
        self.doc_store = doc_store
        self.max_hops = max_hops

    # ------------------------------------------------------------------
    # Primary retrieval entrypoint
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query_embedding: list[float],
        query_text: str,
        k: int,
        agent_id: str,
        seed_memory_ids: Optional[list[str]] = None,
        session_scope: Optional[Any] = None,
    ) -> list[RetrievalCandidate]:
        """
        BFS retrieval from entity graph seeds.

        Parameters
        ----------
        query_embedding  : 768-dim query vector (unused by BFS itself; passed
                           for interface compatibility with other retrievers)
        query_text       : raw query string (used for entity label seed lookup
                           when no seed_memory_ids are provided)
        k                : number of candidates to return
        agent_id         : partition key; only this agent's memories are returned
        seed_memory_ids  : explicit graph seed node IDs (memory_ids or entity
                           node_ids). When provided, skips entity label lookup.
                           The VectorRetriever result set is the canonical source
                           for these — RetrievalEngine passes them after the
                           vector pass.
        session_scope    : unused; reserved for session-scoped activation

        Returns
        -------
        list[RetrievalCandidate] — at most k candidates, each with RetrievalTrace.
        REQ-RET-001: every candidate carries a full RetrievalTrace.

        Notes
        -----
        Seeds: BFS seeds are entity node_ids or memory_ids, NOT raw query text.
        query_text is used only as a fallback entity label seed when no
        seed_memory_ids are provided, letting the graph bootstrap from the
        entity the query is asking about. In practice, RetrievalEngine should
        always supply seed_memory_ids from the vector pass.
        """
        seeds = self._resolve_seeds(seed_memory_ids, query_text)
        if not seeds:
            logger.debug(
                "graph_retriever | no BFS seeds resolved — returning empty | "
                "agent_id={} query_text={!r}",
                agent_id, query_text[:80],
            )
            return []

        limit = max(LIMIT_FLOOR, k * LIMIT_MULTIPLIER)

        # BFS via graph/traversal.py — owns hop decay, cycle detection, dedup,
        # and limit propagation. Returns list[dict] with traversal_score.
        traversed = await bfs_traversal(
            store=self.graph_store,
            seeds=seeds,
            max_hops=self.max_hops,
            limit=limit,
        )

        if not traversed:
            return []

        # Collect best traversal record per end_id.
        # bfs_traversal already deduplicates; this maps end_id → record.
        best: dict[UUID, dict[str, Any]] = {}
        for edge in traversed:
            end_id = edge.get("end_id", "")
            if not end_id:
                continue
            existing = best.get(end_id)
            if existing is None or edge.get("traversal_score", 0.0) > existing.get("traversal_score", 0.0):
                best[end_id] = edge

        # Rank by traversal_score descending; take top k.
        ranked_ids = sorted(
            best.keys(),
            key=lambda eid: best[eid].get("traversal_score", 0.0),
            reverse=True,
        )[:k]

        if not ranked_ids:
            return []

        # Fetch full memory objects from DuckDB.
        try:
            memories = await self.doc_store.get_memories_by_ids(
                agent_id=agent_id,
                memory_ids=[UUID(mid) for mid in ranked_ids],
            )
        except Exception as exc:
            logger.warning(
                "graph_retriever | DuckDB bulk fetch failed | "
                "agent_id={} count={} | error={}",
                agent_id, len(ranked_ids), exc,
            )
            return []

        if not memories:
            return []

        memory_map: dict[str, Any] = {
            str(mem.memory_id): mem for mem in memories
        }

        # Assemble RetrievalCandidate with full RetrievalTrace.
        candidates: list[RetrievalCandidate] = []
        for rank, memory_id in enumerate(ranked_ids, start=1):
            memory = memory_map.get(memory_id)
            if memory is None:
                # Memory not found in DuckDB for this agent — skip silently.
                # Can occur if the graph references a memory from another agent
                # (graph is not partitioned by agent_id at the edge level).
                logger.debug(
                    "graph_retriever | memory_id={} not found for "
                    "agent_id={} — skipping", memory_id, agent_id,
                )
                continue

            edge = best[memory_id]
            traversal_score: float = float(edge.get("traversal_score", 0.0))
            activation_boost: float = float(edge.get("activation_boost", 0.0))
            hop: int = int(edge.get("hop", 1))
            start_id: Optional[str] = edge.get("start_id")

            # Build graph_path for RetrievalTrace.
            if start_id:
                graph_path = [start_id, memory_id]
            else:
                graph_path = [memory_id]

            # Safely extract memory fields; apply defaults for stub fields
            # absent on M1-era memories.
            importance_score: float = float(
                getattr(memory, "importance_score", _DEFAULT_IMPORTANCE)
            )
            provenance: str = _DEFAULT_PROVENANCE
            raw_prov = getattr(memory, "provenance", None)
            if raw_prov is not None:
                provenance = (
                    raw_prov.value
                    if hasattr(raw_prov, "value")
                    else str(raw_prov)
                )
            provenance_confidence: float = float(
                getattr(memory, "provenance_confidence", _DEFAULT_PROVENANCE_CONFIDENCE)
            )

            # Full RetrievalTrace on every candidate.
            trace = RetrievalTrace(
                memory_id=memory_id,
                final_score=traversal_score,
                retrieved_by=["graph"],
                graph_rank=rank,
                importance_score=importance_score,
                recency_boost=0.0,          # graph retriever has no temporal component
                activation_boost=activation_boost,
                graph_path=graph_path,
                provenance=provenance,
                provenance_confidence=provenance_confidence,
            )

            # Safe memory_type extraction — avoid double-cast fragility.
            raw_type = getattr(memory, "memory_type", MemoryTypeEnum.SEMANTIC)
            if isinstance(raw_type, MemoryTypeEnum):
                memory_type = raw_type
            else:
                try:
                    memory_type = MemoryTypeEnum(str(raw_type))
                except ValueError:
                    memory_type = MemoryTypeEnum.SEMANTIC

            candidates.append(
                RetrievalCandidate(
                    memory_id=memory_id,
                    content=memory.content,
                    memory_type=memory_type,
                    final_score=traversal_score,
                    trace=trace,
                )
            )

        return candidates

    # ------------------------------------------------------------------
    # Seed resolution
    # ------------------------------------------------------------------

    def _resolve_seeds(
        self,
        seed_memory_ids: Optional[list[str]],
        query_text: str,
    ) -> list[str]:
        """
        Resolve BFS seed node IDs.

        Priority:
          1. Explicit seed_memory_ids from caller (from vector retriever pass).
          2. Fallback: use query_text as an entity label seed. This allows the
             graph to bootstrap when no prior vector pass has been run (e.g.
             standalone graph retrieval tests). In production, RetrievalEngine
             should always supply seeds from the vector pass.

        Note: BFS seeds must be node_ids (entity labels or memory_ids), NOT
        raw query text. query_text is passed as-is as a fallback entity label
        because KuzuDB Entity.node_id = normalised entity label, and a short
        query ("Alice", "Python") is often a valid entity label. Longer queries
        will produce no matching seed node, returning an empty result — which
        is correct behaviour, not an error.
        """
        if seed_memory_ids:
            return [s for s in seed_memory_ids if s]

        # Fallback: normalise query_text as an entity label seed.
        # KuzuDB Entity.node_id is lowercased + whitespace-normalised. Match that normalisation here.
        normalised = query_text.strip().lower()
        if normalised:
            return [normalised]
        return []

