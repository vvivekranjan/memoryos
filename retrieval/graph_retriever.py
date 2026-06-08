from __future__ import annotations

import asyncio
import logging
from uuid import UUID
from typing import Any, List, Optional

from core.interfaces import (
    BaseRetriever,
    DocStore,
    GraphStore,
    SessionScope,
)
from memory.models import (
    MemoryTypeEnum,
    RetrievalCandidate,
    RetrievalTrace,
    RetrieverContribution,
)
from core.config import MemoryConfig
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self, graph_store: GraphStore, doc_store: DocStore):
        self.graph_store = graph_store
        self.doc_store = doc_store

    async def retrieve(
        self,
        query_embedding: List[float],
        query_text: str,
        k: int,
        agent_id: str,
        session_scope: Optional[SessionScope] = None,
    ) -> List[RetrievalCandidate]:
        traversed = await self.graph_store.bfs_traversal_async(
            seeds=[query_text],
            max_hops=2,
            limit=max(10, k * 4),
        )

        if not traversed:
            return []

        graph_scores: dict[str, tuple[float, dict[str, Any]]] = {}
        for edge in traversed:
            target = edge.get("target")
            if not target:
                continue

            source = edge.get("source")
            hop = int(edge.get("hop") or 1)
            graph_score = max(0.0, 1.0 / (1.0 + hop))

            existing = graph_scores.get(str(target))
            if existing is None or graph_score > existing[0]:
                graph_scores[str(target)] = (
                    graph_score,
                    {
                        "source": str(source) if source is not None else None,
                        "target": str(target),
                        "relation": edge.get("relation"),
                        "hop": hop,
                    },
                )

        memory_ids = [target for target in graph_scores.keys()][:k]
        if not memory_ids:
            return []

        memories = await self.doc_store.get_memories_by_ids(
            agent_id=agent_id,
            memory_ids=[UUID(memory_id) for memory_id in memory_ids],
        )

        memory_map = {str(memory.memory_id): memory for memory in memories}
        candidates: List[RetrievalCandidate] = []
        for rank, memory_id in enumerate(memory_ids, start=1):
            memory = memory_map.get(memory_id)
            if memory is None:
                continue

            score, graph_path = graph_scores[memory_id]
            trace = RetrievalTrace(
                memory_id=memory_id,
                retrieved_by=[
                    RetrieverContribution(
                        retriever="graph",
                        rank=rank,
                        raw_score=score,
                        rrf_contribution=0.0,
                    )
                ],
                final_score=score,
                importance_score=memory.importance_score,
                recency_boost=1.0,
                activation_boost=score,
                graph_path=[
                    graph_path["source"],
                    graph_path["target"],
                ]
                if graph_path.get("source")
                else [graph_path["target"]],
                provenance_tag="OBSERVED",
                conflict_flag=False,
                timestamp_retrieved=datetime.now(timezone.utc),
            )

            candidates.append(
                RetrievalCandidate(
                    memory_id=memory_id,
                    content=memory.content,
                    memory_type=MemoryTypeEnum(str(memory.memory_type)),
                    final_score=score,
                    trace=trace,
                )
            )

        return candidates
