from __future__ import annotations

from typing import Dict, List, Optional
from uuid import UUID

from aimemoryos.core.config import config
from aimemoryos.retrieval.engine import MemoryResult, RetrievalTrace

# Standard RRF constant (default 60)
DEFAULT_RRF_K: int = getattr(config, "rrf_k", 60)

# Default channel weights for hybrid fusion
DEFAULT_CHANNEL_WEIGHTS: Dict[str, float] = {
    "vector": getattr(config, "vector_weight", 0.7),
    "graph": getattr(config, "graph_weight", 0.3),
}


def calculate_rrf_score(
    ranks: Dict[str, int],
    weights: Optional[Dict[str, float]] = None,
    k: int = DEFAULT_RRF_K,
) -> float:
    """
    Compute weighted Reciprocal Rank Fusion score from 1-based retriever ranks.

    Formula:
        RRF(d) = sum_{channel} ( weight_{channel} * (1.0 / (k + rank_{channel})) )
    """
    channel_weights = weights or DEFAULT_CHANNEL_WEIGHTS
    score = 0.0
    for channel, rank in ranks.items():
        if rank <= 0:
            continue
        w = channel_weights.get(channel, 1.0)
        score += w * (1.0 / (k + rank))
    return score


def calculate_composite_score(
    rrf_base_score: float,
    importance_score: float = 0.0,
    provenance_confidence: float = 1.0,
    activation_boost: float = 0.0,
) -> float:
    """
    Adjust base RRF score with memory salience, provenance, and graph activation.
    """
    # Scale RRF score with importance, provenance confidence, and activation boost
    importance_multiplier = 1.0 + (max(0.0, min(1.0, importance_score)) * 0.2)
    activation_multiplier = 1.0 + max(0.0, activation_boost)
    provenance_multiplier = max(0.1, min(1.0, provenance_confidence))

    return rrf_base_score * importance_multiplier * activation_multiplier * provenance_multiplier


def fuse_memory_results(
    ranked_lists: Dict[str, List[MemoryResult]],
    weights: Optional[Dict[str, float]] = None,
    k: int = DEFAULT_RRF_K,
) -> List[MemoryResult]:
    """
    Fuses multiple ranked MemoryResult lists using Weighted Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : Dict[str, List[MemoryResult]]
        Mapping of retriever name (e.g. "vector", "graph") to its ranked results.
    weights : Optional[Dict[str, float]]
        Weights per retriever channel.
    k : int
        RRF smoothing constant (default 60).

    Returns
    -------
    List[MemoryResult] sorted by fused final_score descending.
    """
    if not ranked_lists:
        return []

    channel_weights = weights or DEFAULT_CHANNEL_WEIGHTS

    # 1. Map memory_id -> MemoryResult and collect ranks per channel
    memory_map: Dict[UUID, MemoryResult] = {}
    ranks_by_memory: Dict[UUID, Dict[str, int]] = {}
    graph_paths_by_memory: Dict[UUID, List[str]] = {}
    activation_boosts_by_memory: Dict[UUID, float] = {}

    for channel, results in ranked_lists.items():
        for rank_idx, res in enumerate(results, start=1):
            mid = res.memory.memory_id
            if mid not in memory_map:
                memory_map[mid] = res
                ranks_by_memory[mid] = {}

            ranks_by_memory[mid][channel] = rank_idx

            if res.trace:
                if res.trace.graph_path:
                    graph_paths_by_memory[mid] = res.trace.graph_path
                if res.trace.activation_boost > activation_boosts_by_memory.get(mid, 0.0):
                    activation_boosts_by_memory[mid] = res.trace.activation_boost

    # 2. Compute fused score and build merged trace
    fused_results: List[MemoryResult] = []

    for mid, base_res in memory_map.items():
        ranks = ranks_by_memory[mid]
        rrf_base = calculate_rrf_score(ranks=ranks, weights=channel_weights, k=k)

        memory = base_res.memory
        importance = getattr(memory, "importance_score", 0.5)
        prov_conf = getattr(memory, "provenance_confidence", 1.0)
        prov = getattr(memory, "provenance", "OBSERVED")
        provenance_str = prov.value if hasattr(prov, "value") else str(prov)

        act_boost = activation_boosts_by_memory.get(mid, 0.0)

        final_score = calculate_composite_score(
            rrf_base_score=rrf_base,
            importance_score=importance,
            provenance_confidence=prov_conf,
            activation_boost=act_boost,
        )

        merged_trace = RetrievalTrace(
            memory_id=mid,  # Preserves UUID type directly
            final_score=final_score,
            retrieved_by=sorted(ranks.keys()),
            vector_rank=ranks.get("vector"),
            graph_rank=ranks.get("graph"),
            temporal_rank=ranks.get("temporal"),
            importance_score=importance,
            recency_boost=base_res.trace.recency_boost if base_res.trace else 0.0,
            activation_boost=act_boost,
            graph_path=graph_paths_by_memory.get(mid),
            provenance=provenance_str,
            provenance_confidence=prov_conf,
        )

        fused_results.append(MemoryResult(memory=memory, trace=merged_trace))

    # 3. Sort by final_score descending
    fused_results.sort(key=lambda r: r.trace.final_score, reverse=True)
    return fused_results
