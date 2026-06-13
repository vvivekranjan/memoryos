from typing import List
from uuid import UUID
from .engine import RetrievalCandidate
from memory.models import SessionScope

try:
    import kuzu  # type: ignore
except ImportError:  # pragma: no cover
    kuzu = None  # type: ignore

DECAY_PER_HOP = 0.5 # configurable in config.yaml
SESSION_TTL_S = 3600 # activation resets on session end

class SpreadingActivation:
    def __init__(self):
        pass

    # def propagate(memory_id: UUID, depth: int = 2):
    #     for hop in range(1, depth + 1):
    #         neighbours = kuzu.bfs(memory_id, hops=hop)
    #         boost = DECAY_PER_HOP ** hop
    #         for n in neighbours:
    #             session_activation[n.memory_id] += boost
        
    def apply(self, candidates: List[RetrievalCandidate], session_scope: SessionScope) -> None:
        """Boosts graph-adjacent memories within current session scope."""
        if not session_scope or not session_scope.working_memory_ids:
            return
            
        working_set = set(session_scope.working_memory_ids)
        
        for candidate in candidates:
            # If the candidate is directly in the working set, huge boost
            if candidate.memory_id in working_set:
                candidate.trace.activation_boost = 1.5
                candidate.final_score *= 1.5
            
            # If it shares graph path nodes with working memory (stub logic for Phase 1)
            elif candidate.trace.graph_path:
                candidate.trace.activation_boost = 1.1
                candidate.final_score *= 1.1
