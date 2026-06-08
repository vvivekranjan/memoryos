from typing import List
from memory.models import RetrievalCandidate
from memory.models import SessionScope

class SpreadingActivation:
    def __init__(self):
        pass
        
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
