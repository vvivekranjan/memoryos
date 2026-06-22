from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4
from aimemoryos.memory.working import WorkingMemory

import logging

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL_MINUTES: int = 60
DEFAULT_MAX_WORKING_MEMORIES: int = 25


@dataclass(slots=True)
class AgentSession:
    """
    Active agent session state.

    Fields
    ------
    session_id          : immutable UUID for this session
    agent_id            : owning agent partition key
    created_at          : UTC creation timestamp (immutable)
    expires_at          : rolling TTL; refreshed on every interaction
    conversation_turn   : monotonically increasing turn counter
    working_memories    : in-process working memory list (capped)
    metadata            : arbitrary caller-supplied bag
    active              : False after close_session(); reads only
    """

    session_id: UUID
    agent_id: str
    created_at: datetime
    expires_at: datetime
    conversation_turn: int = 0
    working_memories: list[WorkingMemory] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """
    Per-agent session coordinator.

    Responsibilities
    ----------------
    - Session lifecycle (create / get / refresh / close)
    - Working-memory scoping and eviction
    - Conversational turn tracking
    - TTL expiration (lazy + eager cleanup)

    Does NOT:
    - retrieve memories from any store
    - access FAISS, DuckDB, KuzuDB, or SQLite
    - replay events
    - enforce FAISS write guards (that is StorageOrchestrator's job)
    """

    def __init__(
        self,
        *,
        session_ttl_minutes: int = DEFAULT_SESSION_TTL_MINUTES,
        max_working_memories: int = DEFAULT_MAX_WORKING_MEMORIES,
    ) -> None:
        self.session_ttl_minutes = session_ttl_minutes
        self.max_working_memories = max_working_memories
        self._sessions: dict[UUID, AgentSession] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        *,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentSession:
        """
        Create and register an active session for agent_id.

        Returns the new AgentSession. The session is stored internally
        and retrievable via get_session() until it expires or is closed.
        """
        now = datetime.now(timezone.utc)
        session = AgentSession(
            session_id=uuid4(),
            agent_id=agent_id,
            created_at=now,
            expires_at=now + timedelta(minutes=self.session_ttl_minutes),
            metadata=metadata or {},
        )
        self._sessions[session.session_id] = session
        logger.debug(
            "session_manager | session created | "
            "agent_id={} session_id={} ttl_minutes={}",
            agent_id, session.session_id, self.session_ttl_minutes,
        )
        return session

    def get_session(self, *, session_id: UUID) -> AgentSession | None:
        """
        Return the active session or None if not found / expired.

        Expired sessions are closed lazily on this call.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self.is_expired(session):
            self.close_session(session_id=session_id)
            return None
        return session

    def refresh_session(self, *, session_id: UUID) -> None:
        """
        Extend the session TTL from now.

        Called internally by add_working_memory and advance_turn.
        No-op if the session does not exist.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.expires_at = (
            datetime.now(timezone.utc) + timedelta(minutes=self.session_ttl_minutes)
        )

    def close_session(self, *, session_id: UUID) -> None:
        """
        Mark the session inactive and remove it from the registry.

        Working memories held in the session are released. If any were
        promotable (promoted_to set), the caller is responsible for
        having already persisted them via StorageOrchestrator.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.active = False
        self._sessions.pop(session_id, None)
        logger.debug(
            "session_manager | session closed | session_id={}", session_id
        )

    # ------------------------------------------------------------------
    # Working memory management
    # ------------------------------------------------------------------

    def add_working_memory(
        self,
        *,
        session_id: UUID,
        memory: WorkingMemory,
    ) -> None:
        """
        Add a WorkingMemory to the session's in-process list.

        When the list exceeds max_working_memories, lower-importance entries
        are evicted (importance_score descending; top-N kept). The session
        TTL is refreshed on every call.

        Raises ValueError if the session does not exist or has expired.
        """
        session = self.get_session(session_id=session_id)
        if session is None:
            raise ValueError(f"Unknown or expired session: {session_id}")

        session.working_memories.append(memory)

        if len(session.working_memories) > self.max_working_memories:
            session.working_memories = sorted(
                session.working_memories,
                key=lambda m: m.importance_score,
                reverse=True,
            )[: self.max_working_memories]

        self.refresh_session(session_id=session_id)

    def working_memories(self, *, session_id: UUID) -> list[WorkingMemory]:
        """
        Return non-expired working memories for the session.

        Memories whose expires_at has passed are evicted from the session
        list on this call and not returned. Returns empty list if session
        is not found.
        """
        session = self.get_session(session_id=session_id)
        if session is None:
            return []

        now = datetime.now(timezone.utc)
        active = [
            m for m in session.working_memories
            if m.expires_at is None or now < m.expires_at
        ]
        session.working_memories = active
        return active

    # ------------------------------------------------------------------
    # Turn tracking
    # ------------------------------------------------------------------

    def advance_turn(self, *, session_id: UUID) -> int:
        """
        Increment the conversation turn counter and refresh the session TTL.

        Returns the new turn index (1-indexed from 0).
        Raises ValueError if the session does not exist or has expired.
        """
        session = self.get_session(session_id=session_id)
        if session is None:
            raise ValueError(f"Unknown or expired session: {session_id}")
        session.conversation_turn += 1
        self.refresh_session(session_id=session_id)
        return session.conversation_turn

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """
        Eagerly close all expired sessions.

        Returns the number of sessions closed. Safe to call from a
        background scheduler (core/scheduler.py).
        """
        expired_ids = [
            sid for sid, s in self._sessions.items() if self.is_expired(s)
        ]
        for sid in expired_ids:
            self.close_session(session_id=sid)
        if expired_ids:
            logger.debug(
                "session_manager | cleanup_expired | closed={}", len(expired_ids)
            )
        return len(expired_ids)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """
        Runtime session metrics for /health and Prometheus export.

        Returns counts of active sessions and their total working memories.
        """
        active = [s for s in self._sessions.values() if not self.is_expired(s)]
        return {
            "active_sessions": len(active),
            "working_memories": sum(len(s.working_memories) for s in active),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def is_expired(session: AgentSession) -> bool:
        """Return True when the session's TTL has elapsed."""
        return datetime.now(timezone.utc) >= session.expires_at


