from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from memory.models import WorkingMemory

DEFAULT_SESSION_TTL_MINUTES = 60
DEFAULT_MAX_WORKING_MEMORIES = 25

@dataclass(slots=True)
class AgentSession:
    """
    Active agent session state.
    """

    session_id: UUID
    agent_id: str
    created_at: datetime
    expires_at: datetime
    conversation_turn: int = 0
    working_memories: list[WorkingMemory] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True

class SessionManager:
    """
    Per-agent session coordinator.

    Responsibilities:
    - session lifecycle
    - working-memory scoping
    - conversational continuity
    - TTL expiration
    - active session ownership

    Does NOT:
    - retrieve memories
    - access vector stores
    - replay events
    """

    def __init__(
        self,
        *,
        session_ttl_minutes: int = DEFAULT_SESSION_TTL_MINUTES,
        max_working_memories: int = DEFAULT_MAX_WORKING_MEMORIES,
    ):
        self.session_ttl_minutes = session_ttl_minutes
        self.max_working_memories = max_working_memories 
        self._sessions: dict[UUID, AgentSession] = {}

    def create_session(
        self,
        *,
        agent_id: str,
        metadata: (
            dict[str, Any]
            | None
        ) = None,
    ) -> AgentSession:
        """
        Creates active session.
        """

        now = datetime.now(timezone.utc)

        session = AgentSession(
            session_id=uuid4(),
            agent_id=agent_id,
            created_at=now,
            expires_at=(
                now
                + timedelta(
                    minutes=(
                        self
                        .session_ttl_minutes
                    )
                )
            ),
            metadata=metadata or {},
        )

        self._sessions[session.session_id] = session

        return session

    def get_session(
        self,
        *,
        session_id: UUID,
    ) -> (
        AgentSession | None
    ):
        """
        Returns active session.
        """

        session = self._sessions.get(session_id)

        if session is None:
            return None

        if self.is_expired(session):
            self.close_session(
                session_id=session_id
            )
            return None

        return session

    def add_working_memory(
        self,
        *,
        session_id: UUID,
        memory: WorkingMemory,
    ) -> None:
        """
        Adds working memory to session.
        """

        session = self.get_session(session_id=session_id)

        if session is None:

            raise ValueError(
                f"Unknown session: "
                f"{session_id}"
            )

        session.working_memories.append(memory)

        if (
            len(
                session
                .working_memories
            )
            > self
            .max_working_memories
        ):

            session.working_memories = (
                sorted(
                    session
                    .working_memories,
                    key=lambda m: (
                        m.importance_score
                    ),
                    reverse=True,
                )[
                    : self
                    .max_working_memories
                ]
            )

        self.refresh_session(session_id=session_id)

    def working_memories(
        self,
        *,
        session_id: UUID,
    ) -> list[WorkingMemory]:
        """
        Returns non-expired working memories.
        """

        session = self.get_session(session_id=session_id)

        if session is None:
            return []

        active = [
            memory
            for memory
            in session.working_memories
            if (
                memory.expires_at is None
                or datetime.now(timezone.utc) < memory.expires_at
            )
        ]

        session.working_memories = active

        return active

    def advance_turn(
        self,
        *,
        session_id: UUID,
    ) -> int:
        """
        Advances conversational turn.
        """

        session = self.get_session(session_id=session_id)

        if session is None:

            raise ValueError(
                f"Unknown session: "
                f"{session_id}"
            )

        session.conversation_turn += 1
        self.refresh_session(
            session_id=session_id
        )

        return session.conversation_turn

    def refresh_session(
        self,
        *,
        session_id: UUID,
    ) -> None:
        """
        Extends session TTL.
        """

        session = self._sessions.get(session_id)

        if session is None:
            return

        session.expires_at = (
            datetime.now(
                timezone.utc
            )
            + timedelta(
                minutes=(
                    self
                    .session_ttl_minutes
                )
            )
        )

    def close_session(
        self,
        *,
        session_id: UUID,
    ) -> None:
        """
        Terminates session.
        """

        session = self._sessions.get(session_id)

        if session is None:
            return

        session.active = False
        self._sessions.pop(
            session_id,
            None,
        )

    @staticmethod
    def is_expired(
        session: AgentSession,
    ) -> bool:
        """
        Session TTL check.
        """

        return (
            datetime.now(
                timezone.utc
            )
            >= session.expires_at
        )

    def cleanup_expired(
        self,
    ) -> int:
        """
        Removes expired sessions.
        """

        expired = [
            session_id
            for session_id, session
            in self._sessions.items()
            if self.is_expired(
                session
            )
        ]

        for session_id in expired:
            self.close_session(
                session_id=session_id
            )

        return len(expired)

    def stats(
        self,
    ) -> dict[str, Any]:
        """
        Runtime session metrics.
        """

        active_sessions = [
            session
            for session in self._sessions.values()
            if not self.is_expired(session)
        ]

        return {
            "active_sessions": len(active_sessions),
            "working_memories": sum(
                len(
                    session
                    .working_memories
                )
                for session in active_sessions
            ),
        }

