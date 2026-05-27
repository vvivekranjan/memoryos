from __future__ import annotations

import duckdb
import orjson
import sys
import asyncio

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.models import (
    BaseMemory,
    EpisodicMemory,
    WorkingMemory,
    MemoryTypeEnum,
    SpeakerRoleEnum,
)

# ============================================================
# CONSTANTS
# ============================================================

DB_PATH = Path("data/memory.duckdb")

class DuplicateMemoryError(Exception):
    pass

class MemoryNotFoundError(Exception):
    pass

class InvalidLifecycleTransitionError(Exception):
    pass

# ============================================================
# SQL
# ============================================================

CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id UUID PRIMARY KEY,

    schema_version VARCHAR NOT NULL,
    memory_type VARCHAR NOT NULL,

    agent_id VARCHAR NOT NULL,

    content TEXT NOT NULL,
    sha256 VARCHAR NOT NULL,

    created_at TIMESTAMP NOT NULL,
    last_accessed_at TIMESTAMP NOT NULL,

    access_count INTEGER NOT NULL,

    importance_score DOUBLE NOT NULL,

    metadata JSON
);
"""

CREATE_EPISODIC_TABLE = """
CREATE TABLE IF NOT EXISTS episodic_memories (
    memory_id UUID PRIMARY KEY,

    session_id UUID NOT NULL,

    turn_index INTEGER NOT NULL,

    speaker_role VARCHAR NOT NULL,

    referenced_memory_ids JSON,

    is_system_message BOOLEAN NOT NULL,

    tool_call_id VARCHAR
);
"""

CREATE_WORKING_TABLE = """
CREATE TABLE IF NOT EXISTS working_memories (
    memory_id UUID PRIMARY KEY,

    session_id UUID NOT NULL,

    ttl_seconds INTEGER NOT NULL,

    promoted_to UUID,

    scratch_data JSON,

    expires_at TIMESTAMP
);
"""

CREATE_AGENT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memories_agent
ON memories(agent_id);
"""

CREATE_TYPE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memories_type
ON memories(memory_type);
"""

CREATE_CREATED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memories_created
ON memories(created_at);
"""

CREATE_IMPORTANCE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memories_importance
ON memories(importance_score);
"""

CREATE_SHA256_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memories_sha256
ON memories(sha256);
"""

# ============================================================
# HELPERS
# ============================================================


def json_dumps(value) -> str:
    return orjson.dumps(
        value,
        option=orjson.OPT_SORT_KEYS,
    ).decode()


def json_loads(value: str | dict | list | None):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return orjson.loads(value)


def coerce_uuid(value: UUID | str) -> UUID:
    return value if isinstance(value, UUID) else UUID(value)

class DuckDBStore:
    """
    Materialized memory state store.

    IMPORTANT:
    DuckDB is reconstructable from SQLite replay.

    DuckDB is derived state.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH
    ):
        self.db_path = db_path

    def _connect(self) -> duckdb.DuckDBPyConnection:
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return duckdb.connect(str(self.db_path))

    async def content_hash_exists(
        self,
        *,
        agent_id: str,
        content_hash: str,
    ) -> bool:
        """
        Implements DedupStore.content_hash_exists.
        """
        def _check():
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM memories WHERE agent_id = ? AND sha256 = ?",
                    [agent_id, content_hash],
                ).fetchone()
                return row is not None
        
        return await asyncio.to_thread(_check)

    def initialise(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_MEMORIES_TABLE)
            conn.execute(CREATE_EPISODIC_TABLE)
            conn.execute(CREATE_WORKING_TABLE)

            conn.execute(CREATE_AGENT_INDEX)
            conn.execute(CREATE_TYPE_INDEX)
            conn.execute(CREATE_CREATED_INDEX)
            conn.execute(CREATE_IMPORTANCE_INDEX)
            conn.execute(CREATE_SHA256_INDEX)

    def insert_memory(
        self,
        memory: BaseMemory,
    ) -> None:
        """
        Inserts memory into materialized state store.

        IMPORTANT:
        No orchestration logic here.
        No event logging here.
        """

        with self._connect() as conn:

            conn.begin()

            try:

                existing = conn.execute(
                    """
                    SELECT memory_id
                    FROM memories
                    WHERE memory_id = ?
                    """,
                    [str(memory.memory_id)],
                ).fetchone()

                if existing:
                    raise DuplicateMemoryError(
                        f"Memory already exists: {memory.memory_id}"
                    )
                
                conn.execute(
                    """
                    INSERT INTO memories (
                        memory_id,
                        schema_version,
                        memory_type,
                        agent_id,

                        content,
                        sha256,

                        created_at,
                        last_accessed_at,

                        access_count,

                        importance_score,

                        metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(memory.memory_id),
                        memory.schema_version,
                        memory.memory_type.value,

                        memory.agent_id,

                        memory.content,
                        memory.sha256,

                        memory.created_at,
                        memory.last_accessed_at,

                        memory.access_count,
                        memory.importance_score,

                        json_dumps(memory.metadata),
                    ],
                )

                # Episodic
                if isinstance(memory, EpisodicMemory):

                    conn.execute(
                        """
                        INSERT INTO episodic_memories (
                            memory_id,
                            session_id,
                            turn_index,
                            speaker_role,
                            referenced_memory_ids,
                            is_system_message,
                            tool_call_id
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            str(memory.memory_id),
                            str(memory.session_id),
                            memory.turn_index,
                            memory.speaker_role.value if hasattr(memory.speaker_role, "value") else str(memory.speaker_role),

                            json_dumps(
                                [
                                    str(x)
                                    for x in memory.referenced_memory_ids
                                ]
                            ),

                            memory.is_system_message,

                            memory.tool_call_id,
                        ],
                    )

                # Working
                elif isinstance(memory, WorkingMemory):

                    conn.execute(
                        """
                        INSERT INTO working_memories (
                            memory_id,
                            session_id,
                            ttl_seconds,
                            promoted_to,
                            scratch_data,
                            expires_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            str(memory.memory_id),
                            str(memory.session_id),

                            memory.ttl_seconds,

                            (
                                str(memory.promoted_to)
                                if memory.promoted_to
                                else None
                            ),

                            json_dumps(memory.scratch_data),

                            memory.expires_at,
                        ],
                    )
            
                conn.commit()
            
            except Exception:
                conn.rollback()
                raise
    

    def get_memory(
        self,
        memory_id: UUID,
    ) -> BaseMemory:
        
        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT
                    memory_id,
                    schema_version,
                    memory_type,
                    agent_id,
                    content,
                    sha256,
                    created_at,
                    last_accessed_at,
                    access_count,
                    importance_score,
                    metadata
                FROM memories
                WHERE memory_id = ?
                """,
                [str(memory_id)],
            ).fetchone()

            if row is None:
                raise MemoryNotFoundError(
                    f"Memory not found: {memory_id}"
                )

            base_data = {
                "memory_id": coerce_uuid(row[0]),
                "schema_version": row[1],
                "memory_type": MemoryTypeEnum(row[2]),
                "agent_id": row[3],

                "content": row[4],
                "sha256": row[5],

                "created_at": row[6],
                "last_accessed_at": row[7],

                "access_count": row[8],

                "importance_score": row[9],

                "metadata": json_loads(row[10]),
            }

            # Episodic
            if row[2] == MemoryTypeEnum.EPISODIC.value:

                ext = conn.execute(
                    """
                    SELECT *
                    FROM episodic_memories
                    WHERE memory_id = ?
                    """,
                    [str(memory_id)],
                ).fetchone()

                if ext is None:
                    raise MemoryNotFoundError(
                        f"Episodic extension missing for: {memory_id}"
                    )

                base_data.update(
                    {
                        "session_id": coerce_uuid(ext[1]),
                        "turn_index": ext[2],
                        "speaker_role": SpeakerRoleEnum(ext[3]),

                        "referenced_memory_ids": [
                            coerce_uuid(x)
                            for x in json_loads(ext[4])
                        ],

                        "is_system_message": ext[5],

                        "tool_call_id": ext[6],
                    }
                )

                return EpisodicMemory(**base_data)

            # Working
            elif row[2] == MemoryTypeEnum.WORKING.value:

                ext = conn.execute(
                    """
                    SELECT *
                    FROM working_memories
                    WHERE memory_id = ?
                    """,
                    [str(memory_id)],
                ).fetchone()

                if ext is None:
                    raise MemoryNotFoundError(
                        f"Working extension missing for: {memory_id}"
                    )

                base_data.update(
                    {
                        "session_id": coerce_uuid(ext[1]),

                        "ttl_seconds": ext[2],

                        "promoted_to": (
                            coerce_uuid(ext[3])
                            if ext[3]
                            else None
                        ),

                        "scratch_data": json_loads(ext[4]),

                        "expires_at": ext[5],
                    }
                )

                return WorkingMemory(**base_data)

            raise ValueError(
                f"Unsupported memory type: {row[2]}"
            )

    def update_access_metadata(
        self,
        memory_id: UUID,
    ) -> None:
        
        now = datetime.now(timezone.utc)

        with self._connect() as conn:

            conn.execute(
                """
                UPDATE memories
                SET
                    access_count = access_count + 1,
                    last_accessed_at = ?
                WHERE memory_id = ?
                """,
                [
                    now,
                    str(memory_id),
                ],
            )

    def get_active_memories(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> list[BaseMemory]:
        
        with self._connect() as conn:

            rows = conn.execute(
                """
                SELECT memory_id
                FROM memories
                WHERE
                    agent_id = ?
                ORDER BY importance_score DESC
                LIMIT ?
                """,
                [
                    agent_id,
                    limit,
                ],
            ).fetchall()

        return [
            self.get_memory(coerce_uuid(row[0]))
            for row in rows
        ]

    async def get_memories_by_ids(
        self,
        *,
        agent_id: str,
        memory_ids: list[UUID],
    ) -> list[BaseMemory]:
        """
        Batch fetch for retrieval engine.
        """

        if not memory_ids:
            return []

        placeholders = ", ".join(
            ["?"] * len(memory_ids)
        )

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_id
                FROM memories
                WHERE agent_id = ?
                  AND memory_id IN ({placeholders})
                """,
                [agent_id, *map(str, memory_ids)],
            ).fetchall()

        for row in rows:
            self.update_access_metadata(coerce_uuid(row[0]))

        return [
            self.get_memory(coerce_uuid(row[0]))
            for row in rows
        ]

    async def content_hash_exists(
        self,
        *,
        agent_id: str,
        content_hash: str,
    ) -> bool:
        """
        Deduplication lookup by sha256.
        """

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM memories
                WHERE agent_id = ?
                  AND sha256 = ?
                LIMIT 1
                """,
                [agent_id, content_hash],
            ).fetchone()

        return row is not None

    def projection_exists(
        self,
        *,
        memory_id: UUID,
    ) -> bool:
        """
        Projection integrity check.
        """

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM memories
                WHERE memory_id = ?
                LIMIT 1
                """,
                [str(memory_id)],
            ).fetchone()

        return row is not None

    def projection_count(
        self,
        *,
        agent_id: str,
    ) -> int:
        """
        Projection count for integrity checks.
        """

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(1)
                FROM memories
                WHERE agent_id = ?
                """,
                [agent_id],
            ).fetchone()

        return int(row[0]) if row else 0

