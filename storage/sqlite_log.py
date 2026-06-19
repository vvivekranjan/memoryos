from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

import orjson

from core.exceptions import EventLogCorruptionError, SequenceGapError
from memory.models import (
    BaseEvent,
    EventTypeEnum,
    FeedbackPayload,
    IngestionPayload,
    LifecycleTransitionPayload,
    RetrievalPayload,
    TransactionStateEnum,
    TriggerEnum,
    utc_now,
)

# ============================================================
# CONSTANTS
# ============================================================

DB_PATH = Path("data/events.sqlite")

SCHEMA_VERSION = "1.0.0"

# ============================================================
# DDL
# ============================================================

CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT NOT NULL PRIMARY KEY,
    schema_version  TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    sequence_num    INTEGER NOT NULL,
    payload         TEXT NOT NULL,
    checksum        TEXT NOT NULL
) STRICT;
"""

CREATE_EVENT_TRANSACTION_TABLE = """
CREATE TABLE IF NOT EXISTS event_transactions (
    txn_id       TEXT NOT NULL PRIMARY KEY,
    agent_id     TEXT NOT NULL,
    event_id     TEXT NOT NULL REFERENCES events(event_id),
    state        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    resolved_at  TEXT,
    checksum     TEXT NOT NULL
) STRICT;
"""

CREATE_REFLECTION_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS reflection_audit (
    action_id       TEXT NOT NULL PRIMARY KEY,
    schema_version  TEXT NOT NULL,
    action_type     TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    affected_ids    TEXT NOT NULL,
    reversible      INTEGER NOT NULL DEFAULT 1,
    before_state    TEXT,
    after_state     TEXT,
    checksum        TEXT NOT NULL
) STRICT;
"""

CREATE_AGENT_SEQUENCES_TABLE = """
CREATE TABLE IF NOT EXISTS agent_sequences (
    agent_id         TEXT NOT NULL PRIMARY KEY,
    current_sequence INTEGER NOT NULL
) STRICT;
"""

CREATE_AGENT_SEQ_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_agent_seq
ON events(agent_id, sequence_num);
"""

CREATE_TYPE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_events_type
ON events(event_type);
"""

CREATE_OCCURRED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_events_occurred
ON events(occurred_at);
"""

class SQLiteEventLog:
    """
    Authoritative append-only SQLite
    event log.

    CRITICAL:
    SQLite remains the single source
    of truth for replay reconstruction.

    Responsibilities:
    - immutable event append
    - deterministic sequencing
    - checksum verification
    - transaction auditing
    - replay-safe reads
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
    ):
        self.db_path = db_path
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )

        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._initialise()

    def _initialise(self) -> None:
        """
        Creates tables and indexes.
        Safe on every startup.
        """

        with self.conn:

            self.conn.execute(CREATE_EVENTS_TABLE)
            self.conn.execute(CREATE_EVENT_TRANSACTION_TABLE)
            self.conn.execute(CREATE_REFLECTION_AUDIT_TABLE)
            self.conn.execute(CREATE_AGENT_SEQUENCES_TABLE)
            self.conn.execute(CREATE_AGENT_SEQ_INDEX)
            self.conn.execute(CREATE_TYPE_INDEX)
            self.conn.execute(CREATE_OCCURRED_INDEX)

    @staticmethod
    def compute_checksum(
        *,
        event_id: str,
        schema_version: str,
        event_type: str,
        occurred_at: str,
        sequence_num: int,
        payload: str,
    ) -> str:
        """
        Deterministic SHA-256 checksum.
        """

        raw = (
            f"{event_id}"
            f"{schema_version}"
            f"{event_type}"
            f"{occurred_at}"
            f"{sequence_num}"
            f"{payload}"
        )

        return hashlib.sha256(
            raw.encode()
        ).hexdigest()

    @staticmethod
    def compute_txn_checksum(
        *,
        txn_id: str,
        agent_id: str,
        event_id: str,
        state: str,
        created_at: str,
        resolved_at: str | None,
    ) -> str:

        raw = (
            f"{txn_id}"
            f"{agent_id}"
            f"{event_id}"
            f"{state}"
            f"{created_at}"
            f"{resolved_at or ''}"
        )

        return hashlib.sha256(
            raw.encode()
        ).hexdigest()

    def _next_sequence_num(
        self,
        *,
        agent_id: str,
    ) -> int:
        """
        Per-agent monotonic sequence.
        """

        row = self.conn.execute(
            """
            SELECT current_sequence
            FROM agent_sequences
            WHERE agent_id = ?
            """,
            (agent_id,),
        ).fetchone()

        if row is None:

            self.conn.execute(
                """
                INSERT INTO agent_sequences(
                    agent_id,
                    current_sequence
                )
                VALUES (?, ?)
                """,
                (agent_id, 0),
            )

            return 0

        next_seq = (
            row["current_sequence"]
            + 1
        )

        self.conn.execute(
            """
            UPDATE agent_sequences
            SET current_sequence = ?
            WHERE agent_id = ?
            """,
            (
                next_seq,
                agent_id,
            ),
        )

        return next_seq

    def _write_txn(
        self,
        *,
        agent_id: str,
        event_id: str,
        state: TransactionStateEnum = TransactionStateEnum.PENDING,
    ) -> str:

        txn_id = str(uuid4())

        created_at = (
            utc_now().isoformat()
        )

        checksum = (
            self.compute_txn_checksum(
                txn_id=txn_id,
                agent_id=agent_id,
                event_id=event_id,
                state=state.value,
                created_at=created_at,
                resolved_at=None,
            )
        )

        self.conn.execute(
            """
            INSERT INTO event_transactions (
                txn_id,
                agent_id,
                event_id,
                state,
                created_at,
                resolved_at,
                checksum
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                txn_id,
                agent_id,
                event_id,
                state.value,
                created_at,
                None,
                checksum,
            ),
        )

        return txn_id

    def _resolve_txn(
        self,
        *,
        txn_id: str,
        state: TransactionStateEnum,
    ) -> None:

        row = self.conn.execute(
            """
            SELECT
                agent_id,
                event_id,
                created_at
            FROM event_transactions
            WHERE txn_id = ?
            """,
            (txn_id,),
        ).fetchone()

        if row is None:
            raise ValueError(
                f"Unknown txn_id={txn_id}"
            )

        resolved_at = (
            utc_now().isoformat()
        )

        checksum = (
            self.compute_txn_checksum(
                txn_id=txn_id,
                agent_id=row[
                    "agent_id"
                ],
                event_id=row[
                    "event_id"
                ],
                state=state.value,
                created_at=row[
                    "created_at"
                ],
                resolved_at=resolved_at,
            )
        )

        self.conn.execute(
            """
            UPDATE event_transactions
            SET
                state = ?,
                resolved_at = ?,
                checksum = ?
            WHERE txn_id = ?
            """,
            (
                state.value,
                resolved_at,
                checksum,
                txn_id,
            ),
        )

    def append_event(
        self,
        *,
        event_type: EventTypeEnum,
        agent_id: str,
        payload: dict,
    ) -> BaseEvent:
        """
        Appends immutable event.
        """

        with self._lock:

            payload_bytes = (
                orjson.dumps(
                    payload,
                    option=(
                        orjson
                        .OPT_SORT_KEYS
                    ),
                )
            )

            payload_str = (
                payload_bytes.decode()
            )

            event_id = str(uuid4())

            occurred_at = utc_now()

            occurred_at_str = (
                occurred_at.isoformat()
            )

            with self.conn:

                sequence_num = (
                    self._next_sequence_num(
                        agent_id=agent_id
                    )
                )

                checksum = (
                    self.compute_checksum(
                        event_id=event_id,
                        schema_version=SCHEMA_VERSION,
                        event_type=event_type.value,
                        occurred_at=occurred_at_str,
                        sequence_num=sequence_num,
                        payload=payload_str,
                    )
                )

                self.conn.execute(
                    """
                    INSERT INTO events (
                        event_id,
                        schema_version,
                        event_type,
                        agent_id,
                        occurred_at,
                        sequence_num,
                        payload,
                        checksum
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        SCHEMA_VERSION,
                        event_type.value,
                        agent_id,
                        occurred_at_str,
                        sequence_num,
                        payload_str,
                        checksum,
                    ),
                )

                txn_id = (
                    self._write_txn(
                        agent_id=agent_id,
                        event_id=event_id,
                        state=(
                            TransactionStateEnum
                            .PENDING
                        ),
                    )
                )

                self._resolve_txn(
                    txn_id=txn_id,
                    state=(
                        TransactionStateEnum
                        .COMMITTED
                    ),
                )

            return BaseEvent(
                event_id=UUID(event_id),
                schema_version=(
                    SCHEMA_VERSION
                ),
                event_type=event_type,
                agent_id=agent_id,
                occurred_at=occurred_at,
                payload=payload,
                checksum=checksum,
                sequence_num=sequence_num,
            )

    def read_events(
        self,
        *,
        agent_id: str,
        before: (
            datetime | None
        ) = None,
        event_type: (
            EventTypeEnum | None
        ) = None,
    ) -> list[BaseEvent]:
        """
        Replay-safe event loading with:
        - checksum verification
        - sequence verification
        """

        query = """
        SELECT *
        FROM events
        WHERE agent_id = ?
        """

        with self._lock:

            params: list = [agent_id]

            if before:

                query += (
                    " AND occurred_at <= ?"
                )

                params.append(
                    before.isoformat()
                )

            if event_type:

                query += (
                    " AND event_type = ?"
                )

                params.append(
                    event_type.value
                )

            query += (
                " ORDER BY sequence_num ASC"
            )

            rows = self.conn.execute(
                query,
                params,
            ).fetchall()

            events: list[
                BaseEvent
            ] = []

            prev_seq = -1

            for row in rows:

                expected = (
                    self.compute_checksum(
                        event_id=row["event_id"],
                        schema_version=row["schema_version"],
                        event_type=row["event_type"],
                        occurred_at=row["occurred_at"],
                        sequence_num=row["sequence_num"],
                        payload=row["payload"],
                    )
                )

                if (expected != row["checksum"]):
                    raise (
                        EventLogCorruptionError(
                            f"Checksum mismatch "
                            f"event_id="
                            f"{row['event_id']}"
                        )
                    )

                if (
                    prev_seq == -1
                    and row[
                        "sequence_num"
                    ] != 0
                ):
                    raise (
                        SequenceGapError(
                            f"First sequence "
                            f"is "
                            f"{row['sequence_num']}"
                        )
                    )

                if (
                    prev_seq != -1
                    and row[
                        "sequence_num"
                    ]
                    != prev_seq + 1
                ):
                    raise (
                        SequenceGapError(
                            f"Gap detected "
                            f"at "
                            f"{row['sequence_num']}"
                        )
                    )

                prev_seq = row["sequence_num"]

                events.append(
                    BaseEvent(
                        event_id=UUID(row["event_id"]),
                        schema_version=row["schema_version"],
                        event_type=(
                            EventTypeEnum(
                                row[
                                    "event_type"
                                ]
                            )
                        ),
                        agent_id=row["agent_id"],
                        occurred_at=(
                            datetime
                            .fromisoformat(
                                row[
                                    "occurred_at"
                                ]
                            )
                        ),
                        payload=(
                            orjson.loads(
                                row["payload"]
                            )
                        ),
                        checksum=row["checksum"],
                        sequence_num=row["sequence_num"],
                    )
                )

            return events

    def load_events(
        self,
        *,
        agent_id: str,
    ) -> list[BaseEvent]:
        """
        Replay-compatible alias.
        """

        return self.read_events(
            agent_id=agent_id
        )

    def log_events(
        self,
        *,
        agent_id: str,
    ) -> list[dict]:
        """
        Replay adapter returning dict-shaped events.
        """

        events = self.read_events(agent_id=agent_id)

        return [
            {
                "event_id": str(event.event_id),
                "schema_version": event.schema_version,
                "event_type": event.event_type,
                "agent_id": event.agent_id,
                "occurred_at": event.occurred_at,
                "sequence_num": event.sequence_num,
                "payload": event.payload,
                "checksum": event.checksum,
            }
            for event in events
        ]

    def load_checksums(
        self,
        *,
        agent_id: str,
    ) -> list[dict]:
        """
        Returns checksum validity per event.
        """

        rows = self.conn.execute(
            """
            SELECT
                event_id,
                schema_version,
                event_type,
                occurred_at,
                sequence_num,
                payload,
                checksum
            FROM events
            WHERE agent_id = ?
            """,
            (agent_id,),
        ).fetchall()

        results: list[dict] = []

        for row in rows:
            expected = self.compute_checksum(
                event_id=row["event_id"],
                schema_version=row["schema_version"],
                event_type=row["event_type"],
                occurred_at=row["occurred_at"],
                sequence_num=row["sequence_num"],
                payload=row["payload"],
            )

            results.append(
                {
                    "event_id": row["event_id"],
                    "valid": expected == row["checksum"],
                }
            )

        return results

    def load_sequence_numbers(
        self,
        *,
        agent_id: str,
    ) -> list[int]:
        """
        Returns all sequence numbers for an agent.
        """

        rows = self.conn.execute(
            """
            SELECT sequence_num
            FROM events
            WHERE agent_id = ?
            ORDER BY sequence_num ASC
            """,
            (agent_id,),
        ).fetchall()

        return [row[0] for row in rows]

    def get_unresolved_transactions(
        self,
    ) -> list[sqlite3.Row]:
        """
        Returns unresolved transactions.
        """

        rows = self.conn.execute(
            """
            SELECT *
            FROM event_transactions
            WHERE state != ?
            """,
            (
                TransactionStateEnum
                .COMMITTED
                .value,
            ),
        ).fetchall()

        return list(rows)

    # def heal_pending_transactions(self) -> int:
    #     """
    #     Marks unresolved transactions as corrupted.
    #     """

    #     unresolved = self.get_unresolved_transactions()

    #     healed = 0

    #     for row in unresolved:
    #         state = row["state"]
    #         if state != TransactionStateEnum.PENDING.value:
    #             continue

    #         self._resolve_txn(
    #             txn_id=row["txn_id"],
    #             state=TransactionStateEnum.CORRUPTED,
    #         )

    #         healed += 1

    #     return healed

    def get_corrupted_event_ids(
        self,
        *,
        agent_id: (
            str | None
        ) = None,
    ) -> list[UUID]:
        """
        Detects corrupted events.
        """

        query = (
            "SELECT * FROM events"
        )

        params: list = []

        if agent_id:

            query += (
                " WHERE agent_id = ?"
            )

            params.append(agent_id)

        rows = self.conn.execute(
            query,
            params,
        ).fetchall()

        corrupted: list[
            UUID
        ] = []

        for row in rows:

            expected = (
                self.compute_checksum(
                    event_id=row["event_id"],
                    schema_version=row["schema_version"],
                    event_type=row["event_type"],
                    occurred_at=row["occurred_at"],
                    sequence_num=row["sequence_num"],
                    payload=row["payload"],
                )
            )

            if (expected != row["checksum"]):
                corrupted.append(
                    UUID(
                        row["event_id"]
                    )
                )

        return corrupted

    def log_ingestion(
        self,
        *,
        agent_id: str,
        payload: (
            IngestionPayload
        ),
    ) -> BaseEvent:

        return self.append_event(
            event_type=(
                EventTypeEnum
                .MEMORY_INGESTED
            ),
            agent_id=agent_id,
            payload=(
                payload.model_dump(
                    mode="json"
                )
            ),
        )

    def log_retrieval(
        self,
        *,
        agent_id: str,
        payload: (
            RetrievalPayload
        ),
    ) -> BaseEvent:

        return self.append_event(
            event_type=(
                EventTypeEnum
                .MEMORY_RETRIEVED
            ),
            agent_id=agent_id,
            payload=(
                payload.model_dump(
                    mode="json"
                )
            ),
        )

    def log_feedback(
        self,
        *,
        agent_id: str,
        payload: (
            FeedbackPayload
        ),
    ) -> BaseEvent:

        return self.append_event(
            event_type=(
                EventTypeEnum
                .FEEDBACK_RECEIVED
            ),
            agent_id=agent_id,
            payload=(
                payload.model_dump(
                    mode="json"
                )
            ),
        )

    def log_lifecycle_transition(
        self,
        *,
        agent_id: str,
        payload: LifecycleTransitionPayload,
    ) -> BaseEvent:

        return self.append_event(
            event_type=EventTypeEnum.MEMORY_LIFECYCLE_TRANSITION,
            agent_id=agent_id,
            payload=payload.model_dump(mode="json"),
        )

    def snapshot(
        self,
        output_path: Path,
        since: datetime | None = None,
        verify: bool = True,
    ) -> None:
        """
        Export the SQLite event log to a portable snapshot file.
        """
        if since is not None:
            raise NotImplementedError("Partial snapshots using 'since' are not yet supported")

        if verify:
            corrupted = self.get_corrupted_event_ids()
            if corrupted:
                raise EventLogCorruptionError(
                    f"Checksum validation failed for {len(corrupted)} events. Snapshot aborted."
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(output_path) as dst:
            with self._lock:
                self.conn.backup(dst)

    def close(self) -> None:
        """
        Graceful shutdown.
        """

        self.conn.close()


def compute_checksum(
    event_id: str,
    schema_version: str,
    event_type: str,
    occurred_at: str,
    sequence_num: int,
    payload: str,
) -> str:
    """
    Module-level wrapper for replay checksum validation.
    """

    return SQLiteEventLog.compute_checksum(
        event_id=event_id,
        schema_version=schema_version,
        event_type=event_type,
        occurred_at=occurred_at,
        sequence_num=sequence_num,
        payload=payload,
    )

