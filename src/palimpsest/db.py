import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from palimpsest.models import Event, Memory


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    actor TEXT NOT NULL,
    text TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    tombstoned BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    mem_type TEXT NOT NULL CHECK (mem_type IN ('episodic', 'semantic', 'procedural')),
    user_id UUID NOT NULL,
    agent_id UUID,
    session_id UUID,
    org_id UUID,
    actor TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    observed_at TIMESTAMPTZ NOT NULL,
    superseded_by UUID REFERENCES memories(id),
    supersede_kind TEXT,
    mention_count INTEGER NOT NULL DEFAULT 1 CHECK (mention_count >= 0),
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
    last_accessed TIMESTAMPTZ,
    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    sensitivity TEXT NOT NULL DEFAULT 'normal',
    source_event_ids UUID[] NOT NULL DEFAULT '{}',
    extractor_version TEXT,
    deleted_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS memories_current_user_idx
    ON memories(user_id)
    WHERE valid_to IS NULL AND deleted_at IS NULL;
"""


def _database_url() -> str:
    try:
        return os.environ["DATABASE_URL"]
    except KeyError as exc:
        raise RuntimeError("DATABASE_URL is not set; copy .env.example to .env") from exc


def _event_id(event: Event) -> UUID:
    # JSON gives the tuple unambiguous boundaries. UTC normalization means two
    # equivalent instants with different offsets produce the same dedup key.
    occurred_at = event.occurred_at.astimezone(timezone.utc).isoformat(timespec="microseconds")
    identity = json.dumps(
        [str(event.user_id), str(event.session_id), occurred_at, event.text],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = bytearray(hashlib.sha256(identity).digest()[:16])
    # Mark the digest as an RFC 4122 variant/version-5 UUID while retaining its
    # deterministic content-derived bits.
    digest[6] = (digest[6] & 0x0F) | 0x50
    digest[8] = (digest[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(digest))


def init_schema() -> None:
    with psycopg.connect(_database_url()) as connection:
        connection.execute(SCHEMA_SQL)


def append_event(event: Event) -> UUID:
    event_id = _event_id(event)
    with psycopg.connect(_database_url()) as connection:
        connection.execute(
            """
            INSERT INTO events (
                id, user_id, session_id, actor, text,
                occurred_at, ingested_at, tombstoned
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                event_id,
                event.user_id,
                event.session_id,
                event.actor,
                event.text,
                event.occurred_at,
                event.ingested_at,
                event.tombstoned,
            ),
        )
    return event_id


def get_event(event_id: UUID) -> Event | None:
    with psycopg.connect(_database_url(), row_factory=dict_row) as connection:
        row = connection.execute(
            "SELECT * FROM events WHERE id = %s", (event_id,)
        ).fetchone()
    return Event.model_validate(row) if row else None


def get_current_facts(user_id: UUID, as_of: datetime | None = None) -> list[Memory]:
    if as_of is not None and as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")

    if as_of is None:
        query = """
            SELECT * FROM memories
            WHERE user_id = %s
              AND valid_to IS NULL
              AND deleted_at IS NULL
            ORDER BY observed_at, id
        """
        params = (user_id,)
    else:
        # A historical read must satisfy both knowledge time (observed_at) and
        # valid time (valid_from/valid_to); filtering only one axis is not bitemporal.
        query = """
            SELECT * FROM memories
            WHERE user_id = %s
              AND (deleted_at IS NULL OR deleted_at > %s)
              AND observed_at <= %s
              AND valid_from <= %s
              AND (valid_to IS NULL OR valid_to > %s)
            ORDER BY observed_at, id
        """
        params = (user_id, as_of, as_of, as_of, as_of)

    with psycopg.connect(_database_url(), row_factory=dict_row) as connection:
        rows = connection.execute(query, params).fetchall()
    return [Memory.model_validate(row) for row in rows]
