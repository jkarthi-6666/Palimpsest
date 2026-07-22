from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from palimpsest.db import append_event, insert_memory
from palimpsest.embeddings import embed
from palimpsest.models import Event, Memory
from palimpsest.vectors import upsert
from palimpsest.write.extract import extract_facts


def _occurred_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        result = datetime.now(timezone.utc)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def ingest_session(
    session_messages: list[dict[str, Any]],
    user_id: UUID,
    session_id: UUID,
) -> list[Memory]:
    messages_with_ids: list[dict[str, Any]] = []
    for message in session_messages:
        event = Event(
            user_id=user_id,
            session_id=session_id,
            actor=str(message.get("actor") or message.get("role") or "unknown"),
            text=str(message.get("text") or ""),
            occurred_at=_occurred_at(message.get("occurred_at")),
        )
        event_id = append_event(event)
        messages_with_ids.append({**message, "event_id": event_id, "actor": event.actor})

    memories = extract_facts(messages_with_ids, user_id, session_id)
    if not memories:
        return []

    inserted: list[Memory] = []
    for memory in memories:
        try:
            insert_memory(memory)
            inserted.append(memory)
        except Exception as exc:
            print(f"  memory {memory.id} insert skipped: {exc}")

    if not inserted:
        return []

    try:
        vectors = embed(
            [memory.content for memory in inserted], input_type="passage"
        )
    except Exception as exc:
        print(f"  session vectorization skipped: {exc}")
        return []

    stored: list[Memory] = []
    for memory, vector in zip(inserted, vectors, strict=True):
        try:
            upsert(
                memory.id,
                vector,
                {"user_id": str(user_id), "mem_type": memory.mem_type.value},
            )
            stored.append(memory)
        except Exception as exc:
            print(f"  memory {memory.id} vector skipped: {exc}")
    return stored
