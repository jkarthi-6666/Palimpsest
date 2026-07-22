import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from palimpsest.llm import complete
from palimpsest.models import Memory


EXTRACTOR_VERSION = "v1-phase1"
FAILURE_LOG = (
    Path(__file__).resolve().parents[3]
    / "eval"
    / "results"
    / "extraction_failures.jsonl"
)


def _log_extraction_failure(
    raw_response: str,
    session_messages: list[dict[str, Any]],
    session_id: UUID,
    reason: str,
) -> None:
    first_dia_id = session_messages[0].get("dia_id") if session_messages else None
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": str(session_id),
        "first_dia_id": first_dia_id,
        "reason": reason,
        "raw_response": raw_response,
    }
    try:
        FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FAILURE_LOG.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Diagnostics must never turn one malformed model response into a failed run.
        print(f"  extraction diagnostic could not be written: {exc}")


def _first_json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    # A local model may add prose or examples after its answer. Parse only the first
    # balanced object so a later brace does not invalidate an otherwise usable result.
    start = cleaned.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        character = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(cleaned[start : index + 1])
                except (json.JSONDecodeError, TypeError):
                    return None
                return value if isinstance(value, dict) else None
    return None


def _recover_memory_items(text: str) -> tuple[list[dict[str, Any]], int] | None:
    match = re.search(r'"memories"\s*:\s*\[', text)
    if not match:
        return None

    memories: list[dict[str, Any]] = []
    malformed_count = 0
    object_start: int | None = None
    object_depth = 0
    array_depth = 1
    in_string = False
    escaped = False

    # One malformed fact should not invalidate every other fact in the session.
    # Scan balanced objects inside the memories array, then decode each separately.
    for index in range(match.end(), len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "[":
            array_depth += 1
        elif character == "]":
            array_depth -= 1
            if array_depth == 0:
                return memories, malformed_count
        elif character == "{":
            if object_depth == 0:
                object_start = index
            object_depth += 1
        elif character == "}" and object_depth:
            object_depth -= 1
            if object_depth == 0 and object_start is not None:
                try:
                    item = json.loads(text[object_start : index + 1])
                except (json.JSONDecodeError, TypeError):
                    malformed_count += 1
                else:
                    if isinstance(item, dict):
                        memories.append(item)
                    else:
                        malformed_count += 1
                object_start = None
    if memories or malformed_count:
        if object_start is not None:
            malformed_count += 1
        return memories, malformed_count
    return None


def _is_empty_extraction_response(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return any(
        phrase in normalized
        for phrase in (
            "no durable memories",
            "no durable facts",
            "no memories to extract",
            "nothing to extract",
        )
    )


def extract_facts(
    session_messages: list[dict[str, Any]],
    user_id: UUID,
    session_id: UUID,
) -> list[Memory]:
    transcript = []
    source_event_ids: list[UUID] = []
    for message in session_messages:
        actor = str(message.get("actor") or message.get("role") or "unknown")
        speaker = str(message.get("speaker") or actor)
        occurred_at = message.get("occurred_at") or "unknown time"
        transcript.append(
            f"[{occurred_at}] {actor} ({speaker}): {message.get('text', '')}"
        )
        event_id = message.get("event_id")
        if event_id:
            try:
                source_event_ids.append(UUID(str(event_id)))
            except (TypeError, ValueError):
                pass

    prompt = f"""Extract durable memories from this entire conversation session.

Rules:
- Keep durable facts, preferences, plans, relationships, experiences, and reusable procedures.
- Drop ephemeral state and facts a general language model already knows.
- Make every memory self-contained: resolve pronouns, vague references, and relative dates.
- Attribute each memory to actor \"user\" or \"assistant\".
- mem_type must be episodic, semantic, or procedural.
- confidence must be a number from 0 to 1.
- Return ONLY valid JSON. Do not use Markdown fences or add explanations.
- If there are no durable memories, return exactly: {{"memories":[]}}
- Otherwise return exactly this shape:
{{"memories":[{{"content":"...","mem_type":"semantic","actor":"user","confidence":0.9}}]}}

Session:
{chr(10).join(transcript)}
"""
    try:
        raw_response = complete(prompt, max_tokens=2048)
    except Exception as exc:
        print(f"  extraction skipped: {exc}")
        return []
    parsed = _first_json_object(raw_response)
    if parsed is None and _is_empty_extraction_response(raw_response):
        return []
    if parsed is None:
        recovered = _recover_memory_items(raw_response)
        if recovered is not None:
            recovered_items, malformed_count = recovered
            if recovered_items:
                parsed = {"memories": recovered_items}
                if malformed_count:
                    print(
                        f"  extraction recovered: skipped {malformed_count} "
                        "malformed memory item(s)"
                    )
            elif malformed_count == 0:
                return []
        if parsed is None:
            _log_extraction_failure(
                raw_response, session_messages, session_id, "no valid JSON object"
            )
            print(f"  extraction skipped: invalid memory JSON (logged to {FAILURE_LOG})")
            return []
    if not isinstance(parsed.get("memories"), list):
        _log_extraction_failure(
            raw_response, session_messages, session_id, "missing memories list"
        )
        print(f"  extraction skipped: invalid memory JSON (logged to {FAILURE_LOG})")
        return []

    now = datetime.now(timezone.utc)
    memories: list[Memory] = []
    for raw in parsed["memories"]:
        if (
            not isinstance(raw, dict)
            or raw.get("actor") not in {"user", "assistant"}
            or not str(raw.get("content", "")).strip()
        ):
            continue
        try:
            memories.append(
                Memory(
                    content=raw["content"],
                    mem_type=raw["mem_type"],
                    user_id=user_id,
                    session_id=session_id,
                    actor=raw["actor"],
                    confidence=raw["confidence"],
                    valid_from=None,
                    valid_to=None,
                    observed_at=now,
                    mention_count=1,
                    source_event_ids=source_event_ids,
                    extractor_version=EXTRACTOR_VERSION,
                )
            )
        except (KeyError, TypeError, ValueError):
            # One malformed fact should not discard other valid facts in the batch.
            continue
    return memories
