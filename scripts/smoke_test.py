from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from uuid import uuid4

import psycopg
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env", override=True)

from palimpsest.db import append_event, get_event, init_schema  # noqa: E402
from palimpsest.embeddings import embed, embedding_dimension  # noqa: E402
from palimpsest.llm import complete  # noqa: E402
from palimpsest.models import Event  # noqa: E402
from palimpsest.vectors import ensure_collection  # noqa: E402


def main() -> None:
    init_schema()

    dimension = embedding_dimension()
    ensure_collection("memories", dimension)

    event = Event(
        user_id=uuid4(),
        session_id=uuid4(),
        actor="user",
        text="Phase 0 smoke-test event",
        occurred_at=datetime.now(timezone.utc),
    )
    first_id = append_event(event)
    second_id = append_event(event)
    assert first_id == second_id

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        count = connection.execute(
            "SELECT count(*) FROM events WHERE id = %s", (first_id,)
        ).fetchone()[0]
    assert count == 1, f"expected one event row, found {count}"

    stored = get_event(first_id)
    assert stored is not None
    assert stored.text == event.text
    print(stored.model_dump_json(indent=2))

    vectors = embed(["Palimpsest remembers this sentence."], input_type="passage")
    assert len(vectors[0]) == dimension
    print(f"Embedding dimension: {dimension}")

    llm_response = complete("Reply with exactly: OK")
    assert llm_response.strip(), "NVIDIA returned an empty response"
    print(f"NVIDIA LLM response: {llm_response.strip()}")
    print("PHASE 0 OK")


if __name__ == "__main__":
    main()
