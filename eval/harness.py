import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)

from judge import JUDGE_VERSION, judge  # noqa: E402
from palimpsest.db import init_schema  # noqa: E402
from palimpsest.embeddings import embedding_dimension  # noqa: E402
from palimpsest.read.answer import answer  # noqa: E402
from palimpsest.vectors import ensure_collection  # noqa: E402
from palimpsest.write.extract import EXTRACTOR_VERSION  # noqa: E402
from palimpsest.write.ingest import ingest_session  # noqa: E402


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def _session_time(value: str) -> datetime:
    parsed = datetime.strptime(value, "%I:%M %p on %d %B, %Y")
    # LoCoMo supplies no timezone. UTC makes deterministic aware datetimes without
    # pretending to know the speakers' local zones.
    return parsed.replace(tzinfo=timezone.utc)


def _sessions(sample: dict[str, Any]) -> list[tuple[int, list[dict[str, Any]]]]:
    conversation = sample["conversation"]
    speaker_a = conversation["speaker_a"]
    result = []
    for key, turns in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if not match or not isinstance(turns, list):
            continue
        number = int(match.group(1))
        base_time = _session_time(conversation[f"session_{number}_date_time"])
        messages = []
        for index, turn in enumerate(turns):
            text = str(turn.get("text") or "")
            if turn.get("blip_caption"):
                text = f"{text} [Image: {turn['blip_caption']}]".strip()
            messages.append(
                {
                    "actor": "user" if turn.get("speaker") == speaker_a else "assistant",
                    "speaker": turn.get("speaker"),
                    "text": text,
                    "occurred_at": base_time + timedelta(microseconds=index),
                    "dia_id": turn.get("dia_id"),
                }
            )
        result.append((number, messages))
    return sorted(result)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if row.get("correct") is not None]
    if not judged:
        return {"count": 0, "accuracy": 0.0, "avg_tokens_per_query": 0.0, "p50_latency_ms": 0.0}
    return {
        "count": len(judged),
        "accuracy": sum(bool(row["correct"]) for row in judged) / len(judged),
        "avg_tokens_per_query": sum(int(row["tokens"]) for row in judged) / len(judged),
        "p50_latency_ms": statistics.median(float(row["latency_ms"]) for row in judged),
    }


def _print_table(category_summaries: dict[str, dict[str, Any]], overall: dict[str, Any]) -> None:
    print("\nPhase 1 LoCoMo results")
    print(f"{'category':<16} {'n':>5} {'accuracy':>10} {'avg tokens':>12} {'p50 ms':>12}")
    print("-" * 59)
    for name in ["single-hop", "multi-hop", "temporal", "open-domain", "adversarial"]:
        metric = category_summaries.get(name, _summary([]))
        print(
            f"{name:<16} {metric['count']:>5} {metric['accuracy']:>10.3f} "
            f"{metric['avg_tokens_per_query']:>12.1f} {metric['p50_latency_ms']:>12.1f}"
        )
    print("-" * 59)
    print(
        f"{'overall':<16} {overall['count']:>5} {overall['accuracy']:>10.3f} "
        f"{overall['avg_tokens_per_query']:>12.1f} {overall['p50_latency_ms']:>12.1f}"
    )


def run(dataset: Path, limit_conversations: int | None, max_questions: int | None) -> dict[str, Any]:
    samples = json.loads(dataset.read_text())
    if limit_conversations is not None:
        samples = samples[:limit_conversations]

    init_schema()
    ensure_collection("memories", embedding_dimension())
    run_id = uuid4()
    rows: list[dict[str, Any]] = []

    total_questions = sum(len(sample.get("qa", [])) for sample in samples)
    if max_questions is not None:
        total_questions = min(total_questions, max_questions)
    question_number = 0

    for sample_index, sample in enumerate(samples, start=1):
        user_id = uuid5(NAMESPACE_URL, f"palimpsest:{run_id}:{sample['sample_id']}")
        sessions = _sessions(sample)
        for position, (session_number, messages) in enumerate(sessions, start=1):
            print(
                f"Conversation {sample_index}/{len(samples)}: ingesting session "
                f"{position}/{len(sessions)}"
            )
            session_id = uuid5(user_id, f"session:{session_number}")
            try:
                ingest_session(messages, user_id, session_id)
            except Exception as exc:
                print(f"  session skipped: {exc}")

        for qa in sample.get("qa", []):
            if max_questions is not None and question_number >= max_questions:
                break
            question_number += 1
            print(f"Question {question_number}/{total_questions}")
            category = CATEGORY_NAMES.get(int(qa["category"]), f"category-{qa['category']}")
            gold = qa.get("answer") or qa.get("adversarial_answer")
            if not gold:
                print("  question skipped: no gold answer")
                continue
            row: dict[str, Any] = {
                "sample_id": sample["sample_id"],
                "question": qa["question"],
                "gold_answer": gold,
                "category": category,
                "correct": None,
            }
            try:
                answered = answer(qa["question"], user_id)
                row.update(answered)
                verdict = judge(qa["question"], str(gold), str(answered["text"]))
                if verdict is not None:
                    row["correct"] = verdict.correct
                    row["judge_reason"] = verdict.reason
            except Exception as exc:
                row["error"] = str(exc)
                print(f"  question skipped: {exc}")
            rows.append(row)
        if max_questions is not None and question_number >= max_questions:
            break

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    category_summaries = {
        name: _summary(grouped.get(name, [])) for name in CATEGORY_NAMES.values()
    }
    overall = _summary(rows)
    result = {
        "run_note": {
            "extractor_version": EXTRACTOR_VERSION,
            "judge_version": JUDGE_VERSION,
            "embed_model": os.environ.get("EMBED_MODEL"),
            "llm_model": os.environ.get("LLM_MODEL"),
            "llm_base_url": os.environ.get("LLM_BASE_URL")
            or os.environ.get("NVIDIA_BASE_URL"),
            "judge_model": os.environ.get("JUDGE_MODEL"),
            "judge_base_url": os.environ.get("JUDGE_BASE_URL")
            or os.environ.get("NVIDIA_BASE_URL"),
            "k": 15,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset),
            "conversations": len(samples),
            "question_limit": max_questions,
        },
        "categories": category_summaries,
        "overall": overall,
        "items": rows,
    }
    output = ROOT / "eval" / "results" / "phase1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    _print_table(category_summaries, overall)
    print(f"\nWrote {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 LoCoMo baseline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval" / "datasets" / "locomo" / "locomo10.json",
    )
    parser.add_argument("--limit-conversations", type=int)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(args.dataset, args.limit_conversations, args.max_questions)


if __name__ == "__main__":
    main()
