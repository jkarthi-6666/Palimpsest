import json
import re
from dataclasses import dataclass

from palimpsest.llm import complete


JUDGE_VERSION = "v1"

JUDGE_PROMPT = """Judge whether the predicted answer is correct given the gold answer.
Allow paraphrases and equivalent date formats. Do not require extra details beyond the gold.
Return JSON only: {{"correct": true, "reason": "one short line"}}.

Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}
"""


@dataclass(frozen=True)
class JudgeResult:
    correct: bool
    reason: str


def judge(question: str, gold_answer: str, predicted_answer: str) -> JudgeResult | None:
    try:
        raw = complete(
            JUDGE_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer,
            ),
            max_tokens=128,
            config_prefix="JUDGE",
        )
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        value = json.loads(cleaned[start : end + 1])
        if not isinstance(value.get("correct"), bool):
            return None
        return JudgeResult(bool(value["correct"]), str(value.get("reason", "")))
    except Exception as exc:
        print(f"  judge skipped: {exc}")
        return None
