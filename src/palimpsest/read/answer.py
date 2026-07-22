from time import perf_counter
from uuid import UUID

from palimpsest.llm import complete_with_usage
from palimpsest.read.retrieve import retrieve


def answer(query: str, user_id: UUID) -> dict[str, object]:
    started = perf_counter()
    memories = retrieve(query, user_id, k=15)
    facts = "\n".join(f"- {memory.content}" for memory in memories)
    if not facts:
        facts = "- No relevant facts were retrieved."
    prompt = f"""Answer the question using ONLY the supplied facts.
If the facts do not contain the answer, say: I don't have that information.

Facts:
{facts}

Question: {query}
Answer:"""
    result = complete_with_usage(prompt, max_tokens=256)
    latency_ms = (perf_counter() - started) * 1000
    return {
        "text": result.text,
        "tokens": result.tokens,
        "latency_ms": round(latency_ms, 3),
        "used_memory_ids": [str(memory.id) for memory in memories],
    }
