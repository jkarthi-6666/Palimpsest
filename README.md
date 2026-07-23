# Palimpsest

Palimpsest is a from-scratch LLM memory system with durable storage, semantic
retrieval, memory extraction, question answering, and evaluation.

## Setup

Prerequisites: Docker with Compose and Python 3.11 or newer.

```bash
cp .env.example .env
# Edit .env and replace NVIDIA_API_KEY with a real key from build.nvidia.com.
docker compose up -d

python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/smoke_test.py
```

The real `.env` is gitignored. Never commit `NVIDIA_API_KEY`. Embeddings use
NVIDIA's OpenAI-compatible remote API; chat can use NVIDIA or a local compatible
server. The embedding wrapper supports `passage` mode for indexed facts and
`query` mode for user questions. A successful test finishes with `PALIMPSEST OK`.

Docker publishes Palimpsest's Postgres on host port `5433`, leaving the usual
local Postgres port `5432` available for other projects.

Postgres is the source of truth: it owns the append-only event log and the
bitemporal memory records. Qdrant is only a replaceable search index; vectors use
payload IDs that point back to authoritative Postgres rows. This keeps semantic
search fast without allowing the vector store to become a second, conflicting
source of truth.

Storage remains entirely local in Docker. NVIDIA provides embeddings. Chat
completion can use NVIDIA or a local OpenAI-compatible server selected entirely
through `.env`.

## Run the Phase 1 LoCoMo baseline

Place the LoCoMo JSON at `eval/datasets/locomo/locomo10.json`. The copy currently
included in this repository contains the first two conversations from the official
ten-conversation dataset, which keeps the local baseline manageable. Start with one
conversation and optionally cap its questions for a quick pipeline check:

```bash
python eval/harness.py --limit-conversations 1 --max-questions 5
python eval/harness.py --limit-conversations 1
```

`--limit-conversations 1` means one complete conversation, including all of its
sessions. `--max-questions 5` limits only the evaluation questions. Run every
conversation present in the local file with:

```bash
python eval/harness.py
```

Each run prints accuracy, average tokens per query, and p50 latency together. It
overwrites `eval/results/phase1.json` with the latest metrics, per-question details,
and reproducibility note. Malformed extractor responses are appended separately to
the gitignored `eval/results/extraction_failures.jsonl` diagnostic log.

### Phase 1 local baseline

The checked-in `phase1.json` was produced from the two-conversation subset using
`qwen3-4b-instruct-2507-mlx` (Qwen 3 4B) for memory extraction and answering,
`gemma-4-12b-it-mlx` (Gemma 4 12B) as the judge,
`nvidia/nemotron-3-embed-1b` for embeddings, semantic top-15 retrieval,
extractor `v1-phase1`, and judge prompt `v1`.

| Category | Questions | Accuracy | Avg tokens/query | P50 latency |
| --- | ---: | ---: | ---: | ---: |
| Single-hop | 43 | 44.19% | 665.4 | 1801 ms |
| Temporal | 63 | 4.76% | 621.4 | 1513 ms |
| Multi-hop | 13 | 23.08% | 632.9 | 1598 ms |
| Open-domain | 114 | 69.30% | 613.8 | 1525 ms |
| Adversarial | 71 | 14.08% | 579.9 | 1359 ms |
| **Overall** | **304** | **37.50%** | **615.6** | **1501 ms** |

All 304 questions completed and were judged with no recorded errors. This is a
Qwen/Gemma learning baseline.
