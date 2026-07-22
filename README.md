# Palimpsest

Palimpsest is a local, from-scratch LLM memory learning project. Phase 0 contains
only the storage, vector, remote LLM, and embedding infrastructure.

## Run Phase 0

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
`query` mode for user questions. A successful test finishes with `PHASE 0 OK`.

Docker publishes Palimpsest's Postgres on host port `5433`, leaving the usual
local Postgres port `5432` available for other projects.

Postgres is the source of truth: it owns the append-only event log and the
bitemporal memory records. Qdrant is only a replaceable search index; later phases
will store vectors there with payload IDs that point back to authoritative Postgres
rows. This keeps semantic search fast without allowing the vector store to become a
second, conflicting source of truth.

Storage remains entirely local in Docker. NVIDIA provides embeddings. Chat
completion can use NVIDIA or a local OpenAI-compatible server selected entirely
through `.env`.

### Use a local LM Studio chat model

Start LM Studio's local server, load a model, and confirm its identifier:

```bash
curl -s http://127.0.0.1:1234/v1/models | python -m json.tool
```

Then set these values in `.env`, copying the exact model `id` returned above:

```dotenv
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=copy-the-loaded-model-id-here
```

`LLM_API_KEY` is a non-secret placeholder required by the OpenAI client; LM Studio
does not validate it by default. Keep `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, and
`EMBED_MODEL` configured because embeddings still use NVIDIA. To switch chat back
to NVIDIA, set `LLM_BASE_URL=${NVIDIA_BASE_URL}`,
`LLM_API_KEY=${NVIDIA_API_KEY}`, and choose the NVIDIA model in `LLM_MODEL`.

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
`gemma-4-12b-it-mlx` for extraction, answering, and judging,
`nvidia/nemotron-3-embed-1b` for embeddings, semantic top-15 retrieval, extractor
`v1-phase1`, and judge prompt `v1`.

| Category | Questions | Accuracy | Avg tokens/query | P50 latency |
| --- | ---: | ---: | ---: | ---: |
| Single-hop | 43 | 37.21% | 393.1 | 2647 ms |
| Temporal | 63 | 9.52% | 378.5 | 2485 ms |
| Multi-hop | 13 | 7.69% | 391.5 | 2518 ms |
| Open-domain | 114 | 40.35% | 386.9 | 2533 ms |
| Adversarial | 71 | 4.23% | 385.7 | 2494 ms |
| **Overall** | **304** | **23.68%** | **386.0** | **2503 ms** |

All 304 questions completed and were judged with no recorded errors. This is a
local-Gemma learning baseline, not a score directly comparable to a run judged by
the originally proposed Llama 3.3 70B model.
