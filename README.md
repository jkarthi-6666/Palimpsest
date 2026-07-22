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

The real `.env` is gitignored. Never commit `NVIDIA_API_KEY`. Both LLM inference
and embeddings use NVIDIA's OpenAI-compatible remote API. The embedding wrapper
supports `passage` mode for indexed facts and `query` mode for user questions. A
successful test finishes with `PHASE 0 OK`.

Docker publishes Palimpsest's Postgres on host port `5433`, leaving the usual
local Postgres port `5432` available for other projects.

Postgres is the source of truth: it owns the append-only event log and the
bitemporal memory records. Qdrant is only a replaceable search index; later phases
will store vectors there with payload IDs that point back to authoritative Postgres
rows. This keeps semantic search fast without allowing the vector store to become a
second, conflicting source of truth.

Storage remains entirely local in Docker. LLM and embedding inference use the
remote NVIDIA API at `NVIDIA_BASE_URL`; no local model service is part of
Palimpsest.
