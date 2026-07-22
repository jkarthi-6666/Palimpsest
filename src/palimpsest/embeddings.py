import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

_client: OpenAI | None = None
_dimension: int | None = None


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or value == "replace-with-your-nvidia-api-key":
        raise RuntimeError(f"{name} is not configured; set it in the gitignored .env file")
    return value


def _request(texts: list[str], input_type: str) -> list[list[float]]:
    if input_type not in {"passage", "query"}:
        raise ValueError("input_type must be 'passage' or 'query'")
    if _client is None:
        raise RuntimeError("embedding client has not been initialized")

    response = _client.embeddings.create(
        model=_required_env("EMBED_MODEL"),
        input=texts,
        encoding_format="float",
        # NVIDIA uses asymmetric instructions so indexed facts and search
        # questions occupy compatible sides of the retrieval embedding space.
        extra_body={"input_type": input_type},
    )
    return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]


def _load() -> None:
    global _client, _dimension
    if _client is not None:
        return

    _client = OpenAI(
        base_url=_required_env("NVIDIA_BASE_URL"),
        api_key=_required_env("NVIDIA_API_KEY"),
        timeout=60.0,
        max_retries=1,
    )
    probe = _request(["Palimpsest embedding dimension probe."], "passage")
    if not probe or not probe[0]:
        raise RuntimeError("NVIDIA returned an empty probe embedding")
    _dimension = len(probe[0])
    print(f"Loaded NVIDIA embedding model {_required_env('EMBED_MODEL')} (dimension {_dimension})")


def embedding_dimension() -> int:
    _load()
    assert _dimension is not None
    return _dimension


def embed(texts: list[str], input_type: str) -> list[list[float]]:
    _load()
    if not texts:
        return []

    vectors = _request(texts, input_type)
    if len(vectors) != len(texts):
        raise RuntimeError("NVIDIA returned a different number of embeddings than requested")
    if any(len(vector) != _dimension for vector in vectors):
        raise RuntimeError("NVIDIA returned an inconsistent embedding dimension")
    return vectors
