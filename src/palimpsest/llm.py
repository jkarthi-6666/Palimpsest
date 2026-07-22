import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_client: OpenAI | None = None


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or value == "replace-with-your-nvidia-api-key":
        raise RuntimeError(f"{name} is not configured; set it in the gitignored .env file")
    return value


def _connect() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=_required_env("NVIDIA_BASE_URL"),
            api_key=_required_env("NVIDIA_API_KEY"),
        )
    return _client


def complete(prompt: str) -> str:
    response = _connect().chat.completions.create(
        model=_required_env("LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("NVIDIA returned an empty completion")
    return content
