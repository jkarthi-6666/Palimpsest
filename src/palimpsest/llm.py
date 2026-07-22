import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

_client: OpenAI | None = None


@dataclass(frozen=True)
class CompletionResult:
    text: str
    tokens: int


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or value == "replace-with-your-nvidia-api-key":
        raise RuntimeError(f"{name} is not configured; set it in the gitignored .env file")
    return value


def _llm_config() -> tuple[str, str]:
    custom_base_url = os.getenv("LLM_BASE_URL")
    if custom_base_url:
        # Never forward the NVIDIA secret to a custom server by implicit fallback.
        return custom_base_url, _required_env("LLM_API_KEY")
    return _required_env("NVIDIA_BASE_URL"), _required_env("NVIDIA_API_KEY")


def _connect() -> OpenAI:
    global _client
    if _client is None:
        base_url, api_key = _llm_config()
        _client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=60.0,
            max_retries=1,
        )
    return _client


def complete_with_usage(
    prompt: str,
    *,
    max_tokens: int = 512,
) -> CompletionResult:
    response = _connect().chat.completions.create(
        model=_required_env("LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("the configured LLM returned an empty completion")
    tokens = response.usage.total_tokens if response.usage else 0
    return CompletionResult(text=content, tokens=tokens)


def complete(
    prompt: str,
    *,
    max_tokens: int = 512,
) -> str:
    return complete_with_usage(prompt, max_tokens=max_tokens).text
