import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

_clients: dict[str, OpenAI] = {}


@dataclass(frozen=True)
class CompletionResult:
    text: str
    tokens: int


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or value == "replace-with-your-nvidia-api-key":
        raise RuntimeError(f"{name} is not configured; set it in the gitignored .env file")
    return value


def _llm_config(prefix: str = "LLM") -> tuple[str, str]:
    custom_base_url = os.getenv(f"{prefix}_BASE_URL")
    if custom_base_url:
        # Never forward the NVIDIA secret to a custom server by implicit fallback.
        return custom_base_url, _required_env(f"{prefix}_API_KEY")
    return _required_env("NVIDIA_BASE_URL"), _required_env("NVIDIA_API_KEY")


def _connect(prefix: str = "LLM") -> OpenAI:
    if prefix not in _clients:
        base_url, api_key = _llm_config(prefix)
        _clients[prefix] = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=60.0,
            max_retries=1,
        )
    return _clients[prefix]


def complete_with_usage(
    prompt: str,
    *,
    max_tokens: int = 512,
    config_prefix: str = "LLM",
) -> CompletionResult:
    response = _connect(config_prefix).chat.completions.create(
        model=_required_env(f"{config_prefix}_MODEL"),
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
    config_prefix: str = "LLM",
) -> str:
    return complete_with_usage(
        prompt,
        max_tokens=max_tokens,
        config_prefix=config_prefix,
    ).text
