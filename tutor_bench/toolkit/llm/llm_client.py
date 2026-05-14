"""Provider-agnostic LLM client protocol and shared batch utilities.

A :class:`ModelClient` is anything with ``model``, ``provider``, and a
synchronous ``generate(prompt, ...) -> ModelResponse`` method. Concrete
implementations live alongside this file (e.g. ``llm.anthropic``).

The pilot ports only the Anthropic path. ``infer_provider`` still recognizes
OpenAI/Gemini prefixes so future provider modules slot in without changing
the routing table.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from tutor_bench.toolkit.cost_tracker import TokenUsage

logger = logging.getLogger(__name__)


PROVIDER_PREFIXES: list[tuple[str, str]] = [
    ("gemini", "gemini"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
]

MAX_OUTPUT_TOKENS: Mapping[str, int] = {
    "gemini": 65536,
    "openai": 128000,
    "anthropic": 128000,
}


def infer_provider(model: str) -> str:
    """Return the provider name implied by a model string.

    Example::

        >>> infer_provider("claude-opus-4-6")
        'anthropic'
    """
    lower = model.lower()
    for prefix, provider in PROVIDER_PREFIXES:
        if lower.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot infer provider for model {model!r}. "
        f"Known prefixes: {', '.join(prefix for prefix, _ in PROVIDER_PREFIXES)}."
    )


@dataclass(frozen=True)
class ModelResponse:
    """One unified response shape across providers."""

    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)


@runtime_checkable
class ModelClient(Protocol):
    """Synchronous LLM client interface implemented by per-provider modules."""

    model: str
    provider: ClassVar[str]

    def generate(
        self,
        prompt: str,
        *,
        json_mode: bool = True,
        max_tokens: int = 0,
        timeout_sec: int = 120,
        thinking: bool = False,
        thinking_budget: int = 0,
    ) -> ModelResponse: ...


# ---------------------------------------------------------------------------
# Batch-entry helpers (provider-neutral internal format)
# ---------------------------------------------------------------------------

def build_batch_entry(
    key: str,
    prompt_text: str,
    *,
    json_mode: bool = True,
    max_tokens: int = 65536,
) -> dict[str, Any]:
    """Build a single provider-neutral batch entry."""
    gen_config: dict[str, Any] = {"max_output_tokens": max_tokens}
    if json_mode:
        gen_config["response_mime_type"] = "application/json"
    return {
        "key": key,
        "request": {
            "contents": [{"parts": [{"text": prompt_text}], "role": "user"}],
            "generation_config": gen_config,
        },
    }


def _extract_entry(entry: Mapping[str, Any]) -> tuple[str, str, bool, int]:
    """Pull (key, prompt, json_mode, max_tokens) out of a batch entry."""
    key = entry["key"]
    parts = entry["request"]["contents"][0]["parts"]
    prompt_text = parts[0]["text"]
    gen_config = entry["request"].get("generation_config", {})
    json_mode = "application/json" in gen_config.get("response_mime_type", "")
    max_tokens = gen_config.get("max_output_tokens", 0)
    return key, prompt_text, json_mode, max_tokens


_FENCE_HEAD = re.compile(r"^```(?:json)?\s*\n?")
_FENCE_TAIL = re.compile(r"\n?```\s*$")


def strip_json_fences(text: str) -> str:
    """Strip `````json ... ````` (or bare ```````) wrappers some models emit."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_HEAD.sub("", stripped)
        stripped = _FENCE_TAIL.sub("", stripped)
    return stripped.strip()


# ---------------------------------------------------------------------------
# Synchronous-mode runner (one entry at a time)
# ---------------------------------------------------------------------------

def run_sync_entries(
    client: ModelClient,
    entries: Iterable[Mapping[str, Any]],
    *,
    json_mode: bool = True,
    thinking: bool = False,
    thinking_budget: int = 0,
) -> dict[str, dict[str, Any]]:
    """Run ``entries`` one at a time against ``client``; return ``{key: {text, usage}}``."""
    results: dict[str, dict[str, Any]] = {}
    entries_list = list(entries)
    total = len(entries_list)
    for i, entry in enumerate(entries_list, start=1):
        key, prompt_text, entry_json_mode, entry_max_tokens = _extract_entry(entry)
        logger.info("sync entry %d/%d: %s", i, total, key[:60])
        try:
            response = client.generate(
                prompt_text,
                json_mode=entry_json_mode if json_mode else False,
                max_tokens=entry_max_tokens,
                thinking=thinking,
                thinking_budget=thinking_budget,
            )
            results[key] = {
                "text": response.text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as exc:  # noqa: BLE001 — caller wants per-entry errors, not aborts
            logger.exception("sync entry %s failed", key)
            results[key] = {
                "text": "",
                "error": str(exc),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
    return results


# ---------------------------------------------------------------------------
# Batch dispatcher
# ---------------------------------------------------------------------------

def run_batch(
    client: ModelClient,
    entries: Iterable[Mapping[str, Any]],
    *,
    json_mode: bool = True,
    display_name: str = "batch",
    poll_interval_sec: int = 60,
    batch_timeout_sec: int = 86400,
    thinking: bool = False,
    thinking_budget: int = 0,
    max_retries: int = 5,
    base_delay_sec: int = 5,
) -> dict[str, dict[str, Any]]:
    """Dispatch a batch job to the provider implied by ``client.provider``.

    The pilot only implements the Anthropic path. Other providers raise
    :class:`NotImplementedError` so a future plan can slot them in without
    touching the routing.
    """
    provider = client.provider
    if provider == "anthropic":
        from tutor_bench.toolkit.llm.anthropic import AnthropicClient, run_batch_anthropic

        if not isinstance(client, AnthropicClient):
            raise TypeError(
                f"Expected AnthropicClient when provider='anthropic', got {type(client).__name__}"
            )
        return run_batch_anthropic(
            client,
            list(entries),
            json_mode=json_mode,
            display_name=display_name,
            poll_interval_sec=poll_interval_sec,
            batch_timeout_sec=batch_timeout_sec,
            thinking=thinking,
            thinking_budget=thinking_budget,
            max_retries=max_retries,
            base_delay_sec=base_delay_sec,
        )
    raise NotImplementedError(
        f"Batch mode not implemented for provider {provider!r} in this pilot. "
        "Add a provider module under tutor_bench/toolkit/llm/ to extend support."
    )
