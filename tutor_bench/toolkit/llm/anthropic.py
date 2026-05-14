"""Concrete Anthropic ``ModelClient`` plus a batch-API runner.

The SDK is imported lazily so projects that never call into Anthropic do not
pay the import cost (or require the dependency at install time, eventually).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from typing import Any, ClassVar

from tutor_bench.toolkit.cost_tracker import TokenUsage
from tutor_bench.toolkit.llm.llm_client import (
    MAX_OUTPUT_TOKENS,
    ModelResponse,
    _extract_entry,
    strip_json_fences,
)

logger = logging.getLogger(__name__)


_JSON_SYSTEM = (
    "You must respond with valid JSON only. "
    "Do not include markdown code fences, explanations, or any text "
    "outside the JSON object."
)


class AnthropicClient:
    """Synchronous Anthropic client implementing the ``ModelClient`` protocol."""

    provider: ClassVar[str] = "anthropic"

    def __init__(self, model: str, *, api_key: str | None = None):
        self.model = model
        key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing — set the env var or pass api_key=..."
            )
        import anthropic  # lazy

        self._client = anthropic.Anthropic(api_key=key)

    def generate(
        self,
        prompt: str,
        *,
        json_mode: bool = True,
        max_tokens: int = 0,
        timeout_sec: int = 120,
        thinking: bool = False,
        thinking_budget: int = 0,
    ) -> ModelResponse:
        if max_tokens <= 0:
            max_tokens = MAX_OUTPUT_TOKENS["anthropic"]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": timeout_sec,
        }
        if json_mode:
            kwargs["system"] = _JSON_SYSTEM
        if thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget if thinking_budget > 0 else 16384,
            }

        response = self._client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break
        if json_mode:
            text = strip_json_fences(text)

        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return ModelResponse(text=text, usage=usage)

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self.model!r})"


def run_batch_anthropic(
    client: AnthropicClient,
    entries: list[Mapping[str, Any]],
    *,
    json_mode: bool,
    display_name: str,
    poll_interval_sec: int,
    batch_timeout_sec: int,
    thinking: bool,
    thinking_budget: int,
    max_retries: int,
    base_delay_sec: int,
) -> dict[str, dict[str, Any]]:
    """Submit a Message Batch, poll until completion, and parse results.

    Anthropic's ``custom_id`` field is limited to 64 chars, so we use short
    indexed IDs (``r0``, ``r1``, ...) on the wire and map results back to the
    caller's original keys.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    anthropic_client = client._client
    max_tokens_cap = MAX_OUTPUT_TOKENS["anthropic"]

    id_to_key: dict[str, str] = {}
    requests: list[Any] = []
    for i, entry in enumerate(entries):
        key, prompt_text, entry_json_mode, entry_max_tokens = _extract_entry(entry)
        if not entry_max_tokens or entry_max_tokens > max_tokens_cap:
            entry_max_tokens = max_tokens_cap

        short_id = f"r{i}"
        id_to_key[short_id] = key

        params: dict[str, Any] = {
            "model": client.model,
            "max_tokens": entry_max_tokens,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        if json_mode and entry_json_mode:
            params["system"] = _JSON_SYSTEM
        if thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget if thinking_budget > 0 else 16384,
            }

        requests.append(
            Request(
                custom_id=short_id,
                params=MessageCreateParamsNonStreaming(**params),
            )
        )

    message_batch = _submit_batch_with_retry(
        anthropic_client, requests, max_retries=max_retries, base_delay_sec=base_delay_sec
    )
    logger.info("anthropic batch submitted: id=%s requests=%d", message_batch.id, len(requests))

    message_batch = _poll_until_done(
        anthropic_client,
        message_batch,
        poll_interval_sec=poll_interval_sec,
        batch_timeout_sec=batch_timeout_sec,
    )
    logger.info(
        "anthropic batch finished: status=%s counts=%s",
        message_batch.processing_status,
        message_batch.request_counts,
    )

    results: dict[str, dict[str, Any]] = {}
    for result in anthropic_client.messages.batches.results(message_batch.id):
        custom_id = result.custom_id or ""
        key = id_to_key.get(custom_id, custom_id)
        if result.result.type == "succeeded":
            message = result.result.message
            text = ""
            for block in message.content:
                if block.type == "text":
                    text = block.text
                    break
            if json_mode:
                text = strip_json_fences(text)
            input_tokens = message.usage.input_tokens or 0
            output_tokens = message.usage.output_tokens or 0
            results[key] = {
                "text": text,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        else:
            err = getattr(result.result, "error", None)
            results[key] = {
                "text": "",
                "error": f"{result.result.type}{f': {err}' if err else ''}",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
    return results


def _submit_batch_with_retry(
    anthropic_client: Any,
    requests: list[Any],
    *,
    max_retries: int,
    base_delay_sec: int,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return anthropic_client.messages.batches.create(requests=requests)
        except Exception as exc:  # noqa: BLE001 — retry covers SDK + network classes
            last_error = exc
            delay = base_delay_sec * (2**attempt)
            if attempt < max_retries - 1:
                logger.warning(
                    "anthropic batch submit failed (attempt %d/%d): %s; retrying in %ds",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
    raise RuntimeError(f"anthropic batch submit failed after {max_retries} attempts: {last_error}")


def _poll_until_done(
    anthropic_client: Any,
    message_batch: Any,
    *,
    poll_interval_sec: int,
    batch_timeout_sec: int,
) -> Any:
    poll_start = time.monotonic()
    while message_batch.processing_status != "ended":
        elapsed = time.monotonic() - poll_start
        if elapsed > batch_timeout_sec:
            raise RuntimeError(
                f"anthropic batch timed out after {batch_timeout_sec}s "
                f"(status={message_batch.processing_status})"
            )
        logger.info(
            "anthropic batch polling: status=%s elapsed=%.0fs",
            message_batch.processing_status,
            elapsed,
        )
        time.sleep(poll_interval_sec)
        message_batch = anthropic_client.messages.batches.retrieve(message_batch.id)
    return message_batch
