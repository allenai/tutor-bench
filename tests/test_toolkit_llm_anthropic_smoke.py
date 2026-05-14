"""Opt-in integration smoke test for the Anthropic provider.

Skipped unless ``ANTHROPIC_API_KEY`` is set in the environment. Spends a tiny
amount of money (single Haiku call, max 64 output tokens) to confirm the
plumbing is wired correctly end-to-end. Run with:

    pytest -m integration tests/test_toolkit_llm_anthropic_smoke.py
"""

from __future__ import annotations

import os

import pytest

from tutor_bench.toolkit.cost_tracker import TokenUsage
from tutor_bench.toolkit.llm.anthropic import AnthropicClient
from tutor_bench.toolkit.llm.llm_client import ModelResponse

# Cheap model used only to confirm the client wiring. Real annotator runs
# use the model configured in configs/annotator.yaml — not this one.
_SMOKE_MODEL = "claude-haiku-4-5"

pytestmark = pytest.mark.integration


@pytest.fixture
def anthropic_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set; skipping Anthropic smoke test.")
    return key


def test_anthropic_client_generate_returns_response_with_usage(anthropic_api_key: str) -> None:
    client = AnthropicClient(_SMOKE_MODEL, api_key=anthropic_api_key)
    response = client.generate(
        "Return the JSON object {\"ok\": true} and nothing else.",
        json_mode=True,
        max_tokens=64,
        timeout_sec=30,
    )

    assert isinstance(response, ModelResponse)
    assert response.text, "expected non-empty response text"
    assert isinstance(response.usage, TokenUsage)
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
