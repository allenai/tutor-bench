from __future__ import annotations

import logging

import pytest

from tutor_bench.toolkit.cost_tracker import CostBreakdown, CostTracker, TokenUsage


@pytest.fixture
def pricing() -> dict[str, dict[str, float]]:
    return {
        "anthropic:claude-opus-4-6": {"input_per_1m": 15.0, "output_per_1m": 75.0},
        "openai:gpt-5.4": {"input_per_1m": 5.0, "output_per_1m": 20.0},
    }


def test_cost_tracker_accumulates_usage_across_multiple_calls(
    pricing: dict[str, dict[str, float]],
) -> None:
    tracker = CostTracker(pricing=pricing)

    tracker.record("anthropic", "claude-opus-4-6", {"input_tokens": 1_000_000, "output_tokens": 0})
    tracker.record("anthropic", "claude-opus-4-6", {"input_tokens": 500_000, "output_tokens": 200_000})

    usage = tracker.usage()
    assert usage.input_tokens == 1_500_000
    assert usage.output_tokens == 200_000


def test_cost_tracker_applies_pricing_from_config(
    pricing: dict[str, dict[str, float]],
) -> None:
    tracker = CostTracker(pricing=pricing)

    # 1M input tokens @ $15 + 1M output tokens @ $75 = $90 total
    tracker.record(
        "anthropic",
        "claude-opus-4-6",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )

    cost = tracker.cost()
    assert cost.input_usd == pytest.approx(15.0)
    assert cost.output_usd == pytest.approx(75.0)
    assert cost.total_usd == pytest.approx(90.0)


def test_cost_tracker_records_zero_cost_and_warns_for_unknown_model(
    pricing: dict[str, dict[str, float]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracker = CostTracker(pricing=pricing)

    with caplog.at_level(logging.WARNING):
        tracker.record(
            "anthropic",
            "claude-opus-99-future",
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )

    cost = tracker.cost()
    assert cost.total_usd == 0.0
    assert any("anthropic:claude-opus-99-future" in record.message for record in caplog.records)


def test_cost_tracker_accepts_token_usage_dataclass(
    pricing: dict[str, dict[str, float]],
) -> None:
    tracker = CostTracker(pricing=pricing)

    tracker.record(
        "anthropic",
        "claude-opus-4-6",
        TokenUsage(input_tokens=2_000_000, output_tokens=500_000, total_tokens=2_500_000),
    )

    usage = tracker.usage()
    assert usage.input_tokens == 2_000_000
    assert usage.output_tokens == 500_000
    assert usage.total_tokens == 2_500_000

    cost = tracker.cost()
    assert isinstance(cost, CostBreakdown)
    assert cost.input_usd == pytest.approx(30.0)  # 2M * $15/1M
    assert cost.output_usd == pytest.approx(37.5)  # 500k * $75/1M
