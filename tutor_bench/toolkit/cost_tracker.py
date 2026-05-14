"""Accumulate token usage and dollar cost across many LLM calls in one run.

``TokenUsage`` and ``CostBreakdown`` are the canonical shapes for reporting
LLM consumption from any annotator phase. ``CostTracker`` is the per-run
accumulator that converts (provider, model, usage) records into both.

Pricing is supplied at construction (typically loaded from
``configs/llm_pricing.json``). Keys are ``"{provider}:{model}"``; values are
``{"input_per_1m": float, "output_per_1m": float}``. Unknown keys are
recorded as zero cost with a single warning per call.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    """Token counts returned by an LLM call (or aggregated across many)."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class CostBreakdown:
    """USD cost split into input + output, rounded to 8 decimals."""

    input_usd: float = 0.0
    output_usd: float = 0.0
    total_usd: float = 0.0


def _normalize_usage(usage: TokenUsage | Mapping[str, int]) -> TokenUsage:
    if isinstance(usage, TokenUsage):
        return usage
    input_t = int(usage.get("input_tokens", 0))
    output_t = int(usage.get("output_tokens", 0))
    total_t = int(usage.get("total_tokens", input_t + output_t))
    return TokenUsage(input_tokens=input_t, output_tokens=output_t, total_tokens=total_t)


class CostTracker:
    """Accumulate usage and cost across many LLM calls for a single run."""

    def __init__(self, pricing: Mapping[str, Mapping[str, float]]):
        self._pricing = pricing
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._input_cost = 0.0
        self._output_cost = 0.0

    def record(
        self,
        provider: str,
        model: str,
        usage: TokenUsage | Mapping[str, int],
    ) -> None:
        u = _normalize_usage(usage)
        self._input_tokens += u.input_tokens
        self._output_tokens += u.output_tokens
        self._total_tokens += u.total_tokens

        key = f"{provider}:{model}"
        rate = self._pricing.get(key)
        if rate is None:
            logger.warning("No pricing entry for %s; recording $0 for this call.", key)
            return
        input_per_1m = float(rate.get("input_per_1m", 0.0))
        output_per_1m = float(rate.get("output_per_1m", 0.0))
        self._input_cost += u.input_tokens / 1_000_000 * input_per_1m
        self._output_cost += u.output_tokens / 1_000_000 * output_per_1m

    def usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._total_tokens,
        )

    def cost(self) -> CostBreakdown:
        input_usd = round(self._input_cost, 8)
        output_usd = round(self._output_cost, 8)
        return CostBreakdown(
            input_usd=input_usd,
            output_usd=output_usd,
            total_usd=round(input_usd + output_usd, 8),
        )
