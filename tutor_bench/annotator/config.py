"""Annotator configuration: typed dataclasses + loader.

Loads runtime configuration for the annotator from two data files at the
top of the repo:

- ``configs/annotator.yaml`` — model choice and model parameters
- ``configs/llm_pricing.json`` — per-provider:model token pricing

The function :func:`load_annotator_config` reads both and returns an
:class:`AnnotatorConfig` (a frozen dataclass) that the annotator passes to
its phase functions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = _REPO_ROOT_DEFAULT / "configs" / "annotator.yaml"
_DEFAULT_PRICING = _REPO_ROOT_DEFAULT / "configs" / "llm_pricing.json"


@dataclass(frozen=True)
class ModelConfig:
    """Provider/model selection plus model-call parameters.

    ``provider`` is intentionally a plain string (not a Literal) so the YAML
    schema is forward-compatible with providers added in later plans. The
    pilot only routes ``provider=anthropic`` to a working client; others
    raise :class:`NotImplementedError` when a client is built.
    """

    provider: str
    model: str
    max_tokens: int
    thinking: bool = False
    thinking_budget: int = 0
    reasoning_effort: str | None = None
    batch_poll_interval_sec: int = 60


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 5
    base_delay_sec: int = 5


@dataclass(frozen=True)
class AnnotatorConfig:
    model: ModelConfig
    pricing: Mapping[str, Mapping[str, float]]
    retry: RetryConfig = field(default_factory=RetryConfig)
    batch_timeout_sec: int = 86400


def _require(mapping: Mapping[str, Any], key: str, source: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{source} is missing required key {key!r}")
    return mapping[key]


def _parse_model(raw: Mapping[str, Any]) -> ModelConfig:
    return ModelConfig(
        provider=str(_require(raw, "provider", "model")),
        model=str(_require(raw, "model", "model")),
        max_tokens=int(_require(raw, "max_tokens", "model")),
        thinking=bool(raw.get("thinking", False)),
        thinking_budget=int(raw.get("thinking_budget", 0)),
        reasoning_effort=raw.get("reasoning_effort"),
        batch_poll_interval_sec=int(raw.get("batch_poll_interval_sec", 60)),
    )


def _parse_retry(raw: Mapping[str, Any] | None) -> RetryConfig:
    if not raw:
        return RetryConfig()
    return RetryConfig(
        max_retries=int(raw.get("max_retries", 5)),
        base_delay_sec=int(raw.get("base_delay_sec", 5)),
    )


def load_annotator_config(
    *,
    yaml_path: Path | None = None,
    pricing_path: Path | None = None,
) -> AnnotatorConfig:
    """Load and validate annotator runtime configuration.

    ``yaml_path`` defaults to ``<repo_root>/configs/annotator.yaml``.
    ``pricing_path`` defaults to ``<repo_root>/configs/llm_pricing.json``.
    """
    yaml_path = yaml_path or _DEFAULT_YAML
    pricing_path = pricing_path or _DEFAULT_PRICING

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    model_block = _require(raw, "model", str(yaml_path))
    if not isinstance(model_block, Mapping):
        raise ValueError(f"{yaml_path}: 'model' must be a mapping")
    model = _parse_model(model_block)

    retry = _parse_retry(raw.get("retry"))
    batch_timeout_sec = int(raw.get("batch_timeout_sec", 86400))

    with open(pricing_path, encoding="utf-8") as f:
        pricing = json.load(f)
    if not isinstance(pricing, dict):
        raise ValueError(f"{pricing_path}: expected a JSON object at the top level")

    return AnnotatorConfig(
        model=model,
        pricing=pricing,
        retry=retry,
        batch_timeout_sec=batch_timeout_sec,
    )
