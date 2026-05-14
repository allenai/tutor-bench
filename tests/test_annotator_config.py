from __future__ import annotations

import json
from pathlib import Path

import pytest

from tutor_bench.annotator.config import (
    AnnotatorConfig,
    ModelConfig,
    RetryConfig,
    load_annotator_config,
)


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _write_pricing(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "anthropic:claude-opus-4-6": {"input_per_1m": 15.0, "output_per_1m": 75.0},
            }
        ),
        encoding="utf-8",
    )


def test_load_annotator_config_returns_typed_dataclass(tmp_path: Path) -> None:
    yaml_path = tmp_path / "annotator.yaml"
    pricing_path = tmp_path / "llm_pricing.json"
    _write_yaml(
        yaml_path,
        """
model:
  provider: anthropic
  model: claude-opus-4-6
  max_tokens: 128000
  thinking: true
  thinking_budget: 16384
  batch_poll_interval_sec: 60
retry:
  max_retries: 5
  base_delay_sec: 5
batch_timeout_sec: 86400
""",
    )
    _write_pricing(pricing_path)

    cfg = load_annotator_config(yaml_path=yaml_path, pricing_path=pricing_path)

    assert isinstance(cfg, AnnotatorConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.retry, RetryConfig)
    assert cfg.model.provider == "anthropic"
    assert cfg.model.model == "claude-opus-4-6"
    assert cfg.model.max_tokens == 128000
    assert cfg.model.thinking is True
    assert cfg.model.thinking_budget == 16384
    assert cfg.model.batch_poll_interval_sec == 60
    assert cfg.retry.max_retries == 5
    assert cfg.retry.base_delay_sec == 5
    assert cfg.batch_timeout_sec == 86400


def test_load_annotator_config_loads_pricing_from_json(tmp_path: Path) -> None:
    yaml_path = tmp_path / "annotator.yaml"
    pricing_path = tmp_path / "llm_pricing.json"
    _write_yaml(
        yaml_path,
        """
model:
  provider: anthropic
  model: claude-opus-4-6
  max_tokens: 1000
""",
    )
    _write_pricing(pricing_path)

    cfg = load_annotator_config(yaml_path=yaml_path, pricing_path=pricing_path)

    assert "anthropic:claude-opus-4-6" in cfg.pricing
    assert cfg.pricing["anthropic:claude-opus-4-6"]["input_per_1m"] == 15.0


def test_load_annotator_config_applies_defaults_for_optional_fields(tmp_path: Path) -> None:
    yaml_path = tmp_path / "annotator.yaml"
    pricing_path = tmp_path / "llm_pricing.json"
    _write_yaml(
        yaml_path,
        """
model:
  provider: anthropic
  model: claude-opus-4-6
  max_tokens: 1000
""",
    )
    _write_pricing(pricing_path)

    cfg = load_annotator_config(yaml_path=yaml_path, pricing_path=pricing_path)

    # Defaults from the dataclasses fill in
    assert cfg.model.thinking is False
    assert cfg.model.thinking_budget == 0
    assert cfg.retry.max_retries == 5
    assert cfg.retry.base_delay_sec == 5
    assert cfg.batch_timeout_sec == 86400


def test_load_annotator_config_raises_when_model_block_missing(tmp_path: Path) -> None:
    yaml_path = tmp_path / "annotator.yaml"
    pricing_path = tmp_path / "llm_pricing.json"
    _write_yaml(yaml_path, "retry:\n  max_retries: 1\n")
    _write_pricing(pricing_path)

    with pytest.raises(ValueError, match="model"):
        load_annotator_config(yaml_path=yaml_path, pricing_path=pricing_path)


def test_load_annotator_config_raises_when_required_model_keys_missing(tmp_path: Path) -> None:
    yaml_path = tmp_path / "annotator.yaml"
    pricing_path = tmp_path / "llm_pricing.json"
    _write_yaml(
        yaml_path,
        """
model:
  provider: anthropic
  # missing model + max_tokens
""",
    )
    _write_pricing(pricing_path)

    with pytest.raises(ValueError):
        load_annotator_config(yaml_path=yaml_path, pricing_path=pricing_path)
