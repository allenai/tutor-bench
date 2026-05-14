"""Tier-A parity: tutor-bench parser must match ai2-synthetic-annotations.

The fixture pair (``raw_detections.json`` + ``expected_detections.json``)
captures the source repo's :func:`parse_detection_results` behavior at the
pinned SHA. This test runs the same raw input through the ported
``_parse_detection_results`` and asserts the per-conversation detection
lists, suggested_cut_turn clamping, annotation_type fallback, usage
accumulation, and per-entry error reporting all match.

To regenerate the snapshot:

1. Check out ``ai2-synthetic-annotations`` at the SHA recorded in the
   ``source_sha`` field of ``expected_detections.json``.
2. Run ``annotator.core.detect.parse_detection_results`` on ``raw_detections.json``.
3. Write the result (in this snapshot's shape) into ``expected_detections.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tutor_bench.annotator.detect_key_moments import _parse_detection_results

pytestmark = pytest.mark.integration

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "annotator"


@pytest.fixture
def raw() -> dict:
    return json.loads((_FIXTURE_DIR / "raw_detections.json").read_text(encoding="utf-8"))


@pytest.fixture
def expected() -> dict:
    return json.loads((_FIXTURE_DIR / "expected_detections.json").read_text(encoding="utf-8"))


def _to_source_shape(
    moments_by_conv: dict, usage_by_conv: dict, all_conversations: set[str]
) -> dict[str, dict]:
    """Convert the tutor-bench typed output back into source's dict shape for diffing."""
    out: dict[str, dict] = {}
    for conv_id in all_conversations:
        detections = []
        for km in moments_by_conv.get(conv_id, []):
            d = asdict(km)
            d.pop("conversation_id", None)
            detections.append(d)
        usage = usage_by_conv.get(conv_id)
        out[conv_id] = {
            "detections": detections,
            "usage": {
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }
    return out


def _expected_conv_ids(raw: dict) -> set[str]:
    return {key.rpartition("__")[0] for key in raw}


def test_parser_matches_source_snapshot(raw: dict, expected: dict) -> None:
    moments_by_conv, usage_by_conv, errors = _parse_detection_results(raw)

    converted = _to_source_shape(moments_by_conv, usage_by_conv, _expected_conv_ids(raw))

    assert converted == expected["results"], (
        "tutor-bench parser output diverged from source snapshot.\n"
        f"Got: {json.dumps(converted, indent=2, sort_keys=True)}\n"
        f"Expected: {json.dumps(expected['results'], indent=2, sort_keys=True)}"
    )


def test_parser_records_expected_error_keys(raw: dict, expected: dict) -> None:
    _, _, errors = _parse_detection_results(raw)
    error_keys = sorted(e.key for e in errors)
    assert error_keys == sorted(expected["expected_error_keys"])
