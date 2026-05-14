from __future__ import annotations

import json
from typing import Any, ClassVar

from tutor_bench.annotator.detect_key_moments import (
    KeyMoment,
    PhaseResult,
    _build_detection_entries,
    _parse_detection_results,
    detect_key_moments,
)
from tutor_bench.toolkit.cost_tracker import TokenUsage
from tutor_bench.toolkit.llm.llm_client import ModelResponse


# ---------------------------------------------------------------------------
# Parser tests — load-bearing detect.py:81-139 logic
# ---------------------------------------------------------------------------


def _raw(key: str, detections: list[dict[str, Any]], usage: dict[str, int] | None = None) -> dict:
    return {
        key: {
            "text": json.dumps({"detections": detections}),
            "usage": usage or {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    }


class TestParseDetectionResults:
    def test_clamps_suggested_cut_turn_below_lower_bound(self) -> None:
        raw = _raw(
            "conv1__scaffolding",
            [
                {
                    "annotation_type": "scaffolding",
                    "turn_start": 10,
                    "turn_end": 12,
                    "brief_description": "x",
                    "suggested_cut_turn": 0,  # below max(1, ts-2) = 8
                }
            ],
        )
        moments, _usage, _errors = _parse_detection_results(raw)
        assert moments["conv1"][0].suggested_cut_turn == 9  # falls back to max(1, ts-1)

    def test_clamps_suggested_cut_turn_above_turn_end(self) -> None:
        raw = _raw(
            "conv1__scaffolding",
            [
                {
                    "annotation_type": "scaffolding",
                    "turn_start": 10,
                    "turn_end": 12,
                    "brief_description": "x",
                    "suggested_cut_turn": 999,  # above turn_end
                }
            ],
        )
        moments, _usage, _errors = _parse_detection_results(raw)
        assert moments["conv1"][0].suggested_cut_turn == 9  # falls back to max(1, ts-1)

    def test_falls_back_to_key_annotation_type_when_missing(self) -> None:
        raw = _raw(
            "conv1__rapport",
            [
                {
                    # no annotation_type provided
                    "turn_start": 5,
                    "turn_end": 6,
                    "brief_description": "x",
                    "suggested_cut_turn": 4,
                }
            ],
        )
        moments, _, _ = _parse_detection_results(raw)
        assert moments["conv1"][0].annotation_type == "rapport"

    def test_falls_back_to_key_annotation_type_when_invalid(self) -> None:
        raw = _raw(
            "conv1__scaffolding",
            [
                {
                    "annotation_type": "completely_bogus",
                    "turn_start": 5,
                    "turn_end": 6,
                    "brief_description": "x",
                    "suggested_cut_turn": 4,
                }
            ],
        )
        moments, _, _ = _parse_detection_results(raw)
        assert moments["conv1"][0].annotation_type == "scaffolding"

    def test_records_parse_error_on_invalid_json(self) -> None:
        raw = {
            "conv1__scaffolding": {
                "text": "not json at all",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        }
        moments, _, errors = _parse_detection_results(raw)
        assert moments.get("conv1", []) == []
        assert len(errors) == 1
        assert errors[0].key == "conv1__scaffolding"

    def test_records_error_when_entry_has_error_field(self) -> None:
        raw = {
            "conv1__scaffolding": {
                "text": "",
                "error": "rate_limited",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        }
        moments, _, errors = _parse_detection_results(raw)
        assert moments.get("conv1", []) == []
        assert errors[0].error == "rate_limited"

    def test_accumulates_usage_across_targets_for_same_conv(self) -> None:
        raw = {
            "conv1__scaffolding": {
                "text": json.dumps({"detections": []}),
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            },
            "conv1__rapport": {
                "text": json.dumps({"detections": []}),
                "usage": {"input_tokens": 200, "output_tokens": 75, "total_tokens": 275},
            },
        }
        _, usage_by_conv, _ = _parse_detection_results(raw)
        assert usage_by_conv["conv1"].input_tokens == 300
        assert usage_by_conv["conv1"].output_tokens == 125
        assert usage_by_conv["conv1"].total_tokens == 425


# ---------------------------------------------------------------------------
# Entry-builder tests
# ---------------------------------------------------------------------------


def _transcript(conv_id: str, n_turns: int = 3) -> dict[str, Any]:
    return {
        "conversation_id": conv_id,
        "turns": [
            {"turn_number": i + 1, "role": "TUTOR" if i % 2 == 0 else "STUDENT", "text": f"line {i + 1}"}
            for i in range(n_turns)
        ],
    }


class TestBuildDetectionEntries:
    def test_one_entry_per_conversation_per_target(self) -> None:
        transcripts = [_transcript("c1"), _transcript("c2")]
        entries = _build_detection_entries(
            transcripts,
            targets=("scaffolding", "rapport"),
            prompt_cache={"scaffolding": "PROMPT_S {transcript}", "rapport": "PROMPT_R {transcript}"},
        )
        keys = {e["key"] for e in entries}
        assert keys == {"c1__scaffolding", "c1__rapport", "c2__scaffolding", "c2__rapport"}

    def test_substitutes_transcript_placeholder(self) -> None:
        entries = _build_detection_entries(
            [_transcript("c1", n_turns=2)],
            targets=("scaffolding",),
            prompt_cache={"scaffolding": "Analyze:\n{transcript}\nEnd."},
        )
        assert len(entries) == 1
        prompt = entries[0]["request"]["contents"][0]["parts"][0]["text"]
        assert prompt.startswith("Analyze:\nTurn 1. TUTOR: line 1\nTurn 2. STUDENT: line 2\nEnd.")


# ---------------------------------------------------------------------------
# End-to-end with a fake client (no real LLM)
# ---------------------------------------------------------------------------


class _FakeClient:
    """Returns the same canned JSON for every prompt."""

    provider: ClassVar[str] = "fake"
    model: str = "fake-test"

    def __init__(self, canned_text: str) -> None:
        self._canned_text = canned_text
        self.call_count = 0

    def generate(self, prompt: str, **_: Any) -> ModelResponse:
        self.call_count += 1
        return ModelResponse(
            text=self._canned_text,
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
        )


def test_detect_key_moments_end_to_end_with_fake_client(tmp_path) -> None:
    canned = json.dumps(
        {
            "detections": [
                {
                    "annotation_type": "scaffolding",
                    "turn_start": 2,
                    "turn_end": 3,
                    "brief_description": "Tutor introduces hint.",
                    "suggested_cut_turn": 1,
                }
            ]
        }
    )
    fake = _FakeClient(canned)

    result = detect_key_moments(
        [_transcript("c1", n_turns=5), _transcript("c2", n_turns=5)],
        targets=("scaffolding", "rapport"),
        batch=False,
        client=fake,
    )

    assert isinstance(result, PhaseResult)
    # 2 conversations × 2 targets = 4 generate() calls
    assert fake.call_count == 4
    # Each call returned one detection; each conv gets 2 (one per target)
    assert len(result.moments_by_conversation["c1"]) == 2
    assert len(result.moments_by_conversation["c2"]) == 2
    # Usage totals across the four calls
    assert result.usage.input_tokens == 400
    assert result.usage.output_tokens == 80
    # Metadata records the model the call was made with
    assert result.metadata["model"] == "fake-test"
    assert isinstance(result.moments_by_conversation["c1"][0], KeyMoment)
