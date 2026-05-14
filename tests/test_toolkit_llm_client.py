from __future__ import annotations

import pytest

from tutor_bench.toolkit.llm.llm_client import (
    _extract_entry,
    build_batch_entry,
    infer_provider,
    strip_json_fences,
)


class TestInferProvider:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-4-6", "anthropic"),
            ("claude-sonnet-4-7", "anthropic"),
            ("gpt-5.4", "openai"),
            ("o3-mini", "openai"),
            ("o4-preview", "openai"),
            ("gemini-3.1-pro-preview", "gemini"),
        ],
    )
    def test_known_prefixes_route(self, model: str, expected: str) -> None:
        assert infer_provider(model) == expected

    def test_unknown_prefix_raises_with_helpful_message(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer provider"):
            infer_provider("llama-3.5")


class TestStripJsonFences:
    def test_strips_json_fence(self) -> None:
        wrapped = '```json\n{"a": 1}\n```'
        assert strip_json_fences(wrapped) == '{"a": 1}'

    def test_strips_plain_fence(self) -> None:
        wrapped = '```\n{"a": 1}\n```'
        assert strip_json_fences(wrapped) == '{"a": 1}'

    def test_leaves_unfenced_text_untouched(self) -> None:
        assert strip_json_fences('{"a": 1}') == '{"a": 1}'

    def test_handles_trailing_whitespace(self) -> None:
        assert strip_json_fences('```json\n{"a": 1}\n```   \n') == '{"a": 1}'


class TestBatchEntryRoundtrip:
    def test_build_and_extract_preserves_key_and_prompt(self) -> None:
        entry = build_batch_entry("conv_42__scaffolding", "Tutoring transcript here.")
        key, prompt, json_mode, max_tokens = _extract_entry(entry)
        assert key == "conv_42__scaffolding"
        assert prompt == "Tutoring transcript here."
        assert json_mode is True
        assert max_tokens == 65536

    def test_build_with_json_mode_false_drops_response_mime_type(self) -> None:
        entry = build_batch_entry("k", "p", json_mode=False, max_tokens=4096)
        _, _, json_mode, max_tokens = _extract_entry(entry)
        assert json_mode is False
        assert max_tokens == 4096
