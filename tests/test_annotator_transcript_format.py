from __future__ import annotations

from tutor_bench.annotator.transcript_format import format_transcript


def _conv(turns: list[dict]) -> dict:
    return {"conversation_id": "c1", "turns": turns}


def test_format_transcript_renders_one_line_per_turn_with_index() -> None:
    conv = _conv(
        [
            {"turn_number": 1, "role": "TUTOR", "text": "Let's start."},
            {"turn_number": 2, "role": "STUDENT", "text": "Okay."},
        ]
    )
    out = format_transcript(conv)
    assert out == "Turn 1. TUTOR: Let's start.\nTurn 2. STUDENT: Okay."


def test_format_transcript_preserves_non_dialogue_turns() -> None:
    """No dialogue_only filter in the pilot — enrichment turns must pass through."""
    conv = _conv(
        [
            {"turn_number": 1, "role": "TUTOR", "text": "Hi."},
            {"turn_number": 2, "role": "SYSTEM", "text": "[student joined]", "type": "ENRICHMENT"},
            {"turn_number": 3, "role": "STUDENT", "text": "Hello!"},
        ]
    )
    out = format_transcript(conv)
    assert "[student joined]" in out
    assert "Turn 2. SYSTEM" in out


def test_format_transcript_empty_turns_returns_empty_string() -> None:
    assert format_transcript(_conv([])) == ""


def test_format_transcript_handles_unicode_and_whitespace() -> None:
    conv = _conv(
        [
            {"turn_number": 1, "role": "TUTOR", "text": "¿Cómo estás? — 你好"},
            {"turn_number": 2, "role": "STUDENT", "text": "  trim?  "},
        ]
    )
    out = format_transcript(conv)
    assert "¿Cómo estás? — 你好" in out
    # Whitespace is preserved verbatim — caller's transcript, caller's whitespace.
    assert "  trim?  " in out
