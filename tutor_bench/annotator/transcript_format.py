"""Render an annotator-shaped conversation dict into a flat prompt-ready string.

Input shape (matches the consolidated transcript JSON used throughout the
project)::

    {
        "conversation_id": "c1",
        "turns": [
            {"turn_number": 1, "role": "TUTOR", "text": "..."},
            {"turn_number": 2, "role": "STUDENT", "text": "..."},
            ...
        ]
    }

Output is the body that gets substituted into ``{transcript}`` in the detect
prompts. All turns pass through; the pilot does not expose a dialogue-only
filter (re-add as an experiment if useful — see ``experiments/README.md``).
"""

from __future__ import annotations

from collections.abc import Mapping


def format_transcript(conversation: Mapping[str, object]) -> str:
    """Render ``conversation`` as ``Turn {n}. {ROLE}: {text}`` lines, one per turn."""
    turns = conversation.get("turns") or []
    return "\n".join(
        f"Turn {turn['turn_number']}. {turn['role']}: {turn['text']}"
        for turn in turns
    )
