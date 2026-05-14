"""Pass 1 of the annotator: identify key tutoring moments in transcripts.

Public surface::

    from tutor_bench.annotator import detect_key_moments, KeyMoment, PhaseResult

    result = detect_key_moments(transcripts, config=load_annotator_config())
    for conv_id, moments in result.moments_by_conversation.items():
        for moment in moments:
            ...

The function sends each ``(conversation, target)`` pair to an LLM with one of
the two canonical detect prompts and parses the structured JSON response
into typed :class:`KeyMoment` records.

Parser invariants (load-bearing for downstream Pass 2/3 reproducibility,
ported verbatim from ``ai2-synthetic-annotations`` @ 106dd78):

- ``annotation_type`` falls back to the key suffix (``conv__scaffolding``
  → ``"scaffolding"``) when missing OR set to an unknown value.
- ``suggested_cut_turn`` is clamped to ``[max(1, turn_start - 2), turn_end]``;
  values outside that range are replaced with ``max(1, turn_start - 1)``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, ClassVar, Literal, Protocol

from tutor_bench.annotator.config import AnnotatorConfig, load_annotator_config
from tutor_bench.annotator.transcript_format import format_transcript
from tutor_bench.toolkit.cost_tracker import CostBreakdown, CostTracker, TokenUsage
from tutor_bench.toolkit.llm.llm_client import (
    ModelClient,
    ModelResponse,
    build_batch_entry,
    run_batch,
    run_sync_entries,
)

logger = logging.getLogger(__name__)

VALID_ANNOTATION_TYPES: frozenset[str] = frozenset({"scaffolding", "rapport"})


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyMoment:
    """One detected pedagogical moment in a transcript."""

    conversation_id: str
    annotation_type: Literal["scaffolding", "rapport"]
    turn_start: int
    turn_end: int
    brief_description: str
    suggested_cut_turn: int | None = None


@dataclass(frozen=True)
class ParseError:
    """A single per-entry failure (bad JSON, API error, etc.). Non-fatal."""

    key: str
    error: str
    raw: str | None = None


@dataclass(frozen=True)
class PhaseResult:
    """Aggregated output of one detect run."""

    moments_by_conversation: dict[str, list[KeyMoment]]
    usage: TokenUsage
    cost: CostBreakdown
    errors: list[ParseError] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


class _BuildableClient(Protocol):
    """Internal helper protocol for type-checking the client factory."""

    provider: ClassVar[str]
    model: str

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse: ...


def detect_key_moments(
    transcripts: Sequence[Mapping[str, Any]],
    *,
    config: AnnotatorConfig | None = None,
    targets: Sequence[str] = ("scaffolding", "rapport"),
    batch: bool = True,
    client: ModelClient | None = None,
) -> PhaseResult:
    """Detect scaffolding/rapport key moments across ``transcripts``.

    Parameters
    ----------
    transcripts
        Sequence of consolidated transcript dicts. Each must have a
        ``conversation_id`` and a ``turns`` list (see
        :func:`tutor_bench.annotator.transcript_format.format_transcript`).
    config
        Annotator runtime config; when omitted, loaded from the default
        ``configs/annotator.yaml``.
    targets
        Which annotation types to detect. Default detects both supported
        types in one pass.
    batch
        ``True`` uses the provider's batch API; ``False`` runs entries
        synchronously one at a time. Sync is mostly useful for tests and
        small ad-hoc runs.
    client
        Optional injected :class:`ModelClient`. When ``None`` a client is
        built from ``config.model``. Test code passes a fake.

    Returns
    -------
    PhaseResult
        Moments grouped by conversation, aggregate token usage, cost, and
        any per-entry parse errors.
    """
    cfg = config if config is not None else load_annotator_config()
    invalid = [t for t in targets if t not in VALID_ANNOTATION_TYPES]
    if invalid:
        raise ValueError(
            f"Unknown detection target(s): {invalid}. "
            f"Valid targets: {sorted(VALID_ANNOTATION_TYPES)}."
        )

    llm_client = client if client is not None else _build_default_client(cfg)
    prompt_cache = _load_prompt_cache(targets)
    entries = _build_detection_entries(transcripts, targets, prompt_cache)
    logger.info(
        "detect_key_moments: %d transcripts × %d targets = %d entries (mode=%s, model=%s)",
        len(transcripts),
        len(targets),
        len(entries),
        "batch" if batch else "sync",
        llm_client.model,
    )

    if batch:
        raw_entries = run_batch(
            llm_client,
            entries,
            display_name="detect_key_moments",
            poll_interval_sec=cfg.model.batch_poll_interval_sec,
            batch_timeout_sec=cfg.batch_timeout_sec,
            thinking=cfg.model.thinking,
            thinking_budget=cfg.model.thinking_budget,
            max_retries=cfg.retry.max_retries,
            base_delay_sec=cfg.retry.base_delay_sec,
        )
    else:
        raw_entries = run_sync_entries(
            llm_client,
            entries,
            thinking=cfg.model.thinking,
            thinking_budget=cfg.model.thinking_budget,
        )

    moments_by_conv, _usage_by_conv, errors = _parse_detection_results(raw_entries)

    tracker = CostTracker(pricing=cfg.pricing)
    for entry_payload in raw_entries.values():
        usage_dict = entry_payload.get("usage")
        if usage_dict:
            tracker.record(llm_client.provider, llm_client.model, usage_dict)

    return PhaseResult(
        moments_by_conversation=moments_by_conv,
        usage=tracker.usage(),
        cost=tracker.cost(),
        errors=errors,
        metadata={
            "provider": llm_client.provider,
            "model": llm_client.model,
            "targets": list(targets),
            "mode": "batch" if batch else "sync",
            "thinking": cfg.model.thinking,
            "thinking_budget": cfg.model.thinking_budget,
            "n_transcripts": len(transcripts),
            "n_entries": len(entries),
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_default_client(cfg: AnnotatorConfig) -> ModelClient:
    if cfg.model.provider == "anthropic":
        from tutor_bench.toolkit.llm.anthropic import AnthropicClient

        return AnthropicClient(cfg.model.model)
    raise NotImplementedError(
        f"Provider {cfg.model.provider!r} is not implemented in this pilot. "
        "Add a client module under tutor_bench/toolkit/llm/ to extend support."
    )


def _load_prompt_cache(targets: Sequence[str]) -> dict[str, str]:
    """Read the canonical detect prompt for each target from package data."""
    cache: dict[str, str] = {}
    for target in targets:
        filename = f"detect_key_moments_{target}.md"
        path = resources.files("tutor_bench.prompts") / filename
        cache[target] = path.read_text(encoding="utf-8")
    return cache


def _build_detection_entries(
    transcripts: Iterable[Mapping[str, Any]],
    targets: Sequence[str],
    prompt_cache: Mapping[str, str],
) -> list[dict[str, Any]]:
    """One batch entry per (conversation × target). Key format: ``{conv_id}__{target}``."""
    entries: list[dict[str, Any]] = []
    for conv in transcripts:
        conv_id = conv["conversation_id"]
        transcript_text = format_transcript(conv)
        for target in targets:
            prompt = prompt_cache[target].replace("{transcript}", transcript_text)
            entries.append(build_batch_entry(f"{conv_id}__{target}", prompt))
    return entries


def _parse_detection_results(
    raw_entries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[KeyMoment]], dict[str, TokenUsage], list[ParseError]]:
    """Parse ``{key: {text, usage[, error]}}`` into typed moments + per-conv usage + errors."""
    moments_by_conv: dict[str, list[KeyMoment]] = {}
    usage_by_conv: dict[str, dict[str, int]] = {}
    errors: list[ParseError] = []

    for key, payload in raw_entries.items():
        conv_id, _, ann_type_from_key = key.rpartition("__")
        if not conv_id:
            conv_id, ann_type_from_key = key, ""

        usage_by_conv.setdefault(conv_id, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        usage_raw = payload.get("usage") or {}
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            usage_by_conv[conv_id][field_name] += int(usage_raw.get(field_name, 0))

        if "error" in payload:
            errors.append(ParseError(key=key, error=str(payload["error"])))
            continue

        text = payload.get("text") or ""
        if not text:
            errors.append(ParseError(key=key, error="Empty response"))
            continue

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(ParseError(key=key, error=f"JSON parse error: {exc}", raw=text[:500]))
            continue

        for det in parsed.get("detections", []):
            moment = _parse_one_detection(conv_id, det, ann_type_from_key)
            if moment is None:
                continue
            moments_by_conv.setdefault(conv_id, []).append(moment)

    # Convert usage dicts to TokenUsage dataclasses for return
    return (
        moments_by_conv,
        {
            cid: TokenUsage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
            )
            for cid, usage in usage_by_conv.items()
        },
        errors,
    )


def _parse_one_detection(
    conv_id: str,
    det: Mapping[str, Any],
    ann_type_from_key: str,
) -> KeyMoment | None:
    """Validate + normalize one detection dict. Return ``None`` if unsalvageable."""
    raw_type = det.get("annotation_type")
    if raw_type not in VALID_ANNOTATION_TYPES:
        if ann_type_from_key in VALID_ANNOTATION_TYPES:
            ann_type = ann_type_from_key
        else:
            return None
    else:
        ann_type = raw_type

    try:
        turn_start = int(det["turn_start"])
        turn_end = int(det.get("turn_end", turn_start))
    except (KeyError, TypeError, ValueError):
        return None

    lower_bound = max(1, turn_start - 2)
    sct = det.get("suggested_cut_turn")
    if sct is None or not (lower_bound <= int(sct) <= turn_end):
        sct = max(1, turn_start - 1)
    else:
        sct = int(sct)

    return KeyMoment(
        conversation_id=conv_id,
        annotation_type=ann_type,  # type: ignore[arg-type]  # narrowed via VALID_ANNOTATION_TYPES
        turn_start=turn_start,
        turn_end=turn_end,
        brief_description=str(det.get("brief_description", "")),
        suggested_cut_turn=sct,
    )
