#!/usr/bin/env python
"""Pass 1 CLI: detect key moments in tutoring transcripts.

Thin wrapper around :func:`tutor_bench.annotator.detect_key_moments` that
loads transcripts from disk, calls the function with the configured model,
and writes ``detections.json`` + ``report.json`` to the output directory.

Usage::

    scripts/annotate.py --input data/transcripts.jsonl --run-id pilot-01

Each line of the input JSONL must be a conversation dict::

    {"conversation_id": "c1", "turns": [{"turn_number": 1, "role": "TUTOR",
     "text": "..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Repo-root on path so ``python scripts/annotate.py`` works without install.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tutor_bench.annotator import (  # noqa: E402  (import-after-path-edit is intentional)
    KeyMoment,
    PhaseResult,
    detect_key_moments,
    load_annotator_config,
)
from tutor_bench.annotator.config import ModelConfig
from tutor_bench.toolkit.io_utils import load_jsonl, save_jsonl
from tutor_bench.toolkit.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect key tutoring moments in transcripts (Pass 1).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSONL file where each line is a transcript dict.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write detections.json + report.json. "
        "Default: results/annotator/{run_id}/",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identifier for this run (used for output dir + log file). "
        "Default: timestamp.",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        choices=("scaffolding", "rapport"),
        default=("scaffolding", "rapport"),
        help="Annotation types to detect.",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="Process only the first N transcripts (0 = all).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run synchronously one entry at a time instead of batch mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override config model (e.g. claude-sonnet-4-6).",
    )
    return parser.parse_args()


def _keymoment_to_dict(km: KeyMoment) -> dict:
    return asdict(km)


def _write_outputs(out_dir: Path, result: PhaseResult, run_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    detections_rows = [
        {
            "conversation_id": conv_id,
            "moments": [_keymoment_to_dict(km) for km in moments],
        }
        for conv_id, moments in sorted(result.moments_by_conversation.items())
    ]
    save_jsonl(out_dir / "detections.jsonl", detections_rows)

    report = {
        "run_id": run_id,
        "pass": "detect_key_moments",
        "metadata": result.metadata,
        "n_conversations": len(result.moments_by_conversation),
        "n_moments": sum(len(m) for m in result.moments_by_conversation.values()),
        "usage": asdict(result.usage),
        "cost_usd": asdict(result.cost),
        "errors": [
            {"key": e.key, "error": e.error}
            for e in result.errors
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("detect-%Y%m%dT%H%M%SZ")
    configure_logging(run_id=run_id)

    cfg = load_annotator_config()
    if args.model:
        cfg_model_dict = asdict(cfg.model)
        cfg_model_dict["model"] = args.model
        cfg = type(cfg)(
            model=ModelConfig(**cfg_model_dict),
            pricing=cfg.pricing,
            retry=cfg.retry,
            batch_timeout_sec=cfg.batch_timeout_sec,
        )

    logger.info("loading transcripts from %s", args.input)
    transcripts = load_jsonl(args.input)
    if args.test > 0:
        transcripts = transcripts[: args.test]
    if not transcripts:
        logger.error("no transcripts loaded from %s", args.input)
        return 2
    logger.info(
        "running detect_key_moments: %d transcripts, targets=%s, mode=%s, model=%s",
        len(transcripts),
        list(args.target),
        "sync" if args.sync else "batch",
        cfg.model.model,
    )

    result = detect_key_moments(
        transcripts,
        config=cfg,
        targets=args.target,
        batch=not args.sync,
    )

    out_dir = args.output_dir or (Path(__file__).parent.parent / "results" / "annotator" / run_id)
    _write_outputs(out_dir, result, run_id)

    n_moments = sum(len(m) for m in result.moments_by_conversation.values())
    logger.info(
        "done: %d moments across %d conversations | tokens in=%d out=%d total=%d | cost=$%.4f",
        n_moments,
        len(result.moments_by_conversation),
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens,
        result.cost.total_usd,
    )
    logger.info("outputs written to %s", out_dir)
    if result.errors:
        logger.warning("%d parse errors recorded; see report.json", len(result.errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
