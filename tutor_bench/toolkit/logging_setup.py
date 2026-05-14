"""Process-wide logging configuration.

Call :func:`configure_logging` once at process start. Pass a ``run_id`` to also
attach a per-run file handler at ``logs/{run_id}.log``.

Environment variables:
    LOG_LEVEL              DEBUG | INFO | WARNING | ERROR (default INFO)
    TUTOR_BENCH_LOG_ROOT   override the directory under which ``logs/`` is
                           created (for tests/CI); defaults to repo root.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s — %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONSOLE_SENTINEL = "_tutor_bench_console_handler"
_FILE_SENTINEL = "_tutor_bench_file_handler"


def _log_root() -> Path:
    override = os.environ.get("TUTOR_BENCH_LOG_ROOT", "")
    return Path(override) if override else _REPO_ROOT_DEFAULT


def _resolve_level() -> int:
    name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(name)
    return level if isinstance(level, int) else logging.INFO


def _has_handler(root: logging.Logger, sentinel: str) -> bool:
    return any(getattr(h, sentinel, False) for h in root.handlers)


def configure_logging(run_id: str | None = None) -> None:
    """Configure the root logger. Safe to call repeatedly.

    The first call attaches a stderr console handler and sets the level from
    ``LOG_LEVEL``. Subsequent calls leave existing handlers alone. Passing a
    ``run_id`` additionally opens ``logs/{run_id}.log`` (idempotent).
    """
    root = logging.getLogger()
    root.setLevel(_resolve_level())
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if not _has_handler(root, _CONSOLE_SENTINEL):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        setattr(console, _CONSOLE_SENTINEL, True)
        root.addHandler(console)

    if run_id is None:
        return
    if _has_handler(root, _FILE_SENTINEL):
        return

    log_dir = _log_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{run_id}.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    setattr(file_handler, _FILE_SENTINEL, True)
    root.addHandler(file_handler)
