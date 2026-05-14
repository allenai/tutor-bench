from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tutor_bench.toolkit.logging_setup import configure_logging


@pytest.fixture(autouse=True)
def _reset_root_logger() -> None:
    """Strip handlers from the root logger before each test (global state)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)


def test_configure_logging_creates_log_file_when_run_id_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TUTOR_BENCH_LOG_ROOT", str(tmp_path))

    configure_logging(run_id="test-run-001")
    logging.getLogger("tutor_bench.test").info("hello from test")

    log_path = tmp_path / "logs" / "test-run-001.log"
    assert log_path.exists(), f"expected log file at {log_path}"
    assert "hello from test" in log_path.read_text(encoding="utf-8")


def test_configure_logging_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TUTOR_BENCH_LOG_ROOT", str(tmp_path))

    configure_logging(run_id="test-run-002")
    handler_count_after_first = len(logging.getLogger().handlers)

    configure_logging(run_id="test-run-002")
    handler_count_after_second = len(logging.getLogger().handlers)

    assert handler_count_after_first == handler_count_after_second


def test_configure_logging_console_only_when_no_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TUTOR_BENCH_LOG_ROOT", raising=False)

    configure_logging()  # no run_id

    handlers = logging.getLogger().handlers
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    assert not any(isinstance(h, logging.FileHandler) for h in handlers)
