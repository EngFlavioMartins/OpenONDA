"""Regression tests for the native FVM logging sink."""

from __future__ import annotations

from io import StringIO
import sys

from source.solvers.fvm.io import logging as fvm_logging


def test_console_sink_does_not_follow_later_stdout_redirection(tmp_path, monkeypatch) -> None:
    console = StringIO()
    redirected = StringIO()
    monkeypatch.setattr(fvm_logging, "_CONSOLE_STDOUT", console)

    logger = fvm_logging.Logging(tmp_path)
    monkeypatch.setattr(sys, "stdout", redirected)
    logger.info("stable sink")
    logger.close()

    assert "fvm      stable sink" in console.getvalue()
    assert "stable sink" not in redirected.getvalue()
    assert "fvm      stable sink" in (tmp_path / "solution" / "fvm.log").read_text()
