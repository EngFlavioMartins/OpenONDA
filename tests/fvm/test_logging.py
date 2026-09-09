"""Regression tests for the native FVM logging sink."""

from __future__ import annotations

from io import StringIO
import sys

from source.solvers.fvm.io import logging as fvm_logging
from source.solvers.fvm.mesh.progress import mesh_stage, mesher_log_session


def test_console_sink_does_not_follow_later_stdout_redirection(tmp_path, monkeypatch) -> None:
    console = StringIO()
    redirected = StringIO()
    monkeypatch.setattr(fvm_logging, "_CONSOLE_STDOUT", console)

    logger = fvm_logging.Logging(tmp_path)
    monkeypatch.setattr(sys, "stdout", redirected)
    logger.info("stable sink")
    logger.close()

    assert " EVENTS\n  Stable sink" in console.getvalue()
    assert "stable sink" not in redirected.getvalue()
    assert " EVENTS\n  Stable sink" in (tmp_path / "solution" / "fvm.log").read_text()


def test_mesher_stage_is_visible_before_stage_completion(tmp_path) -> None:
    path = tmp_path / "chosen-solution" / "mesher.log"

    with mesher_log_session(path), mesh_stage("expensive refinement") as stage:
        live = path.read_text(encoding="utf-8")
        assert "START    meshing session" in live
        assert "START    expensive refinement" in live
        stage.details(cells=123)

    complete = path.read_text(encoding="utf-8")
    assert "DONE     expensive refinement" in complete
    assert "cells=123" in complete
    assert "COMPLETE meshing session" in complete
