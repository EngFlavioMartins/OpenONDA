"""Regression tests for the native FVM logging sink."""

from __future__ import annotations

from io import StringIO
import sys

import pytest

from source.solvers.fvm.io import logging as fvm_logging
from source.solvers.fvm.mesh.progress import mesh_stage, mesher_log_session


def test_nested_phase_timing_is_independent_between_solver_owners(monkeypatch) -> None:
    clock = iter((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    monkeypatch.setattr(fvm_logging.time, "perf_counter", lambda: next(clock))
    first = fvm_logging.Timer()
    second = fvm_logging.Timer()
    first.start("pressure")
    second.start("pressure")
    first.start("pressure")
    assert first.stop("pressure") == pytest.approx(1.0)
    assert second.stop("pressure") == pytest.approx(3.0)
    assert first.stop("pressure") == pytest.approx(5.0)
    assert first.stop("pressure") == 0.0


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
        assert "MESH" in live and "meshing session" in live
        assert "expensive refinement" in live and "start" in live
        stage.details(cells=123)

    complete = path.read_text(encoding="utf-8")
    assert "expensive refinement" in complete and "done" in complete
    assert "Cells" in complete and "123" in complete
    assert "complete" in complete
