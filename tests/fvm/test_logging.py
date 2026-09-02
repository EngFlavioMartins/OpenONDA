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

    assert " EVENTS\n  Stable sink" in console.getvalue()
    assert "stable sink" not in redirected.getvalue()
    assert " EVENTS\n  Stable sink" in (tmp_path / "solution" / "fvm.log").read_text()


def test_simple_step_uses_the_shared_block_layout(tmp_path, monkeypatch) -> None:
    console = StringIO()
    monkeypatch.setattr(fvm_logging, "_CONSOLE_STDOUT", console)

    logger = fvm_logging.Logging(tmp_path)
    logger.step_begin(10_041, 40.0, 0.004)
    logger.convergence_info({"velocity": 1.28e-7, "kinematic_pressure": 7.75e-8})
    logger.continuity_info(1.38e-12, 2.0e-12)
    logger.courant_info(0.107)
    logger.force_info({"wall": {"coeffs": {"drag_coefficient": 1.1009}}})
    logger.step_end(0.26)
    logger.close()

    output = console.getvalue()
    assert " FVM TIME STEP 10,041" in output
    assert "FLOW TIME 4.000000e+01 s" in output
    assert "WALL TIME 00:00:00.0" in output
    assert " TIME CONTROL\n  Time step" in output
    assert " CONVERGENCE\n  Residual, velocity" in output
    assert " CONSERVATION\n  Continuity error, max" in output
    assert " AERODYNAMIC LOADS\n  Drag coefficient" in output
    assert " TIMING\n  Wall time" in output
    assert "fvm      step" not in output
