"""Regression tests for the compact VPM block logger."""

from __future__ import annotations

import pytest

from source.solvers.vpm.io.logging import Logging


@pytest.fixture(autouse=True)
def reset_progress_state():
    Logging._last_progress_wall = None
    Logging._active_step = None
    yield
    Logging._last_progress_wall = None
    Logging._active_step = None


def test_routine_suppression_keeps_warnings_visible(capsys) -> None:
    try:
        Logging._last_block_section = None
        Logging.set_routine_messages_enabled(False)
        Logging.message("routine detail")
        Logging.warning("important warning")
    finally:
        Logging.set_routine_messages_enabled(True)

    output = capsys.readouterr().out
    assert "routine detail" not in output
    assert "Warning     | important warning" in output


def test_progress_is_one_line_with_accepted_flow_and_wall_times(capsys) -> None:
    Logging.set_routine_messages_enabled(True)
    Logging.time_step(61, 0.4757141, 929.9, total_steps=100, n_particles=14080)

    output = capsys.readouterr().out
    assert "Progress    | step=     61/100" in output
    assert "t=    0.4757 s" in output
    assert "elapsed=00:15:29.9" in output
    assert "N=   14,080" in output
    assert len(output.splitlines()) == 1
    assert "BEGIN" not in output
    assert "COMPLETED" not in output
    assert "time at start" not in output.lower()


def test_progress_is_throttled_but_final_state_is_visible(capsys):
    Logging.set_routine_messages_enabled(True)
    for step, wall in [(1, 1.0), (2, 2.0), (3, 30.0), (4, 31.0), (5, 32.0)]:
        Logging.time_step(step, step * 0.1, wall, total_steps=5)
    output = capsys.readouterr().out
    assert "step=        1/5" in output
    assert "step=        2/5" not in output
    assert "step=        3/5" not in output
    assert "step=        4/5" in output
    assert "step=        5/5" in output


def test_begin_step_does_not_claim_that_work_has_completed(capsys):
    Logging.begin_step(7)
    assert capsys.readouterr().out == ""
    Logging.warning("failed before acceptance")
    assert "Warning     | step 7 | failed before acceptance" in capsys.readouterr().out
