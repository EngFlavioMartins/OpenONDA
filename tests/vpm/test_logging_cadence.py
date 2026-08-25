"""Regression tests for VPM routine-message cadence."""

from __future__ import annotations

from source.solvers.vpm.io.logging import Logging


def test_routine_suppression_keeps_warnings_visible(capsys) -> None:
    try:
        Logging.set_routine_messages_enabled(False)
        Logging.message("routine detail")
        Logging.warning("important warning")
    finally:
        Logging.set_routine_messages_enabled(True)

    output = capsys.readouterr().out
    assert "routine detail" not in output
    assert "vpm      warning  important warning" in output
