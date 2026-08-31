"""Regression coverage for process-global side effects in the VPM API."""

from __future__ import annotations

import os
import subprocess
import sys


def test_vpm_import_does_not_redirect_streams_or_change_traceback_limit(tmp_path) -> None:
    """VPM configuration must be explicit instead of being inferred at import."""
    log_file = tmp_path / "unexpected-vpm-import.log"
    script = """
import os
import sys
before = (sys.stdout, sys.stderr, getattr(sys, 'tracebacklimit', None))
os.environ['VPM_LOG'] = os.environ['TEST_VPM_LOG']
import source.solvers.vpm
assert (sys.stdout, sys.stderr, getattr(sys, 'tracebacklimit', None)) == before
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "TEST_VPM_LOG": str(log_file)},
    )
    assert completed.returncode == 0, completed.stderr
    assert not log_file.exists()
