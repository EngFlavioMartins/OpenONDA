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


def test_runtime_configuration_preserves_numba_compilation_after_initialization(tmp_path):
    script = """
import os
import numpy as np
import numba
from openonda.runtime import RunConfig

@numba.njit(parallel=True)
def first(x):
    return (x * x).sum()

x = np.arange(16, dtype=np.float64)
assert first(x) == np.dot(x, x)
capacity = numba.config.NUMBA_NUM_THREADS
setting = os.environ.get('NUMBA_NUM_THREADS')
RunConfig(cpu_cores=1).ensure_runtime('unused.py')
numba.config.reload_config()

@numba.njit(parallel=True)
def second(x):
    return (x + x).sum()

assert second(x) == 2 * x.sum()
assert numba.get_num_threads() == 1
assert numba.config.NUMBA_NUM_THREADS == capacity
assert os.environ.get('NUMBA_NUM_THREADS') == setting
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
