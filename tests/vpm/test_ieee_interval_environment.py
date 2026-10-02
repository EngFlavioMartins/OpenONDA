"""Qualification-only checks after the actual Taichi host-mode transition."""

import os

import numpy as np
import pytest

from tests.vpm._gaussian_tail_arithmetic import _platform
from tests.vpm._ieee_interval_environment import IntervalEnvironment


def test_ieee_scope_after_taichi_restores_nested_and_exceptional_calls():
    library = os.environ.get("OPENONDA_QUALIFICATION_FENV_LIBRARY")
    if not library:
        pytest.skip("explicitly built qualification fenv bridge required")
    import taichi as ti

    ti.init(arch=ti.cpu, offline_cache=False, cpu_max_num_threads=1)
    owner = IntervalEnvironment(library)
    probe = lambda: float(np.float64(2.0**-1022)*.5).hex()  # noqa: E731
    before = probe()
    with owner.ieee():
        _platform()
        assert probe() == "0x0.8000000000000p-1022"
        with owner.ieee():
            _platform()
        _platform()
    assert probe() == before
    with pytest.raises(ValueError, match="injected"), owner.ieee():
        _platform()
        raise ValueError("injected")
    assert probe() == before
    ti.reset()
