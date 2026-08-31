"""The VPM exposes one canonical runtime-state model."""

from __future__ import annotations

import pytest

from source.solvers.vpm.config import RestartState
from source.solvers.vpm.config import state as state_module


def test_duplicate_snapshot_models_are_not_exposed():
    assert not hasattr(state_module, "SolverState")
    assert not hasattr(state_module, "ParticlesState")


def test_restart_state_rejects_negative_clock_values():
    with pytest.raises(ValueError, match="time"):
        RestartState(time=-0.01)
    with pytest.raises(ValueError, match="step"):
        RestartState(step=-1)


def test_canonical_fields_still_validate():
    state = RestartState(time=0.02, step=2)
    assert state.step == 2
