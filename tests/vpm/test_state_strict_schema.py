"""The canonical VPM state models reject unknown (obsolete) fields."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from source.solvers.VPM.config.state import ParticlesState, SolverState


def test_solver_state_rejects_obsolete_processing_unit():
    with pytest.raises(ValidationError):
        SolverState(time_step_size=0.01, processing_unit="CUDA")


def test_particles_state_rejects_legacy_circulation_field():
    with pytest.raises(ValidationError):
        ParticlesState(
            position=[[0.0, 0.0, 0.0]],
            circulation=[1.0],
            core_radius=[0.05],
        )


def test_canonical_fields_still_validate():
    state = SolverState(time_step_size=0.01, time=0.02, step=2)
    assert state.step == 2
