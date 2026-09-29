"""A rejected wall crossing subdivides only the particle RK interval."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.solid import SolidParticleGuard
from source.solvers.vpm.core.evolution import EvolutionStepper
from source.solvers.vpm.wall_errors import WallCorrectionTooLargeError


class _Particles:
    def __init__(self):
        self.position = np.array([[0.0, 0.0, 0.0]])
        self.vortex_strength = np.array([[1.0, 0.0, 0.0]])
        self.core_radius = np.array([0.1])
        self.revisions = 0

    def __len__(self):
        return 1

    def touch_state(self):
        self.revisions += 1


class _Boundary:
    revision = 0

    def constrain_motion(self, positions, spacing, *, starts=None):
        corrected = np.asarray(positions).copy()
        corrected[:, 0] = np.maximum(corrected[:, 0], 0)
        delta = corrected - positions
        maximum = float(np.linalg.norm(delta, axis=1).max())
        if maximum > 0.25 * spacing:
            raise WallCorrectionTooLargeError("wall correction too large")
        changed = np.any(delta != 0, axis=1)
        return corrected, changed, delta[changed], maximum


def test_wall_retry_preserves_macro_time_strength_and_accepted_diagnostics(monkeypatch):
    monkeypatch.setattr("source.solvers.vpm.core.evolution.ti.sync", lambda: None)
    particles = _Particles()
    physics = SimpleNamespace(
        _download_vector_field=lambda field, count: field[:count].copy(),
        _upload_vector_array=lambda values, field, count: np.copyto(field[:count], values),
        rate_projection_max_correction_ratio=0.0,
        last_solid_projection=None,
    )
    guard = SolidParticleGuard(_Boundary(), physics, 0.045, logging.getLogger(__name__))
    stage_rhs = SimpleNamespace(accepted_position_projector=guard.accepted)
    calls = []

    class Integrator:
        def advance(
            self,
            *,
            position,
            vortex_strength,
            time,
            time_step_size,
            accepted_position_projector,
            **_,
        ):
            calls.append((time, time_step_size))
            guard.stage(
                SimpleNamespace(
                    position=position, vortex_strength=vortex_strength, count=1, stage_index=0
                )
            )
            position[0, 0] -= time_step_size
            vortex_strength[0, 0] += time_step_size
            accepted_position_projector(position, vortex_strength, 1)

    solver = SimpleNamespace(
        particles=particles,
        physics=physics,
        stage_rhs=stage_rhs,
        integrator=Integrator(),
        time=2.0,
        axisymmetric_axis=-1,
        vlm_solver=None,
    )
    EvolutionStepper(solver)._advance_particles(0.05)

    assert calls[0] == (2.0, 0.05)
    assert len(calls) > 1
    assert max(dt for _, dt in calls[1:]) <= 0.025
    np.testing.assert_allclose(particles.position, [[0, 0, 0]])
    np.testing.assert_allclose(particles.vortex_strength, [[1.05, 0, 0]])
    assert solver.time == 2.0  # The owner commits one macro clock after every phase.
    assert guard._reference is None
    assert 0 < physics.last_solid_projection["accepted_count"] < len(calls)


def test_vlm_wall_rejection_restores_particles_without_splitting(monkeypatch):
    monkeypatch.setattr("source.solvers.vpm.core.evolution.ti.sync", lambda: None)
    particles = _Particles()
    physics = SimpleNamespace(
        _download_vector_field=lambda field, count: field[:count].copy(),
        _upload_vector_array=lambda values, field, count: np.copyto(field[:count], values),
        rate_projection_max_correction_ratio=0.0,
        last_solid_projection=None,
    )
    guard = SolidParticleGuard(_Boundary(), physics, 0.045, logging.getLogger(__name__))
    calls = []

    class Integrator:
        def advance(self, *, position, vortex_strength, accepted_position_projector, **_):
            calls.append(1)
            position[0, 0] = -0.05
            vortex_strength[0, 0] = 9.0
            accepted_position_projector(position, vortex_strength, 1)

    solver = SimpleNamespace(
        particles=particles,
        physics=physics,
        integrator=Integrator(),
        stage_rhs=SimpleNamespace(accepted_position_projector=guard.accepted),
        time=2.0,
        axisymmetric_axis=-1,
        vlm_solver=object(),
    )
    with pytest.raises(WallCorrectionTooLargeError):
        EvolutionStepper(solver)._advance_particles(0.05)
    assert len(calls) == 1
    np.testing.assert_array_equal(particles.position, [[0, 0, 0]])
    np.testing.assert_array_equal(particles.vortex_strength, [[1, 0, 0]])
    assert physics.last_solid_projection is None


def test_later_substep_failure_restores_whole_macro_state(monkeypatch, caplog):
    monkeypatch.setattr("source.solvers.vpm.core.evolution.ti.sync", lambda: None)
    particles = _Particles()
    initial_budget = {
        "stage_count": 0,
        "accepted_count": 0,
        "stage_displacement_l1": 0.0,
        "accepted_displacement_l1": 0.0,
        "stage_impulse_change": np.zeros(3),
        "accepted_impulse_change": np.zeros(3),
    }
    physics = SimpleNamespace(
        _download_vector_field=lambda field, count: field[:count].copy(),
        _upload_vector_array=lambda values, field, count: np.copyto(field[:count], values),
        rate_projection_max_correction_ratio=0.125,
        last_solid_projection=initial_budget.copy(),
    )
    guard = SolidParticleGuard(_Boundary(), physics, 0.1, logging.getLogger(__name__))
    calls = []

    class Integrator:
        def advance(
            self,
            *,
            position,
            vortex_strength,
            time,
            time_step_size,
            accepted_position_projector,
            **_,
        ):
            calls.append((time, time_step_size))
            guard.stage(
                SimpleNamespace(
                    position=position, vortex_strength=vortex_strength, count=1, stage_index=0
                )
            )
            position[0, 0] -= time_step_size
            vortex_strength[0, 0] += time_step_size
            physics.rate_projection_max_correction_ratio = 0.9
            if time > 2.0:
                raise ValueError("later stage failed")
            accepted_position_projector(position, vortex_strength, 1)

    solver = SimpleNamespace(
        particles=particles,
        physics=physics,
        integrator=Integrator(),
        stage_rhs=SimpleNamespace(accepted_position_projector=guard.accepted),
        time=2.0,
        axisymmetric_axis=-1,
        vlm_solver=None,
    )
    with pytest.raises(ValueError, match="later stage failed"):
        EvolutionStepper(solver)._advance_particles(0.05)

    assert calls == [(2.0, 0.05), (2.0, 0.025), (2.025, 0.025)]
    np.testing.assert_array_equal(particles.position, [[0, 0, 0]])
    np.testing.assert_array_equal(particles.vortex_strength, [[1, 0, 0]])
    for key, value in initial_budget.items():
        np.testing.assert_array_equal(physics.last_solid_projection[key], value)
    assert physics.rate_projection_max_correction_ratio == 0.125
    assert guard._reference is None
    assert not [record for record in caplog.records if "Solid-wall exclusion" in record.message]
