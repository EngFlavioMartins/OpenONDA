"""A rejected wall crossing subdivides only the particle RK interval."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.solid import SolidParticleGuard
from source.solvers.vpm.core.evolution import EvolutionStepper
from source.solvers.vpm.wall_errors import WallCorrectionTooLargeError


def test_native_rk_wall_retry_preserves_clock_strength_and_projection_budget(tmp_path, caplog):
    """A real RK stage crosses a rotated wall and retries on native fields."""
    from openonda import vpm
    from tests.coupler._solid_geometry import wall_case

    boundary, start, _, normal = wall_case("rotated")
    spacing = 0.1
    dt = 0.16
    strength = np.array([[0.0, 0.0, 1e-6]])
    # The unrestricted macro trajectory needs 0.06 m of correction, beyond
    # the 0.025 m wall tolerance. This uses the actual segment classifier.
    with pytest.raises(WallCorrectionTooLargeError):
        boundary.constrain_motion((start - dt * normal)[None], spacing, starts=start[None])

    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            backup=vpm.Backup(interval_steps=0),
            numerics=vpm.Numerics(
                time_step_size=dt,
                compute_device="CPU",
                precision="f64",
                max_n_particles=4,
                freestream_velocity=tuple(-normal),
                induction=vpm.DirectInduction(),
                integrator=vpm.RK2(),
                viscous=vpm.ViscousConfig.inviscid(),
                verbose=False,
            ),
        )
    )
    try:
        solver.add_vortex_particles(
            position=start[None],
            velocity=np.zeros((1, 3)),
            vortex_strength=strength,
            core_radius=np.array([0.05]),
            particle_volume=np.array([spacing**3]),
            kinematic_viscosity=np.zeros(1),
        )
        guard = SolidParticleGuard(boundary, solver.physics, spacing, logging.getLogger(__name__))
        solver.stage_rhs.position_guard = guard.stage
        solver.stage_rhs.accepted_position_projector = guard.accepted
        initial_budget = {
            "stage_count": 7,
            "accepted_count": 3,
            "stage_displacement_l1": 0.5,
            "accepted_displacement_l1": 0.25,
            "stage_impulse_change": np.array([0.1, 0.2, 0.3]),
            "accepted_impulse_change": np.array([0.4, 0.5, 0.6]),
        }
        solver.physics.last_solid_projection = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in initial_budget.items()
        }
        initial_time, initial_step = solver.time, solver.step
        with caplog.at_level(logging.INFO, logger=__name__):
            solver.advance(defer_output=True)

        positions = solver.particles.position_cpu()
        assert not boundary.contains(positions).any()
        wall_clearance = np.dot(positions[0] - start, normal) + 0.1
        assert 0 < wall_clearance <= max(8 * boundary.tolerance, 2e-7 * spacing)
        np.testing.assert_allclose(solver.particles.vortex_strength_cpu(), strength, atol=1e-12)
        assert solver.time == pytest.approx(initial_time + dt)
        assert solver.step == initial_step + 1
        assert guard._reference is None
        assert any("wall correction accepted after" in record.message for record in caplog.records)
        projection_logs = [
            record for record in caplog.records if record.message.startswith("Solid-wall exclusion")
        ]
        assert projection_logs
        assert all(record.args[2] <= 0.25 * spacing for record in projection_logs)
        budget = solver.physics.last_solid_projection
        # Retain the prior diagnostic history and count only accepted RK trials.
        total_displacement = 0.0
        for kind in ("stage", "accepted"):
            assert budget[f"{kind}_count"] >= initial_budget[f"{kind}_count"]
            displacement = (
                budget[f"{kind}_displacement_l1"] - initial_budget[f"{kind}_displacement_l1"]
            )
            assert displacement >= 0.0
            kind_logs = [record for record in projection_logs if record.args[1] == kind]
            assert budget[f"{kind}_count"] - initial_budget[f"{kind}_count"] == sum(
                record.args[0] for record in kind_logs
            )
            assert displacement == pytest.approx(
                sum(record.args[2] for record in kind_logs), abs=1e-12
            )
            total_displacement += displacement
            np.testing.assert_allclose(
                budget[f"{kind}_impulse_change"] - initial_budget[f"{kind}_impulse_change"],
                0.5 * np.cross(displacement * normal, strength[0]),
                atol=1e-10,
            )
        assert 0.0 < total_displacement < 2.0 * dt
    finally:
        solver.close()


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
    assert solver.time == 2.0  # The solver commits one macro clock after every phase.
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
