"""Real-VPM certification gates for the experimental flux handoff."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import erf

pytest.importorskip("taichi", reason="VPM integration requires taichi")

from source.coupler.flux_handoff import (  # noqa: E402
    FluxReleaseHandoff,
    inject_vpm_release_batch,
    vorticity_transport_flux,
)
from source.solvers.vpm import (  # noqa: E402
    AdvectionConfig,
    StabilizationConfig,
    StretchingConfig,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
    VPMSetup,
    VPMSolver,
)


def _make_inviscid_solver(
    case_directory: Path,
    *,
    time_step_size: float,
    particle_spacing: float,
    freestream_speed: float,
) -> VPMSolver:
    return VPMSolver(
        VPMSetup(
            time_step_size=time_step_size,
            advection=AdvectionConfig("RK3"),
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig.inviscid(particle_spacing=particle_spacing),
            turbulence=TurbulenceConfig.inviscid(),
            stabilization=StabilizationConfig.disabled(),
            velocity=VelocityConfig.direct(),
            compute_device="CPU",
            precision="f64",
            max_n_particles=64,
            max_evaluation_points=128,
            freestream_velocity=(freestream_speed, 0.0, 0.0),
            checkpoint_directory=str(case_directory / "solution"),
            log_mode="console",
            verbose=False,
            export_flow_integrals=False,
            export_discretization_health=False,
        ),
        case_dir=case_directory,
    )


def _counter_rotating_external_patch(
    *,
    particle_spacing: float,
    pair_separation: float,
    vortex_strength_magnitude: float,
    freestream_speed: float,
) -> dict[str, np.ndarray]:
    h = particle_spacing
    position = np.array([[0.0, -0.5 * pair_separation, 0.0], [0.0, 0.5 * pair_separation, 0.0]])
    normal = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    velocity = np.tile(np.array([freestream_speed, 0.0, 0.0]), (2, 1))
    external_vorticity = np.array(
        [
            [0.0, 0.0, -vortex_strength_magnitude / h**3],
            [0.0, 0.0, vortex_strength_magnitude / h**3],
        ]
    )
    return {
        "slot_id": np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int64),
        "slot_position": position,
        "slot_normal": normal,
        "patch_area": np.full(2, h**2),
        "vorticity_flux": vorticity_transport_flux(
            velocity,
            external_vorticity,
            normal,
        ),
        "normal_velocity": np.full(2, freestream_speed),
    }


def _gaussian_velocity(
    evaluation_position: np.ndarray,
    source_position: np.ndarray,
    source_strength: np.ndarray,
    core_radius: float,
    freestream_velocity: np.ndarray,
) -> np.ndarray:
    displacement = evaluation_position[:, None, :] - source_position[None, :, :]
    radius_squared = np.einsum("npi,npi->np", displacement, displacement)
    radius = np.sqrt(radius_squared)
    density = radius / core_radius
    q = (erf(density) - 2.0 / np.sqrt(np.pi) * density * np.exp(-(density**2))) / (4.0 * np.pi)
    scale = np.divide(
        q,
        radius_squared * radius,
        out=np.zeros_like(q),
        where=radius_squared > 0.0,
    )
    induced = -np.sum(
        scale[..., None] * np.cross(displacement, source_strength[None, :, :]),
        axis=1,
    )
    return induced + freestream_velocity


def _gaussian_vorticity(
    evaluation_position: np.ndarray,
    source_position: np.ndarray,
    source_strength: np.ndarray,
    core_radius: float,
) -> np.ndarray:
    displacement = evaluation_position[:, None, :] - source_position[None, :, :]
    density_squared = np.einsum("npi,npi->np", displacement, displacement) / core_radius**2
    weight = np.exp(-density_squared) / (np.pi**1.5 * core_radius**3)
    return np.einsum("np,pi->ni", weight, source_strength)


def _blob_position_rate(
    _time: float,
    flattened_position: np.ndarray,
    source_strength: np.ndarray,
    core_radius: float,
    freestream_velocity: np.ndarray,
) -> np.ndarray:
    position = flattened_position.reshape(-1, 3)
    return _gaussian_velocity(
        position,
        position,
        source_strength,
        core_radius,
        freestream_velocity,
    ).reshape(-1)


def test_external_flux_pair_is_injected_then_advected_only_by_the_real_vpm(tmp_path):
    h = 0.25
    pair_separation = 4.0 * h
    vortex_strength_magnitude = 1.0
    freestream_speed = 1.0
    vpm_time_step_size = 0.01
    external_time_step_size = h / freestream_speed
    patch = _counter_rotating_external_patch(
        particle_spacing=h,
        pair_separation=pair_separation,
        vortex_strength_magnitude=vortex_strength_magnitude,
        freestream_speed=freestream_speed,
    )
    handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
    batch = handoff.advance(
        **patch,
        time_step_size=external_time_step_size,
    )
    solver = _make_inviscid_solver(
        tmp_path / "analytic_pair",
        time_step_size=vpm_time_step_size,
        particle_spacing=h,
        freestream_speed=freestream_speed,
    )

    injected = inject_vpm_release_batch(
        solver,
        batch,
        particle_volume=h**3,
        particle_group_id=71,
    )
    assert injected == 2
    assert solver.physics.velocity_override is None
    np.testing.assert_array_equal(solver.particle_position, batch.position)
    np.testing.assert_array_equal(solver.particle_vortex_strength, batch.vortex_strength)
    np.testing.assert_array_equal(solver.particle_core_radius, batch.core_radius)
    np.testing.assert_array_equal(solver.particle_volume, np.full(2, h**3))
    np.testing.assert_array_equal(solver.particle_kinematic_viscosity, np.zeros(2))
    np.testing.assert_allclose(
        solver.particle_vorticity,
        batch.vortex_strength / h**3,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(solver.particle_group_id, np.full(2, 71))

    sample_position = np.array([[0.0, 0.0, 0.0], [0.15, -0.2, 0.05], [-0.1, 0.25, -0.08]])
    expected_velocity = _gaussian_velocity(
        sample_position,
        batch.position,
        batch.vortex_strength,
        h,
        np.array([freestream_speed, 0.0, 0.0]),
    )
    expected_vorticity = _gaussian_vorticity(
        sample_position,
        batch.position,
        batch.vortex_strength,
        h,
    )
    np.testing.assert_allclose(
        solver.compute_velocity_at_points(sample_position),
        expected_velocity,
        rtol=2.0e-6,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        solver.compute_vorticity_at_points(sample_position),
        expected_vorticity,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    initial_strength = solver.particle_vortex_strength.copy()
    for _ in range(10):
        # The external handoff is deliberately not called here.  From this
        # point onward the real VPM owns position, velocity, and time.
        solver.advance(defer_output=True)

    density = pair_separation / h
    q = (math.erf(density) - 2.0 / math.sqrt(math.pi) * density * math.exp(-(density**2))) / (
        4.0 * math.pi
    )
    analytic_pair_speed = freestream_speed + q * vortex_strength_magnitude / pair_separation**2
    elapsed_time = 10 * vpm_time_step_size
    expected_position = batch.position.copy()
    expected_position[:, 0] += analytic_pair_speed * elapsed_time
    np.testing.assert_allclose(
        solver.particle_position,
        expected_position,
        rtol=0.0,
        atol=2.0e-9,
    )
    np.testing.assert_array_equal(solver.particle_vortex_strength, initial_strength)
    assert np.linalg.norm(
        solver.particle_position[1] - solver.particle_position[0]
    ) == pytest.approx(
        pair_separation,
        abs=2.0e-12,
    )
    assert solver.time == pytest.approx(elapsed_time)


def _continuous_injection_error(
    case_directory: Path,
    *,
    transport_ratio: float,
) -> float:
    h = 0.25
    pair_separation = 4.0 * h
    vortex_strength_magnitude = 0.2
    freestream_speed = 1.0
    time_step_size = transport_ratio * h / freestream_speed
    patch = _counter_rotating_external_patch(
        particle_spacing=h,
        pair_separation=pair_separation,
        vortex_strength_magnitude=vortex_strength_magnitude,
        freestream_speed=freestream_speed,
    )
    handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
    solver = _make_inviscid_solver(
        case_directory,
        time_step_size=time_step_size,
        particle_spacing=h,
        freestream_speed=freestream_speed,
    )
    reference_position = np.empty((0, 3), dtype=np.float64)
    reference_strength = np.empty((0, 3), dtype=np.float64)
    freestream_velocity = np.array([freestream_speed, 0.0, 0.0])
    n_intervals = int(round(4.0 / transport_ratio))

    for _ in range(n_intervals):
        solver.advance(defer_output=True)
        if len(reference_position):
            reference_solution = solve_ivp(
                _blob_position_rate,
                (0.0, time_step_size),
                reference_position.reshape(-1),
                args=(reference_strength, h, freestream_velocity),
                method="DOP853",
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            assert reference_solution.success
            reference_position = reference_solution.y[:, -1].reshape(-1, 3)

        batch = handoff.advance(
            **patch,
            time_step_size=time_step_size,
            existing_position=solver.particle_position,
        )
        if len(batch.position):
            assert batch.min_new_new_separation >= h
            assert batch.min_new_existing_separation + 1.0e-14 >= h
            inject_vpm_release_batch(
                solver,
                batch,
                particle_volume=h**3,
                particle_group_id=91,
            )
            reference_position = np.vstack((reference_position, batch.position))
            reference_strength = np.vstack((reference_strength, batch.vortex_strength))

    assert len(solver.particle_position) == 8
    assert len(reference_position) == 8
    np.testing.assert_array_equal(solver.particle_vortex_strength, reference_strength)
    np.testing.assert_array_equal(solver.particle_group_id, np.full(8, 91))
    np.testing.assert_allclose(
        handoff.emitted_vortex_strength_total,
        handoff.outward_flux_total,
        rtol=0.0,
        atol=2.0e-15,
    )
    for slot in handoff.slot_status():
        np.testing.assert_allclose(slot.pending_strength, 0.0, rtol=0.0, atol=2.0e-15)
    assert solver.time == pytest.approx(4.0 * h / freestream_speed)

    return float(
        np.linalg.norm(solver.particle_position - reference_position)
        / np.linalg.norm(reference_position)
    )


def test_continuous_external_injection_converges_to_independent_blob_dynamics(tmp_path):
    errors = np.array(
        [
            _continuous_injection_error(
                tmp_path / f"continuous_{index}",
                transport_ratio=transport_ratio,
            )
            for index, transport_ratio in enumerate((0.5, 0.25, 0.125))
        ]
    )

    # Measured errors are approximately 6.7e-6, 8.5e-7, and 1.1e-7.  The
    # factor-of-eight refinement is the expected third-order VPM advection
    # behavior; the wider factor-six gate leaves room for backend variation.
    assert errors[0] < 1.0e-5
    assert errors[1] < 2.0e-6
    assert errors[2] < 3.0e-7
    assert errors[0] / errors[1] > 6.0
    assert errors[1] / errors[2] > 6.0
