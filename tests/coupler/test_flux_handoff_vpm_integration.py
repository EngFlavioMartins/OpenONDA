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
    Backup,
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
    freestream_velocity: tuple[float, float, float] | np.ndarray,
) -> VPMSolver:
    background = tuple(float(component) for component in freestream_velocity)
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
            freestream_velocity=background,
            backup=Backup(
                interval_steps=0,
                directory=str(case_directory / "solution"),
                log_directory=str(case_directory / "solution"),
            ),
            verbose=False,
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


def _gaussian_vorticity_divergence(
    evaluation_position: np.ndarray,
    source_position: np.ndarray,
    source_strength: np.ndarray,
    core_radius: float,
) -> np.ndarray:
    displacement = evaluation_position[:, None, :] - source_position[None, :, :]
    density_squared = np.einsum("npi,npi->np", displacement, displacement) / core_radius**2
    weight = np.exp(-density_squared) / (np.pi**1.5 * core_radius**3)
    return np.sum(
        -2.0 * np.einsum("npi,pi->np", displacement, source_strength) * weight / core_radius**2,
        axis=1,
    )


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
        freestream_velocity=(freestream_speed, 0.0, 0.0),
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


def test_oblique_external_flux_retains_tangential_phase_before_free_vpm_advection(tmp_path):
    h = 0.25
    pair_separation = 4.0 * h
    vortex_strength_magnitude = 0.2
    normal_speed = 1.0
    tangential_speed = math.sqrt(3.0)  # 60-degree crossing direction
    transport_ratio = 0.317
    external_time_step_size = transport_ratio * h / normal_speed
    transport_velocity = np.array([normal_speed, tangential_speed, 0.0])
    slot_position = np.array(
        [[0.0, 0.0, -0.5 * pair_separation], [0.0, 0.0, 0.5 * pair_separation]]
    )
    slot_normal = np.tile([1.0, 0.0, 0.0], (2, 1))
    external_vorticity = np.array(
        [
            [0.0, vortex_strength_magnitude / h**3, 0.0],
            [0.0, -vortex_strength_magnitude / h**3, 0.0],
        ]
    )
    patch = {
        "slot_id": np.array([[0, 0, 0], [0, 0, 1]]),
        "slot_position": slot_position,
        "slot_normal": slot_normal,
        "patch_area": np.full(2, h**2),
        "vorticity_flux": vorticity_transport_flux(
            np.tile(transport_velocity, (2, 1)),
            external_vorticity,
            slot_normal,
        ),
        "normal_velocity": np.full(2, normal_speed),
        "transport_velocity": np.tile(transport_velocity, (2, 1)),
    }
    handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
    batch = None
    for _ in range(4):
        batch = handoff.advance(**patch, time_step_size=external_time_step_size)
    assert batch is not None
    assert len(batch.position) == 2

    remainder_normal_distance = (4.0 * transport_ratio - 1.0) * h
    remaining_flight_time = remainder_normal_distance / normal_speed
    expected_birth_position = slot_position + remaining_flight_time * transport_velocity
    np.testing.assert_allclose(
        batch.position,
        expected_birth_position,
        rtol=0.0,
        atol=3.0e-15,
    )

    vpm_time_step_size = 0.01
    solver = _make_inviscid_solver(
        tmp_path / "oblique_pair",
        time_step_size=vpm_time_step_size,
        particle_spacing=h,
        freestream_velocity=transport_velocity,
    )
    inject_vpm_release_batch(solver, batch, particle_volume=h**3)
    analytic_velocity = _gaussian_velocity(
        batch.position,
        batch.position,
        batch.vortex_strength,
        h,
        transport_velocity,
    )
    np.testing.assert_allclose(analytic_velocity[0], analytic_velocity[1], atol=2.0e-15)
    initial_strength = solver.particle_vortex_strength.copy()
    for _ in range(10):
        solver.advance(defer_output=True)
    np.testing.assert_allclose(
        solver.particle_position,
        batch.position + 10.0 * vpm_time_step_size * analytic_velocity,
        rtol=0.0,
        atol=3.0e-9,
    )
    np.testing.assert_array_equal(solver.particle_vortex_strength, initial_strength)


def test_closed_vortex_ring_preserves_moments_fields_topology_and_self_advection(tmp_path):
    h = 0.25
    ring_radius = 1.0
    tube_circulation = 0.2
    freestream_speed = 0.5
    n_ring_particles = 24
    azimuth = 2.0 * np.pi * np.arange(n_ring_particles) / n_ring_particles
    source_position = np.column_stack(
        (
            np.zeros(n_ring_particles),
            ring_radius * np.cos(azimuth),
            ring_radius * np.sin(azimuth),
        )
    )
    tangent = np.column_stack(
        (
            np.zeros(n_ring_particles),
            -np.sin(azimuth),
            np.cos(azimuth),
        )
    )
    segment_length = 2.0 * np.pi * ring_radius / n_ring_particles
    source_strength = tube_circulation * segment_length * tangent
    slot_normal = np.tile([1.0, 0.0, 0.0], (n_ring_particles, 1))
    transport_velocity = np.tile(
        [freestream_speed, 0.0, 0.0],
        (n_ring_particles, 1),
    )
    external_vorticity = source_strength / h**3
    handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
    batch = handoff.advance(
        slot_id=np.column_stack(
            (
                np.zeros(n_ring_particles, dtype=np.int64),
                np.arange(n_ring_particles, dtype=np.int64),
                np.zeros(n_ring_particles, dtype=np.int64),
            )
        ),
        slot_position=source_position,
        slot_normal=slot_normal,
        patch_area=np.full(n_ring_particles, h**2),
        vorticity_flux=vorticity_transport_flux(
            transport_velocity,
            external_vorticity,
            slot_normal,
        ),
        normal_velocity=np.full(n_ring_particles, freestream_speed),
        transport_velocity=transport_velocity,
        time_step_size=h / freestream_speed,
    )
    assert len(batch.position) == n_ring_particles
    assert batch.min_new_new_separation >= h
    assert batch.nearest_neighbour_distance_over_spacing.min() > 1.03
    np.testing.assert_allclose(batch.core_radius_over_spacing, 1.0)
    np.testing.assert_allclose(batch.position, source_position, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(batch.vortex_strength, source_strength, rtol=0.0, atol=3.0e-18)
    np.testing.assert_allclose(batch.conservation_error, 0.0, rtol=0.0, atol=2.0e-16)

    def first_vorticity_moment(position, strength):
        return position.T @ strength

    def gaussian_angular_impulse(position, strength):
        return np.cross(position, np.cross(position, strength)).sum(axis=0) / 3.0 - (
            1.0 / 3.0
        ) * h**2 * strength.sum(axis=0)

    np.testing.assert_allclose(
        batch.vortex_strength.sum(axis=0),
        source_strength.sum(axis=0),
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        first_vorticity_moment(batch.position, batch.vortex_strength),
        first_vorticity_moment(source_position, source_strength),
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        gaussian_angular_impulse(batch.position, batch.vortex_strength),
        gaussian_angular_impulse(source_position, source_strength),
        rtol=0.0,
        atol=3.0e-16,
    )

    vpm_time_step_size = 0.005
    solver = _make_inviscid_solver(
        tmp_path / "closed_ring",
        time_step_size=vpm_time_step_size,
        particle_spacing=h,
        freestream_velocity=(freestream_speed, 0.0, 0.0),
    )
    inject_vpm_release_batch(
        solver,
        batch,
        particle_volume=h**3,
        particle_group_id=111,
    )
    normal_line = np.column_stack(
        (
            np.linspace(-3.0 * h, 6.0 * h, 73),
            np.full(73, ring_radius),
            np.zeros(73),
        )
    )
    reference_velocity = _gaussian_velocity(
        normal_line,
        source_position,
        source_strength,
        h,
        np.array([freestream_speed, 0.0, 0.0]),
    )
    reference_vorticity = _gaussian_vorticity(
        normal_line,
        source_position,
        source_strength,
        h,
    )
    reference_divergence = _gaussian_vorticity_divergence(
        normal_line,
        source_position,
        source_strength,
        h,
    )
    np.testing.assert_allclose(
        solver.compute_velocity_at_points(normal_line),
        reference_velocity,
        rtol=6.0e-6,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        solver.compute_vorticity_at_points(normal_line),
        reference_vorticity,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    # The authority switch replaces exactly the same Gaussian particle field;
    # therefore velocity, vorticity, their normal-line gradients, and discrete
    # vorticity divergence have no transfer jump.
    np.testing.assert_allclose(
        _gaussian_velocity(
            normal_line,
            batch.position,
            batch.vortex_strength,
            h,
            np.array([freestream_speed, 0.0, 0.0]),
        ),
        reference_velocity,
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        _gaussian_vorticity(normal_line, batch.position, batch.vortex_strength, h),
        reference_vorticity,
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        _gaussian_vorticity_divergence(
            normal_line,
            batch.position,
            batch.vortex_strength,
            h,
        ),
        reference_divergence,
        rtol=0.0,
        atol=3.0e-16,
    )

    initial_particle_velocity = _gaussian_velocity(
        batch.position,
        batch.position,
        batch.vortex_strength,
        h,
        np.array([freestream_speed, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        initial_particle_velocity[:, 0],
        np.full(n_ring_particles, initial_particle_velocity[:, 0].mean()),
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(initial_particle_velocity[:, 1:], 0.0, atol=3.0e-15)
    initial_strength = solver.particle_vortex_strength.copy()
    for _ in range(10):
        solver.advance(defer_output=True)
    elapsed_time = 10.0 * vpm_time_step_size
    np.testing.assert_allclose(
        solver.particle_position,
        batch.position + elapsed_time * initial_particle_velocity,
        rtol=0.0,
        atol=4.0e-9,
    )
    np.testing.assert_array_equal(solver.particle_vortex_strength, initial_strength)
    final_position = solver.particle_position
    final_centroid = np.average(
        final_position,
        axis=0,
        weights=np.linalg.norm(initial_strength, axis=1),
    )
    final_radius = np.sqrt(
        np.mean((final_position[:, 1] - final_centroid[1]) ** 2 + final_position[:, 2] ** 2)
    )
    assert final_centroid[0] == pytest.approx(
        elapsed_time * initial_particle_velocity[:, 0].mean(),
        abs=4.0e-9,
    )
    assert final_radius == pytest.approx(ring_radius, abs=4.0e-9)
    represented_tube_circulation = np.linalg.norm(initial_strength, axis=1).sum() / (
        2.0 * np.pi * ring_radius
    )
    assert represented_tube_circulation == pytest.approx(tube_circulation, abs=3.0e-16)


def test_native_fvm_manufactured_flux_injects_and_enters_vpm_blob_dynamics(tmp_path):
    from source.solvers.fvm import (
        BoundaryConfig,
        FVMSetup,
        FVMSolver,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh

    h = 0.25
    freestream_speed = 1.0
    shear_rate = 2.0
    kinematic_viscosity = 1.0e-12
    bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    fvm_setup = FVMSetup(
        case_name="manufactured_flux_surface",
        time=TimeConfig(time_step_size=0.01, end_time=0.01),
        transport=TransportConfig(kinematic_viscosity=kinematic_viscosity),
        boundaries=[
            BoundaryConfig(
                name="numericalBoundary",
                velocity_type="fixedValue",
                velocity_value=(freestream_speed, 0.0, 0.0),
                pressure_type="fixedFluxPressure",
            )
        ],
        initial_velocity=(freestream_speed, 0.0, 0.0),
    )
    fvm = FVMSolver(
        fvm_setup,
        case_dir=tmp_path / "native_fvm",
        mesh_data=coupling_box_mesh(bounds, h),
    )
    n_cells = int(fvm.mesh_data["n_cells"])
    cell_centre = np.asarray(fvm.geo_data["cell_centre"], dtype=np.float64)[:n_cells]
    # Manufactured incompressible linear shear:
    # u=(U,0,Sx), omega=(0,-S,0).  It has exact constant vorticity and zero
    # normal vorticity gradient on the internal x=0 release plane.
    fvm.velocity[:n_cells] = np.column_stack(
        (
            np.full(n_cells, freestream_speed),
            np.zeros(n_cells),
            shear_rate * cell_centre[:, 0],
        )
    )
    fvm._invalidate_derived_fields()
    cell_velocity = np.asarray(fvm.get_velocity_field(), dtype=np.float64)
    cell_vorticity = np.asarray(fvm.get_vorticity_field(), dtype=np.float64)

    n_interior_faces = int(fvm.mesh_data["n_interior_faces"])
    face_centre = np.asarray(fvm.geo_data["face_centre"], dtype=np.float64)[:n_interior_faces]
    face_area_vector = np.asarray(fvm.geo_data["face_area_vector"], dtype=np.float64)[
        :n_interior_faces
    ]
    release_face = np.flatnonzero(
        np.isclose(face_centre[:, 0], 0.0, rtol=0.0, atol=1.0e-15)
        & (np.abs(face_area_vector[:, 0]) > 0.99 * h**2)
    )
    assert len(release_face) == 64
    owner = np.asarray(fvm.mesh_data["owners"], dtype=np.int64)[release_face]
    neighbour = np.asarray(fvm.mesh_data["neighbours"], dtype=np.int64)[release_face]
    weight = np.asarray(fvm.geo_data["face_interpolation_weight"], dtype=np.float64)[
        release_face, None
    ]
    release_velocity = weight * cell_velocity[neighbour] + (1.0 - weight) * cell_velocity[owner]
    release_vorticity = weight * cell_vorticity[neighbour] + (1.0 - weight) * cell_vorticity[owner]
    release_area_vector = face_area_vector[release_face]
    release_area = np.linalg.norm(release_area_vector, axis=1)
    release_normal = release_area_vector / release_area[:, None]
    normal_velocity = np.einsum("ij,ij->i", release_velocity, release_normal)
    fvm_flux = vorticity_transport_flux(
        release_velocity,
        release_vorticity,
        release_normal,
        np.zeros_like(release_vorticity),
        kinematic_viscosity=kinematic_viscosity,
    )
    np.testing.assert_allclose(
        release_velocity,
        np.broadcast_to([freestream_speed, 0.0, 0.0], release_velocity.shape),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        release_vorticity,
        np.broadcast_to([0.0, -shear_rate, 0.0], release_vorticity.shape),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        fvm_flux,
        np.broadcast_to(
            [0.0, -freestream_speed * shear_rate, 0.0],
            fvm_flux.shape,
        ),
        rtol=0.0,
        atol=2.0e-14,
    )

    tangent_origin = np.array([bounds[2] + 0.5 * h, bounds[4] + 0.5 * h])
    tangent_id = np.rint((face_centre[release_face, 1:] - tangent_origin) / h).astype(np.int64)
    slot_id = np.column_stack((np.zeros(len(release_face), dtype=np.int64), tangent_id))
    assert len(np.unique(slot_id, axis=0)) == len(slot_id)
    handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
    batch = handoff.advance(
        slot_id=slot_id,
        slot_position=face_centre[release_face],
        slot_normal=release_normal,
        patch_area=release_area,
        vorticity_flux=fvm_flux,
        normal_velocity=normal_velocity,
        transport_velocity=release_velocity,
        time_step_size=h / freestream_speed,
    )
    assert len(batch.position) == 64
    np.testing.assert_allclose(
        batch.vortex_strength,
        np.broadcast_to([0.0, -shear_rate * h**3, 0.0], batch.vortex_strength.shape),
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(batch.conservation_error, 0.0, rtol=0.0, atol=2.0e-15)

    vpm_time_step_size = 0.002
    vpm = _make_inviscid_solver(
        tmp_path / "native_fvm_to_vpm",
        time_step_size=vpm_time_step_size,
        particle_spacing=h,
        freestream_velocity=(freestream_speed, 0.0, 0.0),
    )
    inject_vpm_release_batch(
        vpm,
        batch,
        particle_volume=h**3,
        kinematic_viscosity=kinematic_viscosity,
        particle_group_id=121,
    )
    reference = solve_ivp(
        _blob_position_rate,
        (0.0, vpm_time_step_size),
        batch.position.reshape(-1),
        args=(batch.vortex_strength, h, np.array([freestream_speed, 0.0, 0.0])),
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert reference.success
    vpm.advance(defer_output=True)
    np.testing.assert_allclose(
        vpm.particle_position,
        reference.y[:, -1].reshape(-1, 3),
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_array_equal(vpm.particle_vortex_strength, batch.vortex_strength)


def test_injected_gaussian_blob_advects_and_diffuses_with_native_core_spreading(tmp_path):
    h = 0.25
    vortex_strength = np.array([[0.0, 0.2, 0.0]])
    freestream_speed = 0.75
    kinematic_viscosity = 0.02
    handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
    normal = np.array([[1.0, 0.0, 0.0]])
    velocity = np.array([[freestream_speed, 0.0, 0.0]])
    batch = handoff.advance(
        slot_id=np.array([[0, 0, 0]]),
        slot_position=np.zeros((1, 3)),
        slot_normal=normal,
        patch_area=np.array([h**2]),
        vorticity_flux=vorticity_transport_flux(
            velocity,
            vortex_strength / h**3,
            normal,
            np.zeros((1, 3)),
            kinematic_viscosity=kinematic_viscosity,
        ),
        normal_velocity=np.array([freestream_speed]),
        transport_velocity=velocity,
        time_step_size=h / freestream_speed,
    )
    np.testing.assert_allclose(batch.vortex_strength, vortex_strength, rtol=0.0, atol=2.0e-17)

    vpm_time_step_size = 0.01
    solver = VPMSolver(
        VPMSetup(
            time_step_size=vpm_time_step_size,
            advection=AdvectionConfig("RK3"),
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig.cs(
                kinematic_viscosity=kinematic_viscosity,
                particle_spacing=h,
            ),
            turbulence=TurbulenceConfig.dns(),
            stabilization=StabilizationConfig.disabled(),
            velocity=VelocityConfig.direct(),
            compute_device="CPU",
            precision="f64",
            max_n_particles=16,
            max_evaluation_points=16,
            freestream_velocity=(freestream_speed, 0.0, 0.0),
            backup=Backup(
                interval_steps=0,
                directory=str(tmp_path / "diffusive_blob" / "solution"),
                log_directory=str(tmp_path / "diffusive_blob" / "solution"),
            ),
            verbose=False,
        ),
        case_dir=tmp_path / "diffusive_blob",
    )
    inject_vpm_release_batch(
        solver,
        batch,
        particle_volume=h**3,
        kinematic_viscosity=kinematic_viscosity,
    )
    n_steps = 20
    for _ in range(n_steps):
        solver.advance(defer_output=True)
    elapsed_time = n_steps * vpm_time_step_size
    expected_core_radius = math.sqrt(h**2 + 4.0 * kinematic_viscosity * elapsed_time)
    np.testing.assert_allclose(
        solver.particle_position,
        [[freestream_speed * elapsed_time, 0.0, 0.0]],
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        solver.particle_core_radius,
        [expected_core_radius],
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_array_equal(solver.particle_vortex_strength, vortex_strength)
    centre_vorticity = solver.compute_vorticity_at_points(solver.particle_position)
    expected_centre_vorticity = vortex_strength / (np.pi**1.5 * expected_core_radius**3)
    np.testing.assert_allclose(
        centre_vorticity,
        expected_centre_vorticity,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_injected_three_dimensional_cloud_matches_pure_vpm_stretching_reference(tmp_path):
    h = 0.5
    freestream_speed = 1.0
    source_position = np.array(
        [
            [0.0, -0.5, -0.5],
            [0.0, 0.5, -0.5],
            [0.0, -0.5, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    source_strength = 0.2 * np.array(
        [
            [0.0, 1.0, 0.2],
            [0.0, -0.3, 1.0],
            [0.0, -0.8, -0.2],
            [0.0, 0.4, -1.0],
        ]
    )
    normal = np.tile([1.0, 0.0, 0.0], (4, 1))
    transport_velocity = np.tile([freestream_speed, 0.0, 0.0], (4, 1))
    handoff = FluxReleaseHandoff(particle_spacing=h, core_radius=h)
    batch = handoff.advance(
        slot_id=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]),
        slot_position=source_position,
        slot_normal=normal,
        patch_area=np.full(4, h**2),
        vorticity_flux=vorticity_transport_flux(
            transport_velocity,
            source_strength / h**3,
            normal,
        ),
        normal_velocity=np.full(4, freestream_speed),
        transport_velocity=transport_velocity,
        time_step_size=h / freestream_speed,
    )
    assert len(batch.position) == 4

    def make_stretching_solver(case_name: str) -> VPMSolver:
        case_directory = tmp_path / case_name
        return VPMSolver(
            VPMSetup(
                time_step_size=0.005,
                time_integration="COUPLED",
                advection=AdvectionConfig("RK3"),
                stretching=StretchingConfig.direct("RK3"),
                viscous=ViscousConfig.inviscid(particle_spacing=h),
                turbulence=TurbulenceConfig.inviscid(),
                stabilization=StabilizationConfig.disabled(),
                velocity=VelocityConfig.direct(),
                compute_device="CPU",
                precision="f64",
                max_n_particles=16,
                max_evaluation_points=16,
                freestream_velocity=(freestream_speed, 0.0, 0.0),
                backup=Backup(
                    interval_steps=0,
                    directory=str(case_directory / "solution"),
                    log_directory=str(case_directory / "solution"),
                ),
                verbose=False,
            ),
            case_dir=case_directory,
        )

    injected_solver = make_stretching_solver("stretching_injected")
    reference_solver = make_stretching_solver("stretching_reference")
    inject_vpm_release_batch(
        injected_solver,
        batch,
        particle_volume=h**3,
        particle_group_id=131,
    )
    reference_solver.add_vortex_particles(
        position=batch.position,
        velocity=np.zeros_like(batch.position),
        vortex_strength=batch.vortex_strength,
        core_radius=batch.core_radius,
        particle_volume=np.full(4, h**3),
        kinematic_viscosity=np.zeros(4),
        group_id=np.full(4, 131, dtype=np.int32),
    )
    initial_strength = batch.vortex_strength.copy()
    for _ in range(10):
        injected_solver.advance(defer_output=True)
        reference_solver.advance(defer_output=True)
    np.testing.assert_array_equal(
        injected_solver.particle_position,
        reference_solver.particle_position,
    )
    np.testing.assert_array_equal(
        injected_solver.particle_vortex_strength,
        reference_solver.particle_vortex_strength,
    )
    assert np.linalg.norm(injected_solver.particle_vortex_strength - initial_strength) > 1.0e-4
    # F1 is no longer called after emission.  Physical stretching belongs to
    # VPM evolution and cannot be miscounted as new external circulation.
    np.testing.assert_allclose(
        handoff.emitted_vortex_strength_total,
        handoff.outward_flux_total,
        rtol=0.0,
        atol=2.0e-16,
    )


def _continuous_injection_error(
    case_directory: Path,
    *,
    transport_ratio: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
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
        freestream_velocity=(freestream_speed, 0.0, 0.0),
    )
    reference_position = np.empty((0, 3), dtype=np.float64)
    reference_strength = np.empty((0, 3), dtype=np.float64)
    freestream_velocity = np.array([freestream_speed, 0.0, 0.0])
    n_intervals = int(round(4.0 / transport_ratio))
    maximum_pending_strength = 0.0

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
        maximum_pending_strength = max(
            maximum_pending_strength,
            *(np.linalg.norm(slot.pending_strength) for slot in handoff.slot_status()),
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

    relative_error = float(
        np.linalg.norm(solver.particle_position - reference_position)
        / np.linalg.norm(reference_position)
    )
    final_position = solver.particle_position
    pair_distance = np.linalg.norm(
        final_position[:, None, :] - final_position[None, :, :],
        axis=2,
    )
    np.fill_diagonal(pair_distance, np.inf)
    nearest_neighbour_ratio = pair_distance.min(axis=1) / h
    separation_statistics = np.array(
        [
            nearest_neighbour_ratio.min(),
            np.percentile(nearest_neighbour_ratio, 1.0),
            np.percentile(nearest_neighbour_ratio, 5.0),
            np.median(nearest_neighbour_ratio),
        ]
    )
    check_x = float(np.median(reference_position[:, 0]))
    check_y, check_z = np.meshgrid(
        np.linspace(-0.75, 0.75, 9),
        np.linspace(-0.5, 0.5, 9),
        indexing="ij",
    )
    check_position = np.column_stack(
        (
            np.full(check_y.size, check_x),
            check_y.ravel(),
            check_z.ravel(),
        )
    )
    reference_velocity = _gaussian_velocity(
        check_position,
        reference_position,
        reference_strength,
        h,
        freestream_velocity,
    )
    reference_vorticity = _gaussian_vorticity(
        check_position,
        reference_position,
        reference_strength,
        h,
    )
    reference_divergence = _gaussian_vorticity_divergence(
        check_position,
        reference_position,
        reference_strength,
        h,
    )
    actual_velocity = solver.compute_velocity_at_points(check_position)
    actual_vorticity = solver.compute_vorticity_at_points(check_position)
    actual_divergence = _gaussian_vorticity_divergence(
        check_position,
        solver.particle_position,
        solver.particle_vortex_strength,
        h,
    )

    def relative_rms(actual: np.ndarray, expected: np.ndarray) -> float:
        return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))

    field_errors = np.array(
        [
            relative_rms(actual_velocity, reference_velocity),
            relative_rms(actual_vorticity, reference_vorticity),
            relative_rms(actual_divergence, reference_divergence),
        ]
    )
    return relative_error, separation_statistics, field_errors, maximum_pending_strength


def test_continuous_external_injection_converges_to_independent_blob_dynamics(tmp_path):
    result = [
        _continuous_injection_error(
            tmp_path / f"continuous_{index}",
            transport_ratio=transport_ratio,
        )
        for index, transport_ratio in enumerate((0.5, 0.25, 0.125))
    ]
    errors = np.array([case[0] for case in result])
    separation_statistics = np.vstack([case[1] for case in result])
    field_errors = np.vstack([case[2] for case in result])
    maximum_pending_strength = np.array([case[3] for case in result])

    # Measured errors are approximately 6.7e-6, 8.5e-7, and 1.1e-7.  The
    # factor-of-eight refinement is the expected third-order VPM advection
    # behavior; the wider factor-six gate leaves room for backend variation.
    assert errors[0] < 1.0e-5
    assert errors[1] < 2.0e-6
    assert errors[2] < 3.0e-7
    assert errors[0] / errors[1] > 6.0
    assert errors[1] / errors[2] > 6.0
    # Columns are minimum, 1st percentile, 5th percentile, and median d_nn/h.
    # Any later compression is VPM dynamics, but this short evolving wake stays
    # comfortably above the release-spacing scale at every refinement.
    assert np.all(separation_statistics[:, 0] > 0.95)
    # On a downstream check plane, compare the actual VPM fields against an
    # independently integrated pure Gaussian-blob reference.  Columns are
    # relative RMS errors in velocity, vorticity, and div(omega).
    assert np.all(field_errors[:, 0] < 5.0e-3)
    assert np.all(field_errors[:, 1] < 1.0e-2)
    assert np.all(field_errors[:, 2] < 1.0e-2)
    assert np.all(field_errors[1:] < field_errors[:-1])
    expected_maximum_pending = 0.2 * (1.0 - np.array([0.5, 0.25, 0.125]))
    np.testing.assert_allclose(
        maximum_pending_strength,
        expected_maximum_pending,
        rtol=0.0,
        atol=3.0e-16,
    )
