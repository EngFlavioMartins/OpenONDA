"""Regression gates for basis-agnostic projected renewal after GBD."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import CouplerSetup
from source.coupler.renewal_projection import (
    evaluate_sparse_gaussian_vorticity,
    gaussian_velocity_operator,
    gaussian_vorticity_basis,
    gbd_guard_width,
    geometric_renewal_mask,
    project_gbd_renewal_basis,
    solve_sparse_renewal_projection,
)
from source.coupler.vorticity_transfer import VorticityTransfer, apply_projected_gbd_renewal


def _field(
    evaluation_position: np.ndarray,
    particle_position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
) -> np.ndarray:
    return (
        gaussian_vorticity_basis(
            evaluation_position,
            particle_position,
            core_radius,
        )
        @ vortex_strength
    )


def _velocity(
    evaluation_position: np.ndarray,
    particle_position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
) -> np.ndarray:
    return (
        gaussian_velocity_operator(
            evaluation_position,
            particle_position,
            core_radius,
        )
        @ vortex_strength.reshape(-1)
    ).reshape(-1, 3)


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = float(np.linalg.norm(expected))
    error = float(np.linalg.norm(actual - expected))
    return error / scale if scale > 0.0 else error


def _gradient_with_curl(vorticity: np.ndarray) -> np.ndarray:
    """Construct the antisymmetric FVM gradient carrying a prescribed curl."""
    omega = np.asarray(vorticity, dtype=np.float64).reshape(-1, 3)
    gradient = np.zeros((len(omega), 3, 3), dtype=np.float64)
    gradient[:, 1, 2] = 0.5 * omega[:, 0]
    gradient[:, 2, 1] = -0.5 * omega[:, 0]
    gradient[:, 2, 0] = 0.5 * omega[:, 1]
    gradient[:, 0, 2] = -0.5 * omega[:, 1]
    gradient[:, 0, 1] = 0.5 * omega[:, 2]
    gradient[:, 1, 0] = -0.5 * omega[:, 2]
    return gradient


def _box_lattice(bounds: tuple[float, ...], spacing: float) -> np.ndarray:
    axes = [
        np.arange(bounds[index], bounds[index + 1] + 0.25 * spacing, spacing) for index in (0, 2, 4)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def test_post_gbd_geometric_authority_absorbs_lattice_roundoff() -> None:
    h = 0.1
    bounds = np.array([-0.2, 0.2, -0.1, 0.1, -0.1, 0.1])
    position = _box_lattice(tuple(bounds), h)
    position[-1] = bounds[1::2] + np.array([8.4e-17, 5.6e-17, 5.6e-17])

    renewable = geometric_renewal_mask(
        position,
        bounds,
        particle_spacing=h,
    )

    assert np.all(renewable)
    assert not geometric_renewal_mask(
        np.array([[bounds[1] + 1.0e-6, 0.0, 0.0]]),
        bounds,
        particle_spacing=h,
    )[0]


def test_exact_current_basis_never_triggers_blanket_support_births() -> None:
    h = 0.1
    sigma = 1.25 * h
    bounds = (-0.2, 0.2, -0.1, 0.1, -0.1, 0.1)
    position = _box_lattice(bounds, h)
    displacement = position - np.array([-0.02, 0.0, 0.0])
    amplitude = np.exp(-np.einsum("ij,ij->i", displacement, displacement) / 0.08**2)
    strength = h**3 * amplitude[:, None] * np.array([0.8, -0.4, 0.6])
    collocation = position + np.array([0.017, 0.013, 0.009])
    target = _field(
        collocation,
        position,
        strength,
        np.full(len(position), sigma),
    )
    irrelevant_background = _box_lattice(bounds, 0.5 * h)

    result = project_gbd_renewal_basis(
        collocation_position=collocation,
        target_vorticity=target,
        particle_position=position,
        vortex_strength=strength,
        core_radius=sigma,
        renewal_bounds=bounds,
        particle_spacing=h,
        support_candidate_position=irrelevant_background,
        support_core_radius=sigma,
        maximum_vorticity_error=1.0e-10,
    )

    assert not result.used_selective_births
    assert len(result.birth_position) == 0
    np.testing.assert_allclose(result.updated_vortex_strength, strength, rtol=0.0, atol=2.0e-16)
    assert result.projection.vorticity_relative_error < 1.0e-13


def test_missing_physical_support_is_created_selectively() -> None:
    h = 0.1
    sigma = 1.25 * h
    bounds = (-0.2, 0.2, -0.2, 0.2, -0.2, 0.2)
    physical_position = np.array([[0.0, 0.0, 0.0]])
    physical_strength = np.array([[1.0e-3, -0.7e-3, 0.5e-3]])
    collocation = _box_lattice(bounds, h)
    target = _field(
        collocation,
        physical_position,
        physical_strength,
        np.array([sigma]),
    )
    candidates = np.vstack(
        (
            physical_position,
            np.array([[h, 0.0, 0.0], [-h, 0.0, 0.0], [0.0, h, 0.0]]),
        )
    )

    result = project_gbd_renewal_basis(
        collocation_position=collocation,
        target_vorticity=target,
        particle_position=np.empty((0, 3)),
        vortex_strength=np.empty((0, 3)),
        core_radius=sigma,
        renewal_bounds=bounds,
        particle_spacing=h,
        support_candidate_position=candidates,
        support_core_radius=sigma,
        maximum_births=1,
        maximum_vorticity_error=1.0e-12,
    )

    assert result.used_selective_births
    np.testing.assert_array_equal(result.birth_position, physical_position)
    np.testing.assert_allclose(
        result.birth_vortex_strength,
        physical_strength,
        rtol=2.0e-13,
        atol=2.0e-16,
    )
    assert result.projection.vorticity_relative_error < 1.0e-13


def test_gbd_guard_accounts_for_m4_support_and_every_laplacian_stage() -> None:
    assert gbd_guard_width(particle_spacing=0.04, diffusion_substeps=1) == pytest.approx(0.12)
    assert gbd_guard_width(particle_spacing=0.04, diffusion_substeps=4) == pytest.approx(0.24)


def test_sparse_production_projection_recovers_dense_gaussian_state() -> None:
    h = 0.08
    sigma = 1.25 * h
    position = _box_lattice((-0.16, 0.16, -0.16, 0.16, -0.16, 0.16), h)
    displacement = position - np.array([0.02, -0.01, 0.03])
    envelope = np.exp(-np.einsum("ij,ij->i", displacement, displacement) / 0.11**2)
    strength = h**3 * envelope[:, None] * np.array([0.9, -0.6, 0.4])
    target = _field(position, position, strength, np.full(len(position), sigma))

    result = solve_sparse_renewal_projection(
        collocation_position=position,
        target_vorticity=target,
        particle_position=position,
        core_radius=sigma,
        relative_tail_cutoff=1.0e-10,
        relative_tolerance=1.0e-12,
    )
    validation = position + np.array([0.17, 0.13, 0.11]) * h
    sparse_field = evaluate_sparse_gaussian_vorticity(
        validation,
        position,
        result.vortex_strength,
        sigma,
        relative_tail_cutoff=1.0e-10,
    )
    exact_field = _field(validation, position, strength, np.full(len(position), sigma))

    assert result.converged
    assert result.vorticity_relative_error < 2.0e-10
    assert _relative_error(sparse_field, exact_field) < 2.0e-9
    assert result.operator_nonzeros < len(position) ** 2


@dataclass(frozen=True)
class _Snapshot:
    position: np.ndarray
    vortex_strength: np.ndarray
    core_radius: np.ndarray


def _make_gbd_vpm(
    case_dir,
    *,
    h: float,
    time_step_size: float,
    capacity: int,
    threshold: float = 1.0e-12,
):
    from source.solvers.vpm import (
        Backup,
        DirectInduction,
        Numerics,
        ViscousConfig,
        VPMCase,
        VPMSolver,
    )

    return VPMSolver(
        VPMCase(
            directory=case_dir,
            backup=Backup(interval_steps=0),
            numerics=Numerics(
                time_step_size=time_step_size,
                compute_device="CPU",
                precision="f32",
                max_n_particles=capacity,
                domain_bounds=(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0),
                freestream_velocity=(1.0, 0.0, 0.0),
                induction=DirectInduction(),
                viscous=ViscousConfig.gbd(
                    particle_spacing=h,
                    padding=5.0,
                    threshold=threshold,
                    threshold_mode="absolute",
                    kinematic_viscosity=1.0e-3,
                    max_nodes=capacity,
                    core_radius_ratio=1.25,
                ),
                verbose=False,
            ),
        )
    )


def _add_particles(solver, position: np.ndarray, strength: np.ndarray, h: float) -> None:
    count = len(position)
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=strength,
        core_radius=np.full(count, 1.25 * h),
        particle_volume=np.full(count, h**3),
        kinematic_viscosity=np.full(count, 1.0e-3),
    )


def _snapshot(solver) -> _Snapshot:
    return _Snapshot(
        position=solver.particles.position_cpu(use_cache=False).astype(np.float64),
        vortex_strength=solver.particles.vortex_strength_cpu(use_cache=False).astype(np.float64),
        core_radius=solver.particles.core_radius_cpu(use_cache=False).astype(np.float64),
    )


def _make_production_transfer(
    setup: CouplerSetup,
    *,
    h: float,
    fvm_bounds: tuple[float, ...],
    fvm_position: np.ndarray,
) -> VorticityTransfer:
    transfer = VorticityTransfer(
        SimpleNamespace(
            setup=setup,
            kinematic_viscosity=1.0e-3,
            fvm_box=np.asarray(fvm_bounds),
            vpm_core_radius_ratio=1.25,
            vpm_particle_spacing=h,
        )
    )
    fvm = SimpleNamespace(
        setup=SimpleNamespace(boundaries=()),
        ibm=None,
        get_cell_centre_coordinates=lambda: fvm_position,
        get_cell_volume=lambda: np.full(len(fvm_position), h**3),
    )
    transfer.setup(fvm)
    return transfer


def test_production_transfer_uses_post_gbd_geometry_and_preserves_outer_tail(tmp_path) -> None:
    h = 0.08
    sigma = 1.25 * h
    fit_bounds = (-0.08, 0.08, -0.08, 0.08, -0.08, 0.08)
    fvm_bounds = (-0.48, 0.48, -0.48, 0.48, -0.48, 0.48)
    fvm_position = _box_lattice(fvm_bounds, h)
    renewable_position = _box_lattice(
        (-0.16, 0.16, -0.16, 0.16, -0.16, 0.16),
        h,
    )
    displacement = renewable_position - np.array([0.01, -0.02, 0.03])
    envelope = np.exp(-np.einsum("ij,ij->i", displacement, displacement) / 0.11**2)
    renewable_strength = h**3 * envelope[:, None] * np.array([0.8, -0.6, 0.5])
    preserved_position = np.array([[0.40, 0.0, 0.0]])
    preserved_strength = np.array([[1.5e-4, -0.8e-4, 0.6e-4]])
    particle_position = np.vstack((renewable_position, preserved_position))
    particle_strength = np.vstack((renewable_strength, preserved_strength))

    solver = _make_gbd_vpm(
        tmp_path / "production_transfer",
        h=h,
        time_step_size=0.317 * h,
        capacity=4000,
    )
    solver.physics.configure_grid_lattice_anchor(fvm_position[0], h)
    _add_particles(solver, particle_position, particle_strength, h)
    setup = CouplerSetup(
        freestream_velocity=[1.0, 0.0, 0.0],
        transfer_method="projected_renewal",
        transfer_region_bounds=fit_bounds,
        eta_blend_width=0.0,
        renewal_vorticity_error_limit=1.0e-7,
        renewal_velocity_error_limit=1.0e-7,
        renewal_gaussian_tail_cutoff=1.0e-10,
        # GBD is intentionally stored in f32; use a solver tolerance that is
        # meaningful for the migrated fixture's storage precision.
        renewal_solver_tolerance=1.0e-8,
    )
    transfer = _make_production_transfer(
        setup,
        h=h,
        fvm_bounds=fvm_bounds,
        fvm_position=fvm_position,
    )

    target_vorticity = evaluate_sparse_gaussian_vorticity(
        fvm_position,
        particle_position,
        particle_strength,
        np.full(len(particle_position), sigma),
        relative_tail_cutoff=setup.renewal_gaussian_tail_cutoff,
    )
    target_velocity = np.asarray(
        solver.compute_velocity_at_points(
            fvm_position,
            include_freestream=True,
            include_body=True,
        ),
        dtype=np.float64,
    )
    before = _snapshot(solver)

    result = transfer.transfer(
        solver,
        target_velocity,
        _gradient_with_curl(target_vorticity),
    )
    after = _snapshot(solver)

    assert result.transfer_method == "projected_gbd_renewal"
    assert result.n_particles_injected == 0
    assert result.renewal_diffusion_substeps == 1
    assert result.renewal_guard_width == pytest.approx(3.0 * h)
    assert result.projection_vorticity_relative_error < 1.0e-7
    assert result.projection_velocity_relative_error is not None
    assert result.projection_velocity_relative_error < 1.0e-7
    np.testing.assert_allclose(
        after.vortex_strength[-1], preserved_strength[0], rtol=0.0, atol=1.0e-11
    )
    assert _relative_error(after.vortex_strength, before.vortex_strength) < 1.0e-8
    assert not solver._is_particle_regeneration_pending
    solver.reset_gpu()


def test_production_transfer_injects_oblique_vorticity_and_releases_it_to_vpm(tmp_path) -> None:
    h = 0.08
    sigma = 1.25 * h
    fit_bounds = (-0.04, 0.04, -0.04, 0.04, -0.04, 0.04)
    fvm_bounds = (-0.40, 0.40, -0.40, 0.40, -0.40, 0.40)
    fvm_position = _box_lattice(fvm_bounds, h)
    source_position = np.array([[0.0, 0.0, 0.0]])
    source_strength = np.array([[8.0e-4, -6.0e-4, 5.0e-4]])
    threshold = 0.02 * h**3
    target_vorticity = evaluate_sparse_gaussian_vorticity(
        fvm_position,
        source_position,
        source_strength,
        np.array([sigma]),
        relative_tail_cutoff=1.0e-10,
    )
    target_velocity = _velocity(
        fvm_position,
        source_position,
        source_strength,
        np.array([sigma]),
    ) + np.array([1.0, 0.0, 0.0])

    solver = _make_gbd_vpm(
        tmp_path / "production_injection",
        h=h,
        time_step_size=0.317 * h,
        capacity=4000,
        threshold=threshold,
    )
    solver.physics.configure_grid_lattice_anchor(fvm_position[0], h)
    setup = CouplerSetup(
        freestream_velocity=[1.0, 0.0, 0.0],
        transfer_method="projected_renewal",
        transfer_region_bounds=fit_bounds,
        eta_blend_width=0.0,
        renewal_vorticity_error_limit=5.0e-3,
        renewal_velocity_error_limit=1.0e-3,
        renewal_gaussian_tail_cutoff=1.0e-10,
        renewal_solver_tolerance=1.0e-8,
    )
    transfer = _make_production_transfer(
        setup,
        h=h,
        fvm_bounds=fvm_bounds,
        fvm_position=fvm_position,
    )

    result = transfer.transfer(
        solver,
        target_velocity,
        _gradient_with_curl(target_vorticity),
    )
    injected = _snapshot(solver)
    actual_vorticity = evaluate_sparse_gaussian_vorticity(
        fvm_position,
        injected.position,
        injected.vortex_strength,
        injected.core_radius,
        relative_tail_cutoff=setup.renewal_gaussian_tail_cutoff,
    )
    magnitude = np.linalg.norm(target_vorticity, axis=1)
    meaningful = magnitude >= 0.1 * magnitude.max(initial=0.0)
    actual_magnitude = np.linalg.norm(actual_vorticity, axis=1)
    cosine = np.einsum(
        "ij,ij->i",
        actual_vorticity[meaningful],
        target_vorticity[meaningful],
    ) / (actual_magnitude[meaningful] * magnitude[meaningful])
    direction_error = float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))).max(initial=0.0))

    assert result.n_particles_before == 0
    assert result.n_particles_injected > 0
    assert result.projection_vorticity_relative_error < 5.0e-3
    assert result.projection_velocity_relative_error is not None
    assert result.projection_velocity_relative_error < 1.0e-3
    assert direction_error < 0.5
    assert _relative_error(actual_vorticity, target_vorticity) < 5.0e-3
    assert not solver._is_particle_regeneration_pending

    weights_before = np.linalg.norm(injected.vortex_strength, axis=1)
    centroid_before = float(np.average(injected.position[:, 0], weights=weights_before))
    solver.advance(defer_output=True)
    advected = _snapshot(solver)
    weights_after = np.linalg.norm(advected.vortex_strength, axis=1)
    centroid_after = float(np.average(advected.position[:, 0], weights=weights_after))
    assert centroid_after > centroid_before + 0.25 * solver.time_step_size
    solver.reset_gpu()


def test_real_gbd_lifecycle_adds_negligible_renewal_error(tmp_path) -> None:
    """Pure GBD and GBD+renewal remain the same physical field through passage."""
    h = 0.08
    time_step_size = 0.317 * h
    events = 12
    bounds = (-0.24, 0.16, -0.16, 0.16, -0.16, 0.16)
    position = _box_lattice(bounds, h)
    displacement = position - np.array([-0.08, 0.0, 0.0])
    envelope = np.exp(-np.einsum("ij,ij->i", displacement, displacement) / 0.09**2)
    initial_strength = h**3 * envelope[:, None] * np.array([0.9, 0.7, 0.5])
    fit_position = position + np.array([0.325, 0.275, 0.225]) * h
    boundary_position = _box_lattice(
        (bounds[1], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]),
        h,
    )
    validation_position = _box_lattice(
        (bounds[0], bounds[1] + 5.0 * h, -0.24, 0.24, -0.24, 0.24),
        h,
    )

    reference_solver = _make_gbd_vpm(
        tmp_path / "reference",
        h=h,
        time_step_size=time_step_size,
        capacity=4000,
    )
    reference_solver.physics.configure_grid_lattice_anchor(position[0], h)
    _add_particles(reference_solver, position, initial_strength, h)
    reference = [_snapshot(reference_solver)]
    for _event in range(events):
        reference_solver.advance(defer_output=True)
        reference.append(_snapshot(reference_solver))
    expected_substeps, _diffusion_number = (
        reference_solver.physics._explicit_diffusion_substep_count(
            1.0e-3,
            time_step_size,
            h,
        )
    )
    assert reference_solver.physics.last_gbd_diffusion_substeps == expected_substeps
    reference_solver.reset_gpu()

    renewal_solver = _make_gbd_vpm(
        tmp_path / "renewal",
        h=h,
        time_step_size=time_step_size,
        capacity=4000,
    )
    renewal_solver.physics.configure_grid_lattice_anchor(position[0], h)
    _add_particles(renewal_solver, position, np.zeros_like(initial_strength), h)

    maximum_vorticity_error = 0.0
    maximum_velocity_error = 0.0
    maximum_direction_error = 0.0
    maximum_particle_count = 0
    observed_preserved = False
    for event, target in enumerate(reference):
        if event:
            renewal_solver.advance(defer_output=True)
        current = _snapshot(renewal_solver)
        target_vorticity = _field(
            fit_position,
            target.position,
            target.vortex_strength,
            target.core_radius,
        )
        target_boundary_velocity = _velocity(
            boundary_position,
            target.position,
            target.vortex_strength,
            target.core_radius,
        )
        omega_scale = float(np.sqrt(np.mean(target_vorticity**2)))
        velocity_scale = float(np.sqrt(np.mean(target_boundary_velocity**2)))
        velocity_weight = (
            np.sqrt(target_vorticity.size / target_boundary_velocity.size)
            * omega_scale
            / max(velocity_scale, 1.0e-30)
        )
        projected = project_gbd_renewal_basis(
            collocation_position=fit_position,
            target_vorticity=target_vorticity,
            particle_position=current.position,
            vortex_strength=current.vortex_strength,
            core_radius=current.core_radius,
            renewal_bounds=bounds,
            particle_spacing=h,
            velocity_position=boundary_position,
            target_velocity=target_boundary_velocity,
            velocity_weight=velocity_weight,
            # Independent GBD replays use a production f32 atomic scatter.
            # Their run-order noise can exceed float64 roundoff while remaining
            # orders of magnitude below the physical transfer budget.
            maximum_vorticity_error=1.0e-7,
            maximum_velocity_error=1.0e-7,
        )
        observed_preserved |= bool(np.any(projected.preserved_mask))
        maximum_particle_count = max(maximum_particle_count, len(current.position))
        assert not projected.used_selective_births
        assert len(projected.birth_position) == 0
        preserved_before = current.vortex_strength[projected.preserved_mask].copy()
        transfer = apply_projected_gbd_renewal(
            renewal_solver,
            projected,
            particle_spacing=h,
            kinematic_viscosity=1.0e-3,
        )
        assert transfer.transfer_method == "projected_gbd_renewal"
        assert transfer.n_particles_injected == 0
        assert not renewal_solver._is_particle_regeneration_pending
        after = _snapshot(renewal_solver)
        np.testing.assert_array_equal(
            after.vortex_strength[projected.preserved_mask],
            preserved_before,
        )

        expected_vorticity = _field(
            validation_position,
            target.position,
            target.vortex_strength,
            target.core_radius,
        )
        actual_vorticity = _field(
            validation_position,
            after.position,
            after.vortex_strength,
            after.core_radius,
        )
        expected_velocity = _velocity(
            validation_position,
            target.position,
            target.vortex_strength,
            target.core_radius,
        )
        actual_velocity = _velocity(
            validation_position,
            after.position,
            after.vortex_strength,
            after.core_radius,
        )
        maximum_vorticity_error = max(
            maximum_vorticity_error,
            _relative_error(actual_vorticity, expected_vorticity),
        )
        maximum_velocity_error = max(
            maximum_velocity_error,
            _relative_error(actual_velocity, expected_velocity),
        )
        magnitude = np.linalg.norm(expected_vorticity, axis=1)
        meaningful = magnitude > 0.1 * magnitude.max(initial=0.0)
        actual_magnitude = np.linalg.norm(actual_vorticity, axis=1)
        cosine = np.einsum(
            "ij,ij->i",
            actual_vorticity[meaningful],
            expected_vorticity[meaningful],
        ) / (actual_magnitude[meaningful] * magnitude[meaningful])
        maximum_direction_error = max(
            maximum_direction_error,
            float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))).max(initial=0.0)),
        )

    renewal_solver.reset_gpu()
    print(
        "GBD projected renewal",
        f"omega={maximum_vorticity_error:.3e}",
        f"velocity={maximum_velocity_error:.3e}",
        f"angle_deg={maximum_direction_error:.3e}",
        f"maximum_particles={maximum_particle_count}",
        "births=0",
    )
    assert observed_preserved
    assert maximum_vorticity_error < 1.0e-7
    assert maximum_velocity_error < 1.0e-7
    assert maximum_direction_error < 1.0e-4
