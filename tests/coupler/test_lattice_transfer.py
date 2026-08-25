"""Numerical contracts for conservative FVM-cell to VPM-lattice transfer."""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler.lattice_transfer import (
    blend_fvm_vpm_circulation_on_lattice,
    correct_state_blend_cross_divergence,
    evaluate_gaussian_vorticity,
    first_vorticity_moment,
    m4_prime,
    map_cell_circulation_to_lattice,
    map_vortex_strength_to_lattice,
    spectral_lattice_divergence,
    state_blend_weight,
)


def _uniform_donors(*, shift: np.ndarray, dtype=np.float64):
    h = 0.25
    axis = -1.0 + h * np.arange(9, dtype=np.float64)
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    position = np.column_stack([component.ravel() for component in mesh]) + shift
    volume = np.full(len(position), h**3, dtype=dtype)
    vorticity = np.tile(np.array([1.5, -2.0, 0.25], dtype=dtype), (len(position), 1))
    return position.astype(dtype), volume, vorticity, h


def test_m4_prime_has_partition_and_first_moment_on_its_complete_support():
    fractions = np.array([-0.875, -0.5, -0.125, 0.0, 0.125, 0.5, 0.875])
    for fraction in fractions:
        nodes = np.arange(np.floor(fraction) - 1, np.floor(fraction) + 3)
        weights = m4_prime(fraction - nodes)
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=0.0, atol=2.0e-15)
        np.testing.assert_allclose((nodes * weights).sum(), fraction, rtol=0.0, atol=2.0e-15)


def test_gaussian_vorticity_reference_matches_analytic_single_and_multi_particle_sums():
    points = np.array([[0.0, 0.0, 0.0], [0.25, -0.5, 0.125], [-0.1, 0.2, 0.3]])
    position = np.array([[0.1, -0.2, 0.05], [-0.25, 0.3, -0.1]])
    strength = np.array([[0.2, -0.4, 0.1], [-0.3, 0.5, 0.7]])
    radius = np.array([0.125, 0.35])

    displacement = points[:, None, :] - position[None, :, :]
    sigma = radius[None, :, None]
    analytic_weight = (
        np.pi ** (-1.5) * np.exp(-np.sum((displacement / sigma) ** 2, axis=2)) / sigma[..., 0] ** 3
    )
    expected = analytic_weight @ strength
    actual = evaluate_gaussian_vorticity(points, position, strength, radius)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)

    single = evaluate_gaussian_vorticity(points, position[:1], strength[:1], radius[:1])
    np.testing.assert_allclose(single, analytic_weight[:, :1] @ strength[:1], atol=2.0e-15)


def test_m4_prime_reproduces_quadratic_moments_for_ten_thousand_random_phases():
    rng = np.random.default_rng(20260825)
    phase = rng.uniform(-1.0e4, 1.0e4, size=10_000)
    nodes = np.floor(phase)[:, None] + np.arange(-1, 3)[None, :]
    weights = m4_prime(phase[:, None] - nodes)
    scale = 512.0 * np.finfo(np.float64).eps * (1.0 + np.abs(phase) ** 2)
    assert np.all(np.abs(weights.sum(axis=1) - 1.0) <= scale)
    assert np.all(np.abs((nodes * weights).sum(axis=1) - phase) <= scale)
    assert np.all(np.abs((nodes**2 * weights).sum(axis=1) - phase**2) <= scale)


def test_tensor_m4_prime_reproduces_3d_moments_for_random_phases():
    rng = np.random.default_rng(63)
    phase = rng.uniform(-3.0, 3.0, size=(10_000, 3))
    nodes = np.floor(phase)[:, :, None] + np.arange(-1, 3)[None, None, :]
    axis_weight = m4_prime(phase[:, :, None] - nodes)
    weight = np.einsum("ni,nj,nk->nijk", axis_weight[:, 0], axis_weight[:, 1], axis_weight[:, 2])
    x, y, z = (nodes[:, axis] for axis in range(3))
    np.testing.assert_allclose(weight.sum(axis=(1, 2, 3)), 1.0, rtol=0.0, atol=5.0e-15)
    for axis, coordinate in enumerate((x, y, z)):
        target = np.expand_dims(
            coordinate,
            tuple(other for other in range(1, 4) if other != axis + 1),
        )
        # The insertion above leaves one coordinate axis; broadcasting it over
        # the two remaining stencil dimensions tests the tensor product itself.
        reconstructed = (weight * target).sum(axis=(1, 2, 3))
        np.testing.assert_allclose(reconstructed, phase[:, axis], rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        (weight * x[:, :, None, None] ** 2).sum(axis=(1, 2, 3)),
        phase[:, 0] ** 2,
        rtol=0.0,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        (weight * x[:, :, None, None] * y[:, None, :, None]).sum(axis=(1, 2, 3)),
        phase[:, 0] * phase[:, 1],
        rtol=0.0,
        atol=3.0e-14,
    )


def test_coupler_m4_prime_matches_the_grid_diffusion_kernel():
    pytest.importorskip("taichi", reason="grid diffusion requires taichi")
    from source.solvers.vpm.physics.diffusion.grid import _m4_prime_1d

    rng = np.random.default_rng(116)
    distance = rng.uniform(-2.5, 2.5, size=10_000)
    np.testing.assert_allclose(m4_prime(distance), _m4_prime_1d(distance), rtol=0.0, atol=0.0)


def _numerical_eta_gradient(point: np.ndarray, *, step: float) -> np.ndarray:
    gradient = np.empty(3)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        gradient[axis] = (
            state_blend_weight((point + offset)[None], _UNIT_BOX, _BLEND_WIDTH)[0]
            - state_blend_weight((point - offset)[None], _UNIT_BOX, _BLEND_WIDTH)[0]
        ) / (2.0 * step)
    return gradient


_UNIT_BOX = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_BLEND_WIDTH = 0.4


def _assert_eta_gradient_jump_vanishes(base: np.ndarray, direction: np.ndarray) -> None:
    """Check the normal-gradient jump across a box-distance bisector."""
    coarse_offset, fine_offset = 2.0e-5, 1.0e-5
    coarse_step, fine_step = 2.0e-6, 1.0e-6
    coarse_jump = np.linalg.norm(
        _numerical_eta_gradient(base + coarse_offset * direction, step=coarse_step)
        - _numerical_eta_gradient(base - coarse_offset * direction, step=coarse_step)
    )
    fine_jump = np.linalg.norm(
        _numerical_eta_gradient(base + fine_offset * direction, step=fine_step)
        - _numerical_eta_gradient(base - fine_offset * direction, step=fine_step)
    )
    assert fine_jump < 0.75 * coarse_jump
    assert fine_jump < 2.0e-3


def test_eta_is_c1_across_face_bisectors():
    _assert_eta_gradient_jump_vanishes(
        np.array([-0.8, -0.8, 0.0]),
        np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
    )


def test_eta_is_c1_across_edge_bisectors():
    _assert_eta_gradient_jump_vanishes(
        np.array([-0.8, -0.8, 0.0]),
        np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0),
    )


def test_eta_is_c1_near_box_corners():
    _assert_eta_gradient_jump_vanishes(
        np.array([-0.8, -0.8, -0.8]),
        np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
    )


@pytest.mark.parametrize("dtype,atol", [(np.float64, 5.0e-14), (np.float32, 3.0e-7)])
def test_map_conserves_componentwise_gamma_and_first_moments(dtype, atol):
    rng = np.random.default_rng(43)
    position = rng.uniform(-0.35, 0.45, size=(43, 3)).astype(dtype)
    volume = rng.uniform(0.001, 0.02, size=43).astype(dtype)
    vorticity = rng.normal(size=(43, 3)).astype(dtype)
    mapped = map_cell_circulation_to_lattice(
        position,
        volume,
        vorticity,
        lattice_anchor=np.array([-0.5, 0.125, -0.25]),
        spacing=0.125,
    )
    np.testing.assert_allclose(mapped.target_gamma_net, mapped.donor_gamma_net, rtol=0.0, atol=atol)
    np.testing.assert_allclose(
        mapped.target_first_moment,
        mapped.donor_first_moment,
        rtol=0.0,
        atol=2.0 * atol,
    )
    assert np.isfinite(mapped.vortex_strength).all()
    assert len(np.unique(mapped.position, axis=0)) == len(mapped.position)


def test_constant_vorticity_is_reproduced_for_a_phase_shifted_uniform_cell_grid():
    position, volume, vorticity, h = _uniform_donors(shift=np.array([0.125, -0.125, 0.125]))
    mapped = map_cell_circulation_to_lattice(
        position, volume, vorticity, lattice_anchor=np.zeros(3), spacing=h
    )
    # Exclude the two M4' support layers adjacent to the finite donor patch.
    lower = position.min(axis=0) + 2.0 * h
    upper = position.max(axis=0) - 2.0 * h
    interior = np.all((mapped.position >= lower) & (mapped.position <= upper), axis=1)
    assert interior.any()
    expected = h**3 * vorticity[0]
    np.testing.assert_allclose(
        mapped.vortex_strength[interior],
        np.tile(expected, (int(interior.sum()), 1)),
        rtol=0.0,
        atol=5.0e-15,
    )


def test_translation_phase_does_not_change_gamma_or_first_moment():
    position, volume, vorticity, h = _uniform_donors(shift=np.zeros(3))
    initial = map_cell_circulation_to_lattice(
        position, volume, vorticity, lattice_anchor=np.zeros(3), spacing=h
    )
    phase = np.array([0.073, -0.091, 0.037])
    shifted = map_cell_circulation_to_lattice(
        position + phase,
        volume,
        vorticity,
        lattice_anchor=np.zeros(3),
        spacing=h,
    )
    np.testing.assert_allclose(shifted.target_gamma_net, initial.target_gamma_net, atol=2.0e-14)
    expected_moment_shift = np.outer(phase, initial.target_gamma_net)
    np.testing.assert_allclose(
        shifted.target_first_moment - initial.target_first_moment,
        expected_moment_shift,
        rtol=0.0,
        atol=3.0e-14,
    )


def test_particle_scatter_conserves_vortex_strength_and_first_moment():
    rng = np.random.default_rng(19)
    position = rng.uniform(-0.8, 0.9, size=(31, 3))
    vortex_strength = rng.normal(size=(31, 3))
    mapped = map_vortex_strength_to_lattice(
        position,
        vortex_strength,
        lattice_anchor=np.array([0.125, -0.25, 0.375]),
        spacing=0.25,
    )
    np.testing.assert_allclose(mapped.target_gamma_net, mapped.donor_gamma_net, atol=2.0e-14)
    np.testing.assert_allclose(
        mapped.target_first_moment,
        mapped.donor_first_moment,
        atol=3.0e-14,
    )


def test_common_lattice_blend_applies_eta_to_two_states_node_by_node():
    h = 0.25
    box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    position = np.array([[0.75, 0.0, 0.0]])
    volume = np.array([h**3])
    fvm_vorticity = np.array([[0.0, 8.0, 0.0]])
    vpm_strength = np.array([[0.0, -0.25, 0.0]])
    state = blend_fvm_vpm_circulation_on_lattice(
        fvm_position=position,
        fvm_cell_volume=volume,
        fvm_vorticity=fvm_vorticity,
        vpm_position=position,
        vpm_vortex_strength=vpm_strength,
        transfer_box=box,
        blend_width=0.5,
        lattice_anchor=np.zeros(3),
        spacing=h,
    )
    node = np.all(state.position == position[0], axis=1)
    assert node.sum() == 1
    assert state.eta[node][0] == pytest.approx(0.5)
    expected = 0.5 * volume[0] * fvm_vorticity[0] + 0.5 * vpm_strength[0]
    np.testing.assert_allclose(state.partitioned_vortex_strength[node][0], expected, atol=2.0e-15)


def test_blend_cross_divergence_correction_preserves_net_and_matching_states():
    n = 12
    h = 2.0 * np.pi / n
    axis = h * np.arange(n)
    x, y, _z = np.meshgrid(axis, axis, axis, indexing="ij")
    fvm = np.zeros((n, n, n, 3))
    vpm = np.zeros_like(fvm)
    fvm[..., 1] = np.sin(x)
    vpm[..., 0] = np.sin(y)
    eta = 0.5 + 0.25 * np.cos(x)

    partitioned, corrected, before, after = correct_state_blend_cross_divergence(
        fvm,
        vpm,
        eta,
        spacing=h,
    )
    assert before > 1.0e-2
    assert after < 1.0e-12 * before
    np.testing.assert_allclose(
        corrected.sum(axis=(0, 1, 2)),
        partitioned.sum(axis=(0, 1, 2)),
        atol=1.0e-12,
    )

    matching_partition, matching_corrected, matching_before, matching_after = (
        correct_state_blend_cross_divergence(fvm, fvm, eta, spacing=h)
    )
    np.testing.assert_array_equal(matching_corrected, matching_partition)
    assert matching_before == 0.0
    assert matching_after == 0.0


def test_blend_correction_uses_the_same_discrete_divergence_as_its_diagnostics():
    n = 14
    h = 2.0 * np.pi / n
    axis = h * np.arange(n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    fvm = np.empty((n, n, n, 3))
    vpm = np.empty_like(fvm)
    fvm[..., 0] = np.sin(y) + 0.2 * np.cos(z)
    fvm[..., 1] = np.sin(z) + 0.1 * np.cos(x)
    fvm[..., 2] = np.sin(x) + 0.3 * np.cos(y)
    vpm[..., 0] = np.cos(y) - 0.1 * np.sin(z)
    vpm[..., 1] = np.cos(z) - 0.2 * np.sin(x)
    vpm[..., 2] = np.cos(x) - 0.3 * np.sin(y)
    eta = 0.5 + 0.2 * np.cos(x) * np.cos(y) * np.cos(z)

    _partitioned, corrected, before, after = correct_state_blend_cross_divergence(
        fvm,
        vpm,
        eta,
        spacing=h,
    )
    discrete_defect = spectral_lattice_divergence(corrected, spacing=h) - (
        eta * spectral_lattice_divergence(fvm, spacing=h)
        + (1.0 - eta) * spectral_lattice_divergence(vpm, spacing=h)
    )
    assert before > 1.0e-3
    assert after < 1.0e-12 * before
    assert np.sqrt(np.mean(discrete_defect**2)) < 1.0e-12 * before


def test_zero_width_common_lattice_state_keeps_complete_fvm_release_support():
    h = 0.25
    state = blend_fvm_vpm_circulation_on_lattice(
        fvm_position=np.array([[0.875, 0.0, 0.0]]),
        fvm_cell_volume=np.array([h**3]),
        fvm_vorticity=np.array([[0.0, 2.0, 0.0]]),
        vpm_position=np.empty((0, 3)),
        vpm_vortex_strength=np.empty((0, 3)),
        transfer_box=np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
        blend_width=0.0,
        lattice_anchor=np.zeros(3),
        spacing=h,
    )
    release = state.position[:, 0] > 1.0
    assert release.any()
    assert np.linalg.norm(state.vortex_strength[release], axis=1).sum() > 0.0
    expected = h**3 * np.array([0.0, 2.0, 0.0])
    np.testing.assert_allclose(state.gamma_net, expected, atol=2.0e-15)


@pytest.mark.parametrize(
    "point",
    [
        [-0.499, 0.0, 0.0],
        [0.499, 0.0, 0.0],
        [0.0, -0.499, 0.0],
        [0.0, 0.499, 0.0],
        [0.0, 0.0, -0.499],
        [0.0, 0.0, 0.499],
    ],
)
def test_complete_support_is_conservative_near_every_ownership_face(point):
    point = np.asarray(point, dtype=np.float64).reshape(1, 3)
    gamma = np.array([[0.2, -0.3, 0.5]])
    mapped = map_cell_circulation_to_lattice(
        point,
        np.array([1.0]),
        gamma,
        lattice_anchor=np.zeros(3),
        spacing=0.25,
    )
    np.testing.assert_allclose(mapped.target_gamma_net, gamma[0], rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        mapped.target_first_moment, first_vorticity_moment(point, gamma), rtol=0.0, atol=2.0e-15
    )
    assert np.prod(mapped.shape) == len(mapped.position)


def test_solid_and_zero_vorticity_donors_are_handled_without_nan_or_gamma_leakage():
    position = np.array([[0.1, 0.2, 0.3], [-0.2, 0.1, 0.0], [0.3, 0.2, -0.1]])
    volume = np.array([0.01, 0.03, 0.04])
    vorticity = np.array([[2.0, 1.0, 0.0], [100.0, 50.0, -2.0], [0.0, 0.0, 0.0]])
    mapped = map_cell_circulation_to_lattice(
        position,
        volume,
        vorticity,
        lattice_anchor=np.zeros(3),
        spacing=0.125,
        solid_mask=np.array([False, True, False]),
    )
    expected = volume[0] * vorticity[0]
    np.testing.assert_allclose(mapped.target_gamma_net, expected, rtol=0.0, atol=2.0e-15)
    assert np.isfinite(mapped.vortex_strength).all()


def test_actual_f32_stored_state_conservation(tmp_path, monkeypatch):
    """The authoritative budget is downloaded from the f32 VPM state."""
    pytest.importorskip("taichi", reason="VPM requires taichi")
    monkeypatch.chdir(tmp_path)
    from source.coupler.vorticity_transfer import replace_particles_from_lattice_blend
    from source.solvers.vpm import ViscousConfig, VPMSetup, VPMSolver

    h = 0.03
    position = np.array([[0.5 * h, -0.5 * h, 0.25 * h]])
    volume = np.array([h**3])
    vorticity = np.array([[1.5, -2.0, 0.25]])
    expected = map_cell_circulation_to_lattice(
        position,
        volume,
        vorticity,
        lattice_anchor=np.zeros(3),
        spacing=h,
    )
    solver = VPMSolver(
        VPMSetup(
            time_step_size=0.01,
            compute_device="CPU",
            precision="f32",
            max_n_particles=128,
            domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            viscous=ViscousConfig.cs(
                kinematic_viscosity=1.0e-3,
                particle_spacing=h,
                core_radius_ratio=1.0,
            ),
        )
    )

    replace_particles_from_lattice_blend(
        solver,
        transfer_box=np.array([-0.2, 0.2, -0.2, 0.2, -0.2, 0.2]),
        eta_blend_width=0.0,
        fvm_position=position,
        fvm_cell_volume=volume,
        fvm_vorticity=vorticity,
        lattice_anchor=np.zeros(3),
        particle_spacing=h,
        core_radius_ratio=1.0,
        kinematic_viscosity=1.0e-3,
    )

    actual_position = solver.particles.position_cpu()
    actual_strength = solver.particles.vortex_strength_cpu()
    expected_position = expected.position.astype(np.float32)
    expected_strength = expected.vortex_strength.astype(np.float32)
    np.testing.assert_array_equal(actual_position, expected_position)
    np.testing.assert_array_equal(actual_strength, expected_strength)
    np.testing.assert_allclose(
        actual_strength.sum(axis=0, dtype=np.float64),
        expected_strength.sum(axis=0, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_gaussian_vorticity_reference_matches_vpm_field_evaluation(tmp_path, monkeypatch):
    """The residual evaluator uses the same Gaussian convention as VPM."""
    pytest.importorskip("taichi", reason="VPM requires taichi")
    monkeypatch.chdir(tmp_path)
    from source.solvers.vpm import VelocityConfig, ViscousConfig, VPMSetup, VPMSolver

    position = np.array([[0.1, -0.2, 0.05], [-0.25, 0.3, -0.1]])
    strength = np.array([[0.2, -0.4, 0.1], [-0.3, 0.5, 0.7]])
    core_radius = np.array([0.125, 0.35])
    points = np.array([[0.0, 0.0, 0.0], [0.25, -0.5, 0.125], [-0.1, 0.2, 0.3]])
    solver = VPMSolver(
        VPMSetup(
            time_step_size=0.01,
            compute_device="CPU",
            precision="f64",
            max_n_particles=16,
            domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            velocity=VelocityConfig.direct(),
            viscous=ViscousConfig.cs(
                kinematic_viscosity=1.0e-3,
                particle_spacing=0.125,
                core_radius_ratio=1.0,
            ),
        )
    )
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=strength,
        core_radius=core_radius,
        particle_volume=core_radius**3,
        kinematic_viscosity=np.full(len(position), 1.0e-3),
    )

    reference = evaluate_gaussian_vorticity(points, position, strength, core_radius)
    actual = solver.compute_vorticity_at_points(points)
    # Target-field storage is currently single precision even in the f64 VPM
    # configuration; this checks kernel convention rather than that storage
    # precision choice.
    np.testing.assert_allclose(actual, reference, rtol=5.0e-8, atol=3.0e-10)


def test_f32_matching_state_is_a_fixed_point_over_one_thousand_transfers(tmp_path, monkeypatch):
    """A matching hard-state handoff cannot accumulate f32 representation drift."""
    pytest.importorskip("taichi", reason="VPM requires taichi")
    monkeypatch.chdir(tmp_path)
    from source.coupler.vorticity_transfer import replace_particles_from_lattice_blend
    from source.solvers.vpm import ViscousConfig, VPMSetup, VPMSolver

    h = 0.03125
    position = np.array([[0.0, 0.0, 0.0]])
    volume = np.array([h**3])
    vorticity = np.array([[1.5, -2.0, 0.25]])
    solver = VPMSolver(
        VPMSetup(
            time_step_size=0.01,
            compute_device="CPU",
            precision="f32",
            max_n_particles=128,
            domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            viscous=ViscousConfig.cs(
                kinematic_viscosity=1.0e-3,
                particle_spacing=h,
                core_radius_ratio=1.0,
            ),
        )
    )
    transfer_arguments = {
        "transfer_box": np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1]),
        "eta_blend_width": 0.0,
        "fvm_position": position,
        "fvm_cell_volume": volume,
        "fvm_vorticity": vorticity,
        "lattice_anchor": np.zeros(3),
        "particle_spacing": h,
        "core_radius_ratio": 1.0,
        "kinematic_viscosity": 1.0e-3,
    }

    replace_particles_from_lattice_blend(solver, **transfer_arguments)
    first_position = solver.particles.position_cpu().copy()
    first_strength = solver.particles.vortex_strength_cpu().copy()
    first_count = solver.particles.n_particles_total
    for _ in range(999):
        result = replace_particles_from_lattice_blend(solver, **transfer_arguments)
        assert result.n_particles_after == first_count
    np.testing.assert_array_equal(solver.particles.position_cpu(), first_position)
    np.testing.assert_array_equal(solver.particles.vortex_strength_cpu(), first_strength)


def test_m4_lattice_particles_are_preserved_by_cs_core_spreading(tmp_path, monkeypatch):
    """CS diffuses a coupled lattice blend by core growth without pruning it."""
    pytest.importorskip("taichi", reason="VPM requires taichi")
    monkeypatch.chdir(tmp_path)
    from source.coupler.vorticity_transfer import replace_particles_from_lattice_blend
    from source.solvers.vpm import ViscousConfig, VPMSetup, VPMSolver

    h = 0.125
    kinematic_viscosity = 0.01
    time_step_size = 0.02
    solver = VPMSolver(
        VPMSetup(
            time_step_size=time_step_size,
            compute_device="CPU",
            max_n_particles=256,
            domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            viscous=ViscousConfig.cs(
                kinematic_viscosity=kinematic_viscosity,
                particle_spacing=h,
                core_radius_ratio=1.0,
            ),
        )
    )
    result = replace_particles_from_lattice_blend(
        solver,
        transfer_box=np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        eta_blend_width=2.0 * h,
        fvm_position=np.array([[0.5 * h, -0.5 * h, 0.25 * h]]),
        fvm_cell_volume=np.array([h**3]),
        fvm_vorticity=np.array([[2.0, -1.0, 0.5]]),
        lattice_anchor=np.zeros(3),
        particle_spacing=h,
        core_radius_ratio=1.0,
        kinematic_viscosity=kinematic_viscosity,
    )
    count = result.n_particles_after
    position_before = solver.particles.position_cpu().copy()
    strength_before = solver.particles.vortex_strength_cpu().copy()
    radius_before = solver.particles.core_radius_cpu().copy()

    solver.physics.core_spreading_diffusion(solver.particles, time_step_size)

    assert solver.particles.n_particles_total == count
    np.testing.assert_allclose(solver.particles.position_cpu(), position_before, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        solver.particles.vortex_strength_cpu(), strength_before, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        solver.particles.core_radius_cpu() ** 2,
        radius_before**2 + 4.0 * kinematic_viscosity * time_step_size,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
