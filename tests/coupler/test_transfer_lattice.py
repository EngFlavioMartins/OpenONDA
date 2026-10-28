"""Vorticity transfer on the transfer lattice: remeshing, strength correction, removal and moments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from openonda.coupler import vorticity_transfer as transfer_module
from openonda.coupler.solver import _validate_gbd_moment_correction
from openonda.coupler.transfer_lattice import (
    blending_weight,
    build_transfer_lattice,
    correct_lattice_strength,
    fvm_target_strength,
    gaussian_representation,
    redistribute_removed_strength,
    remove_weak_strengths,
    scatter_m4_prime_to_lattice,
    transfer_on_lattice,
    vortex_moments,
)
from openonda.coupler.vorticity_transfer import transfer_vorticity_to_particles

TRANSFER_REGION = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])


def _zero_target(points: np.ndarray) -> np.ndarray:
    return np.zeros((len(points), 3), dtype=np.float64)


def test_applied_particle_reductions_are_chunked_and_match_direct_float64_diagnostics(
    monkeypatch,
):
    rng = np.random.default_rng(20260826)
    position = rng.uniform(-2.0, 3.0, size=(11, 3)).astype(np.float32)
    strength = rng.normal(scale=0.02, size=(11, 3)).astype(np.float32)
    reference = strength.astype(np.float64)
    reference[:7] += rng.normal(scale=1.0e-9, size=(7, 3))
    monkeypatch.setattr(transfer_module, "_PARTICLE_REDUCTION_CHUNK_SIZE", 3)

    reduced = transfer_module._reduce_applied_particle_state(
        position,
        strength,
        transfer_count=7,
        reference_vortex_strength=reference,
    )
    expected = vortex_moments(position[:7].astype(np.float64), strength[:7].astype(np.float64))

    np.testing.assert_allclose(
        reduced.transfer_moments.total_vortex_strength,
        expected.total_vortex_strength,
        rtol=0.0,
        atol=1.0e-16,
    )
    np.testing.assert_allclose(
        reduced.transfer_moments.linear_impulse,
        expected.linear_impulse,
        rtol=0.0,
        atol=1.0e-16,
    )
    np.testing.assert_allclose(
        reduced.transfer_moments.angular_impulse,
        expected.angular_impulse,
        rtol=0.0,
        atol=1.0e-16,
    )
    assert reduced.transfer_vortex_strength_l1 == pytest.approx(
        np.linalg.norm(strength[:7].astype(np.float64), axis=1).sum()
    )
    assert reduced.cast_correction_l1 == pytest.approx(
        np.linalg.norm(strength[:7].astype(np.float64) - reference[:7], axis=1).sum()
    )
    assert reduced.transfer_position_scale == pytest.approx(
        np.linalg.norm(position[:7].astype(np.float64), axis=1).max()
    )
    np.testing.assert_allclose(
        reduced.output_vortex_strength_net,
        strength.sum(axis=0, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-16,
    )
    assert reduced.maximum_output_vortex_strength == pytest.approx(
        np.linalg.norm(strength.astype(np.float64), axis=1).max()
    )


class _Particles:
    def __init__(self, position: np.ndarray, vortex_strength: np.ndarray):
        self.position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
        self.vortex_strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3)
        self.n_particles_total = len(self.position)
        self.capacity = 10_000

    def position_cpu(self) -> np.ndarray:
        return self.position.copy()

    def vortex_strength_cpu(self) -> np.ndarray:
        return self.vortex_strength.copy()


class _GBDVPM:
    viscous_scheme = "gbd"
    np_dtype = np.float64

    def __init__(
        self,
        position: np.ndarray,
        vortex_strength: np.ndarray,
        *,
        dtype=np.float64,
    ):
        self.np_dtype = np.dtype(dtype)
        self.particles = _Particles(position, vortex_strength)
        self.last_replacement: dict[str, np.ndarray | bool] | None = None

    def replace_vortex_particles(self, **fields) -> None:
        self.last_replacement = {
            name: np.asarray(value).copy() if name != "report_removal" else bool(value)
            for name, value in fields.items()
        }
        self.particles.position = np.asarray(fields["position"]).copy()
        self.particles.vortex_strength = np.asarray(fields["vortex_strength"]).copy()
        self.particles.n_particles_total = len(self.particles.position)


def test_fixed_lattice_scatter_is_complete_and_conservative_at_remeshing_region_faces():
    spacing = 0.125
    lattice = build_transfer_lattice(
        TRANSFER_REGION,
        spacing,
        remeshing_extension_length=0.2,
        blend_end_depth=0.25,
        blend_start_depth=0.05,
        lattice_anchor=np.array([0.03125, -0.046875, 0.015625]),
    )
    position = np.array(
        [
            [-0.699, -0.412, 0.533],
            [0.699, 0.641, -0.587],
            [0.173, -0.699, 0.699],
        ]
    )
    strength = np.array(
        [
            [0.2, -0.3, 0.5],
            [-0.7, 0.1, 0.25],
            [0.4, 0.6, -0.2],
        ]
    )

    target = scatter_m4_prime_to_lattice(position, strength, lattice)

    before = vortex_moments(position, strength)
    after = vortex_moments(lattice.positions, target)
    np.testing.assert_allclose(
        after.total_vortex_strength,
        before.total_vortex_strength,
        rtol=0.0,
        atol=4.0e-15,
    )
    np.testing.assert_allclose(
        after.linear_impulse,
        before.linear_impulse,
        rtol=0.0,
        atol=4.0e-15,
    )


def test_fvm_target_strength_recovers_the_curl_of_a_linear_velocity_exactly():
    rng = np.random.default_rng(20260826)
    position = rng.uniform(-1.0, 1.0, size=(20, 3))
    spacing = 0.08
    gradient = np.array(
        [
            [0.1, -0.3, 0.7],
            [0.4, 0.2, -0.5],
            [-0.6, 0.9, -0.1],
        ]
    )

    def velocity(points: np.ndarray) -> np.ndarray:
        return np.asarray(points) @ gradient + np.array([0.4, -0.2, 0.1])

    strength = fvm_target_strength(position, spacing, velocity)
    curl = np.array(
        [
            gradient[1, 2] - gradient[2, 1],
            gradient[2, 0] - gradient[0, 2],
            gradient[0, 1] - gradient[1, 0],
        ]
    )
    np.testing.assert_allclose(
        strength,
        np.tile(curl * spacing**3, (len(position), 1)),
        rtol=0.0,
        atol=2.0e-17,
    )


def test_blending_weight_is_zero_in_the_zero_weight_layer_and_one_in_the_replacement_region():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.30, 0.0, 0.0],
            [0.40, 0.0, 0.0],
            [0.46, 0.0, 0.0],
            [0.50, 0.0, 0.0],
            [0.51, 0.0, 0.0],
        ]
    )
    blend_weight = blending_weight(points, TRANSFER_REGION, 0.05, 0.2)

    np.testing.assert_allclose(blend_weight[:2], 1.0)
    assert 0.0 < blend_weight[2] < 1.0
    np.testing.assert_allclose(blend_weight[3:], 0.0)


def test_limited_strength_correction_reduces_the_representation_residual():
    shape = (9, 9, 9)
    rng = np.random.default_rng(11)
    vpm_strength = rng.normal(scale=1.0e-4, size=(np.prod(shape), 3))
    fvm_target = rng.normal(scale=1.0e-4, size=(np.prod(shape), 3))
    blend_weight = np.linspace(0.0, 1.0, np.prod(shape))

    correction = correct_lattice_strength(
        vpm_strength,
        fvm_target,
        blend_weight,
        shape,
        0.1,
        core_radius=0.1,
        max_strength_ratio=2.0,
    )

    assert correction.residual_after_correction is not None
    assert correction.residual_after_correction < correction.residual_before_correction
    assert np.isfinite(correction.limited_strength).all()


def test_weak_strength_removal_is_continuous_and_zero_below_its_threshold():
    threshold = 1.0e-3
    magnitude = np.linspace(0.0, 5.0 * threshold, 4001)
    strength = np.zeros((len(magnitude), 3), dtype=np.float64)
    strength[:, 2] = magnitude

    retained, removed = remove_weak_strengths(strength, threshold)

    assert np.all(retained[magnitude <= threshold] == 0.0)
    assert np.max(np.abs(np.diff(retained[:, 2]))) < 2.0 * (magnitude[1] - magnitude[0])
    np.testing.assert_allclose(retained + removed, strength, rtol=0.0, atol=1.0e-18)


def test_removed_strength_redistribution_preserves_strength_and_linear_impulse():
    shape = (5, 5, 5)
    spacing = 0.2
    positions = np.stack(
        np.meshgrid(*[spacing * np.arange(size) for size in shape], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    removed = np.zeros((*shape, 3), dtype=np.float64)
    removed[2, 2, 2] = [0.0, 0.0, 1.0e-4]
    retained = np.zeros((*shape, 3), dtype=np.float64)
    for axis in range(3):
        for step in (-1, 1):
            index = [2, 2, 2]
            index[axis] += step
            retained[tuple(index)] = [0.0, 0.0, 2.0e-4]

    redistributed = redistribute_removed_strength(
        removed.reshape(-1, 3), retained.reshape(-1, 3), shape
    )

    before = vortex_moments(positions, (retained + removed).reshape(-1, 3))
    after = vortex_moments(positions, redistributed)
    np.testing.assert_allclose(
        after.total_vortex_strength,
        before.total_vortex_strength,
        rtol=0.0,
        atol=1.0e-16,
    )
    np.testing.assert_allclose(
        after.linear_impulse,
        before.linear_impulse,
        rtol=0.0,
        atol=1.0e-16,
    )


def test_resolved_weak_wake_survives_weak_strength_removal_across_the_transfer_surface():
    spacing = 0.1
    lattice = build_transfer_lattice(
        [-0.3, 0.3, -0.3, 0.3, -0.3, 0.3],
        spacing,
        remeshing_extension_length=0.2,
        blend_end_depth=0.2,
        blend_start_depth=0.1,
        lattice_anchor=np.zeros(3),
    )

    def weak_wake(points: np.ndarray) -> np.ndarray:
        vorticity = np.zeros_like(points)
        radius_squared = points[:, 1] ** 2 + points[:, 2] ** 2
        vorticity[:, 2] = 0.08 * np.exp(-radius_squared / 0.22**2)
        return vorticity * spacing**3

    position = lattice.positions.copy()
    vortex_strength = weak_wake(position)
    common = {
        "fvm_target_strength_at_node": weak_wake,
        "max_strength_ratio": 1.8,
        "compute_diagnostics": False,
    }
    thresholded = transfer_on_lattice(
        position,
        vortex_strength,
        lattice,
        strength_threshold=0.05 * spacing**3,
        surface_strength_threshold=0.002 * spacing**3,
        **common,
    )
    unthresholded = transfer_on_lattice(
        position,
        vortex_strength,
        lattice,
        strength_threshold=0.0,
        **common,
    )

    def represented_strength(result) -> np.ndarray:
        # Weak-strength removal can change which kept particles merge with
        # transfer-lattice particles. Compare the complete wake in both cases.
        index = np.rint((result.position - lattice.origin) / spacing).astype(int)
        flat_index = np.ravel_multi_index(index.T, lattice.shape)
        lattice_strength = np.zeros_like(lattice.positions)
        np.add.at(lattice_strength, flat_index, result.vortex_strength)
        return gaussian_representation(
            lattice_strength,
            lattice.shape,
            spacing,
            core_radius=1.1 * spacing,
        )

    represented_thresholded = represented_strength(thresholded)
    represented_unthresholded = represented_strength(unthresholded)
    outside = (lattice.positions[:, 0] >= 0.3) & (
        np.linalg.norm(represented_unthresholded, axis=1) > 0.1 * 0.08 * spacing**3
    )
    difference = represented_thresholded[outside] - represented_unthresholded[outside]
    relative_rms = np.sqrt(np.mean(difference**2)) / np.sqrt(
        np.mean(represented_unthresholded[outside] ** 2)
    )

    assert thresholded.transfer_output_count < unthresholded.transfer_output_count
    assert np.count_nonzero(outside) > 20
    assert relative_rms < 0.012


def test_remeshing_region_transfer_keeps_outer_wake_and_does_not_accumulate():
    from dataclasses import replace

    spacing = 0.1
    lattice = build_transfer_lattice(
        TRANSFER_REGION,
        spacing,
        remeshing_extension_length=0.2,
        blend_end_depth=0.2,
        blend_start_depth=0.05,
        lattice_anchor=np.zeros(3),
    )
    # Isolate the remeshing: a zero FVM target strength does not match the
    # nonzero Gaussian tails of the outer test particles.
    lattice = replace(lattice, blending_weight=np.zeros(len(lattice.positions)))
    position = np.array([[0.65, 0.13, -0.07], [0.95, 0.22, 0.11]])
    strength = np.array([[0.2, -0.1, 0.3], [-0.05, 0.07, 0.02]])

    first = transfer_on_lattice(
        position,
        strength,
        lattice,
        fvm_target_strength_at_node=_zero_target,
        max_strength_ratio=1.0,
        compute_diagnostics=False,
    )
    second = transfer_on_lattice(
        first.position,
        first.vortex_strength,
        lattice,
        fvm_target_strength_at_node=_zero_target,
        max_strength_ratio=1.0,
        compute_diagnostics=False,
    )

    assert first.transfer_input_count == 1
    assert first.preserved_outer_count == 1
    assert second.particle_count == first.particle_count
    np.testing.assert_allclose(second.position, first.position, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(second.vortex_strength, first.vortex_strength, atol=2.0e-15)
    np.testing.assert_allclose(first.core_radius, spacing)
    outer = np.flatnonzero(np.all(first.position == position[1], axis=1))
    assert len(outer) == 1
    np.testing.assert_allclose(first.vortex_strength[outer[0]], strength[1])

    before = vortex_moments(position, strength)
    after = vortex_moments(first.position, first.vortex_strength)
    np.testing.assert_allclose(
        after.total_vortex_strength, before.total_vortex_strength, atol=2e-14
    )
    np.testing.assert_allclose(after.linear_impulse, before.linear_impulse, atol=2e-14)
    assert first.conservation_raw_mismatch["total_vortex_strength"] < 2.0e-14
    assert first.conservation_applied_correction["total_vortex_strength"] < 2.0e-14
    assert first.conservation_residual["total_vortex_strength"] < 2.0e-14
    assert first.conservation_applied_particle_strength_fraction < 2.0e-14


def test_removal_diagnostics_expose_raw_mismatch_and_applied_moment_correction():
    spacing = 0.1
    lattice = build_transfer_lattice(
        TRANSFER_REGION,
        spacing,
        remeshing_extension_length=0.2,
        blend_end_depth=0.2,
        lattice_anchor=np.zeros(3),
        mesh_weight_at_node=lambda points: np.zeros(len(points)),
    )
    strong_position = np.array(
        [[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)]
    )
    position = np.vstack((strong_position, np.array([[0.6, 0.6, 0.6]])))
    strength = np.zeros_like(position)
    strength[:-1, 2] = 1.0
    strength[-1, 2] = 0.01

    result = transfer_on_lattice(
        position,
        strength,
        lattice,
        fvm_target_strength_at_node=_zero_target,
        strength_threshold=0.05,
        max_strength_ratio=1.0,
        compute_diagnostics=False,
    )

    assert result.conservation_raw_mismatch["total_vortex_strength"] > 1.0e-3
    assert result.conservation_applied_correction["total_vortex_strength"] > 1.0e-3
    assert result.conservation_residual["total_vortex_strength"] < 1.0e-12
    assert result.conservation_raw_mismatch["linear_impulse"] > 1.0e-4
    assert result.conservation_applied_correction["linear_impulse"] > 1.0e-4
    assert result.conservation_residual["linear_impulse"] < 1.0e-12
    assert result.conservation_applied_particle_strength_fraction > 0.0

    vpm = _GBDVPM(position, strength)
    with pytest.raises(RuntimeError, match="moment correction of the vorticity transfer exceeds"):
        transfer_vorticity_to_particles(
            vpm,
            lattice=lattice,
            fvm_target_strength_at_node=_zero_target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            strength_threshold=0.05,
            core_radius_ratio=1.0,
            max_strength_ratio=1.0,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
            max_moment_correction_fraction=1.0e-6,
        )
    np.testing.assert_array_equal(vpm.particles.position, position)
    np.testing.assert_array_equal(vpm.particles.vortex_strength, strength)


def test_repeated_vorticity_transfer_keeps_a_fixed_particle_set_and_base_radii():
    spacing = 0.1
    lattice = build_transfer_lattice(
        [-0.3, 0.3, -0.3, 0.3, -0.3, 0.3],
        spacing,
        remeshing_extension_length=0.5,
        blend_end_depth=0.2,
        blend_start_depth=0.05,
        lattice_anchor=np.zeros(3),
    )

    def target(points: np.ndarray) -> np.ndarray:
        radius_squared = np.einsum("ij,ij->i", points, points)
        strength = np.zeros_like(points)
        strength[:, 2] = 2.0e-4 * np.exp(-radius_squared / 0.08)
        return strength

    first = transfer_on_lattice(
        np.empty((0, 3)),
        np.empty((0, 3)),
        lattice,
        fvm_target_strength_at_node=target,
        strength_threshold=1.0e-12,
        compute_diagnostics=False,
    )
    second = transfer_on_lattice(
        first.position,
        first.vortex_strength,
        lattice,
        fvm_target_strength_at_node=target,
        strength_threshold=1.0e-12,
        compute_diagnostics=False,
    )

    assert first.particle_count > 0
    assert second.particle_count == first.particle_count
    np.testing.assert_allclose(second.position, first.position, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(second.core_radius, spacing)


def test_particle_capacity_limit_preserves_total_strength_and_linear_impulse():
    rng = np.random.default_rng(3)
    position = rng.uniform([1.0, -0.5, -0.5], [2.0, 0.5, 0.5], (8, 3))
    strength = rng.normal(size=(8, 3)) * 0.05
    lattice = build_transfer_lattice(
        TRANSFER_REGION,
        0.1,
        remeshing_extension_length=0.0,
        blend_end_depth=0.2,
        lattice_anchor=np.zeros(3),
    )

    result = transfer_on_lattice(
        position,
        strength,
        lattice,
        fvm_target_strength_at_node=_zero_target,
        maximum_particle_count=4,
        compute_diagnostics=False,
    )

    assert result.particle_count == 4
    assert result.particle_set_removed_count == 4
    before = vortex_moments(position, strength)
    after = vortex_moments(result.position, result.vortex_strength)
    np.testing.assert_allclose(
        after.total_vortex_strength, before.total_vortex_strength, atol=1e-12
    )
    np.testing.assert_allclose(after.linear_impulse, before.linear_impulse, atol=1e-12)
    assert np.isfinite(result.particle_set_removed_velocity_bound)
    assert result.particle_set_conservation_raw_mismatch["total_vortex_strength"] > 0.0
    assert result.particle_set_conservation_applied_correction["total_vortex_strength"] > 0.0
    assert result.particle_set_conservation_residual["total_vortex_strength"] < 1.0e-12
    assert result.particle_set_conservation_raw_mismatch["linear_impulse"] > 0.0
    assert result.particle_set_conservation_applied_correction["linear_impulse"] > 0.0
    assert result.particle_set_conservation_residual["linear_impulse"] < 1.0e-12
    assert result.particle_set_conservation_applied_particle_strength_fraction > 0.0

    vpm = _GBDVPM(position, strength)
    vpm.particles.capacity = 4
    with pytest.raises(RuntimeError, match="reached the VPM particle capacity"):
        transfer_vorticity_to_particles(
            vpm,
            lattice=lattice,
            fvm_target_strength_at_node=_zero_target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            strength_threshold=0.0,
            core_radius_ratio=1.0,
            max_strength_ratio=1.0,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
            max_moment_correction_fraction=1.0e-6,
        )
    np.testing.assert_array_equal(vpm.particles.position, position)
    np.testing.assert_array_equal(vpm.particles.vortex_strength, strength)


def test_vorticity_transfer_replaces_one_complete_gbd_particle_set_without_accumulation():
    spacing = 0.1
    lattice = build_transfer_lattice(
        TRANSFER_REGION,
        spacing,
        remeshing_extension_length=0.2,
        blend_end_depth=0.2,
        lattice_anchor=np.zeros(3),
    )

    def target(points: np.ndarray) -> np.ndarray:
        strength = np.zeros_like(points)
        strength[:, 2] = 1.0e-3 * np.exp(-8.0 * np.einsum("ij,ij->i", points, points))
        return strength

    vpm = _GBDVPM(
        np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0e-3], [0.0, 0.0, 2.0e-4]]),
    )
    counts: list[int] = []
    for _ in range(3):
        result = transfer_vorticity_to_particles(
            vpm,
            lattice=lattice,
            fvm_target_strength_at_node=target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            strength_threshold=1.0e-8,
            core_radius_ratio=1.0,
            max_strength_ratio=1.8,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
        )
        counts.append(result.n_particles_after)
        assert result.preserved_outer_particles == 1
        assert vpm.last_replacement is not None
        assert vpm.last_replacement["report_removal"] is False
        np.testing.assert_allclose(vpm.last_replacement["core_radius"], spacing)
        np.testing.assert_allclose(vpm.last_replacement["particle_volume"], spacing**3)

    assert counts[1:] == [counts[0], counts[0]]
    outer = np.flatnonzero(np.isclose(vpm.particles.position[:, 0], 0.95))
    assert len(outer) == 1
    np.testing.assert_allclose(vpm.particles.vortex_strength[outer[0]], [0.0, 0.0, 2.0e-4])


def test_vorticity_transfer_checks_the_actual_float32_particle_state():
    spacing = 0.1
    lattice = build_transfer_lattice(
        TRANSFER_REGION,
        spacing,
        remeshing_extension_length=0.2,
        blend_end_depth=0.2,
        lattice_anchor=np.full(3, -0.05),
    )

    def target(points: np.ndarray) -> np.ndarray:
        strength = np.empty_like(points)
        strength[:, 0] = 1.0e-4 * (1.0 + 0.13 * points[:, 1])
        strength[:, 1] = -2.0e-4 * (1.0 - 0.07 * points[:, 2])
        strength[:, 2] = 3.0e-4 * (1.0 + 0.11 * points[:, 0])
        return strength

    numerical = transfer_on_lattice(
        np.empty((0, 3)),
        np.empty((0, 3)),
        lattice,
        fvm_target_strength_at_node=target,
        strength_threshold=0.0,
        max_strength_ratio=1.0,
        compute_diagnostics=False,
    )
    assert numerical.conservation_target_moments is not None
    vpm = _GBDVPM(np.empty((0, 3)), np.empty((0, 3)), dtype=np.float32)

    transfer = transfer_vorticity_to_particles(
        vpm,
        lattice=lattice,
        fvm_target_strength_at_node=target,
        particle_fluid_weight=None,
        particle_in_solid=None,
        strength_threshold=0.0,
        core_radius_ratio=1.0,
        max_strength_ratio=1.0,
        kinematic_viscosity=1.0e-3,
        freestream_speed=1.0,
        time_step_size=0.01,
        compute_diagnostics=False,
        max_moment_correction_fraction=0.08,
    )

    actual = vortex_moments(vpm.particles.position, vpm.particles.vortex_strength)
    expected = numerical.conservation_target_moments
    strength_error = np.linalg.norm(actual.total_vortex_strength - expected.total_vortex_strength)
    impulse_error = np.linalg.norm(actual.linear_impulse - expected.linear_impulse)
    assert vpm.particles.vortex_strength.dtype == np.float32
    assert transfer.transfer_conservation_error == pytest.approx(strength_error)
    assert transfer.transfer_linear_impulse_error == pytest.approx(impulse_error)
    assert transfer.transfer_conservation_error <= transfer.transfer_vortex_strength_tolerance
    assert transfer.transfer_linear_impulse_error <= transfer.transfer_linear_impulse_tolerance


def test_support_output_merges_with_a_kept_particle_on_the_same_lattice_node():
    spacing = 0.1
    lattice = build_transfer_lattice(
        TRANSFER_REGION,
        spacing,
        remeshing_extension_length=0.2,
        blend_end_depth=0.2,
        lattice_anchor=np.full(3, -0.05),
        mesh_weight_at_node=lambda points: np.zeros(len(points)),
    )
    position = np.array(
        [
            [0.699, -0.05, -0.05],
            [0.75, -0.05, -0.05],
        ]
    )
    strength = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.2],
        ]
    )

    result = transfer_on_lattice(
        position,
        strength,
        lattice,
        fvm_target_strength_at_node=_zero_target,
        strength_threshold=0.0,
        max_strength_ratio=1.0,
        compute_diagnostics=False,
    )

    hits = np.all(np.isclose(result.position, position[1], rtol=0.0, atol=1.0e-14), axis=1)
    assert np.count_nonzero(hits) == 1
    assert result.coalesced_outer_count == 1
    assert result.preserved_outer_count == 0
    before = vortex_moments(position, strength)
    after = vortex_moments(result.position, result.vortex_strength)
    np.testing.assert_allclose(
        after.total_vortex_strength,
        before.total_vortex_strength,
        rtol=0.0,
        atol=2.0e-13,
    )

    vpm = _GBDVPM(position, strength)
    transfer = transfer_vorticity_to_particles(
        vpm,
        lattice=lattice,
        fvm_target_strength_at_node=_zero_target,
        particle_fluid_weight=None,
        particle_in_solid=None,
        strength_threshold=0.0,
        core_radius_ratio=1.0,
        max_strength_ratio=1.0,
        kinematic_viscosity=1.0e-3,
        freestream_speed=1.0,
        time_step_size=0.01,
        compute_diagnostics=False,
    )
    assert transfer.coalesced_outer_particles == 1
    assert transfer.n_particles_removed == 2
    assert transfer.n_particles_retained == 0
    assert (
        transfer.n_particles_before - transfer.n_particles_removed + transfer.n_particles_injected
        == transfer.n_particles_after
    )


def test_invalid_coalesced_input_index_is_rejected_before_vpm_mutation(monkeypatch):
    spacing = 0.1
    lattice = build_transfer_lattice(
        TRANSFER_REGION,
        spacing,
        remeshing_extension_length=0.2,
        blend_end_depth=0.2,
        lattice_anchor=np.full(3, -0.05),
        mesh_weight_at_node=lambda points: np.zeros(len(points)),
    )
    position = np.array([[0.699, -0.05, -0.05], [0.75, -0.05, -0.05]])
    strength = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.2]])
    numerical = transfer_on_lattice(
        position,
        strength,
        lattice,
        fvm_target_strength_at_node=_zero_target,
        strength_threshold=0.0,
        max_strength_ratio=1.0,
        compute_diagnostics=False,
    )
    invalid = replace(
        numerical,
        coalesced_outer_input_indices=np.array([len(position)], dtype=np.int64),
    )
    monkeypatch.setattr(transfer_module, "transfer_on_lattice", lambda *args, **kwargs: invalid)
    vpm = _GBDVPM(position, strength)

    with pytest.raises(RuntimeError, match="invalid merged-particle index"):
        transfer_vorticity_to_particles(
            vpm,
            lattice=lattice,
            fvm_target_strength_at_node=_zero_target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            strength_threshold=0.0,
            core_radius_ratio=1.0,
            max_strength_ratio=1.0,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
        )

    assert vpm.last_replacement is None
    np.testing.assert_array_equal(vpm.particles.position, position)
    np.testing.assert_array_equal(vpm.particles.vortex_strength, strength)


def test_support_edge_merges_multiple_particles_but_keeps_a_near_miss_across_steps():
    spacing = 0.1
    lattice = build_transfer_lattice(
        TRANSFER_REGION,
        spacing,
        remeshing_extension_length=0.2,
        blend_end_depth=0.2,
        lattice_anchor=np.full(3, -0.05),
        mesh_weight_at_node=lambda points: np.zeros(len(points)),
    )
    position = np.array(
        [
            [0.699, -0.05, -0.05],
            [0.75, -0.05, -0.05],
            [0.75, -0.05, -0.05],
            [0.75001, -0.05, -0.05],
        ]
    )
    strength = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.05],
            [0.0, 0.0, 0.03],
        ]
    )
    vpm = _GBDVPM(position, strength)
    counts: list[int] = []

    for iteration in range(3):
        transfer = transfer_vorticity_to_particles(
            vpm,
            lattice=lattice,
            fvm_target_strength_at_node=_zero_target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            strength_threshold=0.0,
            core_radius_ratio=1.0,
            max_strength_ratio=1.0,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
            max_moment_correction_fraction=1.0,
        )
        counts.append(transfer.n_particles_after)
        exact_seam = np.all(
            np.isclose(vpm.particles.position, position[1], rtol=0.0, atol=1.0e-14),
            axis=1,
        )
        near_miss = np.all(
            np.isclose(vpm.particles.position, position[3], rtol=0.0, atol=1.0e-14),
            axis=1,
        )
        assert np.count_nonzero(exact_seam) == 1
        assert np.count_nonzero(near_miss) == 1
        assert len(np.unique(vpm.particles.position, axis=0)) == transfer.n_particles_after
        assert (
            transfer.n_particles_before
            - transfer.n_particles_removed
            + transfer.n_particles_injected
            == transfer.n_particles_after
        )
        if iteration == 0:
            assert transfer.coalesced_outer_particles == 2

    assert counts[1:] == [counts[0], counts[0]]


def _valid_moment_correction(**overrides):
    correction = {
        "applied": True,
        "nonzero_node_count": 100,
        "retained_node_count": 90,
        "removed_node_count": 10,
        "support_augmented_node_count": 2,
        "correction_fraction": 0.01,
        "normalized_vortex_strength_residual": 1.0e-8,
        "normalized_linear_impulse_residual": 2.0e-8,
        "normalized_angular_impulse_residual": 3.0e-8,
    }
    correction.update(overrides)
    return correction


def test_gbd_moment_correction_accepts_a_conservative_weak_strength_removal():
    _validate_gbd_moment_correction(_valid_moment_correction(), 0.08)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"applied": False}, "without the moment correction"),
        ({"correction_fraction": np.nan}, "is non-finite"),
        ({"normalized_linear_impulse_residual": np.inf}, "is non-finite"),
        ({"normalized_angular_impulse_residual": 1.1e-5}, "residual tolerance"),
        ({"correction_fraction": 0.081}, "moment correction above its limit"),
    ],
)
def test_gbd_moment_correction_rejects_missing_or_excessive_corrections(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_gbd_moment_correction(_valid_moment_correction(**overrides), 0.08)
