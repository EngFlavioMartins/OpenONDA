"""Focused contracts for the recovered long-run stable renewal mechanism."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from source.coupler import vorticity_transfer as transfer_module
from source.coupler.stable_renewal import (
    blend_represented_state,
    build_stable_renewal_lattice,
    inward_cosine_authority,
    m4_prime,
    redistribute_pruned_vortex_strength_locally,
    renew_stable_overlap,
    required_buffer_length,
    scatter_m4_prime_to_lattice,
    soft_prune_vortex_strength,
    vortex_invariants,
    vortex_strength_from_velocity_trace,
)
from source.coupler.vorticity_transfer import replace_particles_from_buffered_m4_renewal

BOX = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])


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
        renewed_count=7,
        reference_vortex_strength=reference,
    )
    expected = vortex_invariants(position[:7].astype(np.float64), strength[:7].astype(np.float64))

    np.testing.assert_allclose(
        reduced.renewed_invariants.total_vortex_strength,
        expected.total_vortex_strength,
        rtol=0.0,
        atol=1.0e-16,
    )
    np.testing.assert_allclose(
        reduced.renewed_invariants.linear_impulse,
        expected.linear_impulse,
        rtol=0.0,
        atol=1.0e-16,
    )
    np.testing.assert_allclose(
        reduced.renewed_invariants.angular_impulse,
        expected.angular_impulse,
        rtol=0.0,
        atol=1.0e-16,
    )
    assert reduced.renewed_vortex_strength_l1 == pytest.approx(
        np.linalg.norm(strength[:7].astype(np.float64), axis=1).sum()
    )
    assert reduced.cast_correction_l1 == pytest.approx(
        np.linalg.norm(strength[:7].astype(np.float64) - reference[:7], axis=1).sum()
    )
    assert reduced.renewed_position_scale == pytest.approx(
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
    viscous_scheme = "GBD"
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


def test_buffer_contains_advection_and_complete_m4_support():
    assert required_buffer_length(1.0, 0.01, 0.03125) == 0.0775


def test_m4_prime_partitions_unity_and_reproduces_first_moment():
    for phase in np.linspace(-0.95, 0.95, 31):
        nodes = np.arange(np.floor(phase) - 1, np.floor(phase) + 3)
        weight = m4_prime(phase - nodes)
        np.testing.assert_allclose(weight.sum(), 1.0, rtol=0.0, atol=2.0e-15)
        np.testing.assert_allclose(nodes @ weight, phase, rtol=0.0, atol=2.0e-15)


def test_fixed_lattice_scatter_is_complete_and_conservative_at_belt_faces():
    spacing = 0.125
    lattice = build_stable_renewal_lattice(
        BOX,
        spacing,
        buffer_length=0.2,
        authority_ramp_width=0.25,
        vpm_dead_zone=0.05,
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

    before = vortex_invariants(position, strength)
    after = vortex_invariants(lattice.positions, target)
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


def test_aligned_duplicate_particles_are_inserted_directly_on_one_node():
    lattice = build_stable_renewal_lattice(
        BOX,
        0.1,
        buffer_length=0.2,
        authority_ramp_width=0.2,
        lattice_anchor=np.zeros(3),
    )
    position = np.array([[0.2, -0.1, 0.3], [0.2, -0.1, 0.3]])
    strength = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]])

    target = scatter_m4_prime_to_lattice(position, strength, lattice)

    active = np.flatnonzero(np.linalg.norm(target, axis=1) > 0.0)
    assert len(active) == 1
    np.testing.assert_allclose(lattice.positions[active[0]], position[0])
    np.testing.assert_allclose(target[active[0]], strength.sum(axis=0))


def test_velocity_trace_recovers_the_curl_of_a_linear_velocity_exactly():
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

    strength = vortex_strength_from_velocity_trace(position, spacing, velocity)
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


def test_authority_is_inward_and_leaves_the_surface_under_vpm_control():
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
    authority = inward_cosine_authority(points, BOX, 0.2, 0.05)

    np.testing.assert_allclose(authority[:2], 1.0)
    assert 0.0 < authority[2] < 1.0
    np.testing.assert_allclose(authority[3:], 0.0)


def test_bounded_local_correction_improves_gaussian_represented_state():
    shape = (9, 9, 9)
    rng = np.random.default_rng(11)
    vpm_strength = rng.normal(scale=1.0e-4, size=(np.prod(shape), 3))
    fvm_target = rng.normal(scale=1.0e-4, size=(np.prod(shape), 3))
    authority = np.linspace(0.0, 1.0, np.prod(shape))

    blend = blend_represented_state(
        vpm_strength,
        fvm_target,
        authority,
        shape,
        0.1,
        core_radius=0.1,
        amplification_cap=2.0,
    )

    assert blend.residual_after_correction is not None
    assert blend.residual_after_correction < blend.residual_before_correction
    assert np.isfinite(blend.vortex_strength).all()


def test_soft_prune_is_continuous_and_zero_below_its_threshold():
    threshold = 1.0e-3
    magnitude = np.linspace(0.0, 5.0 * threshold, 4001)
    strength = np.zeros((len(magnitude), 3), dtype=np.float64)
    strength[:, 2] = magnitude

    shrunk, removed = soft_prune_vortex_strength(strength, threshold)

    assert np.all(shrunk[magnitude <= threshold] == 0.0)
    assert np.max(np.abs(np.diff(shrunk[:, 2]))) < 2.0 * (magnitude[1] - magnitude[0])
    np.testing.assert_allclose(shrunk + removed, strength, rtol=0.0, atol=1.0e-18)


def test_local_prune_redistribution_preserves_strength_and_linear_impulse():
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

    redistributed = redistribute_pruned_vortex_strength_locally(
        removed.reshape(-1, 3), retained.reshape(-1, 3), shape
    )

    before = vortex_invariants(positions, (retained + removed).reshape(-1, 3))
    after = vortex_invariants(positions, redistributed)
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


def test_whole_belt_remesh_preserves_outer_wake_and_does_not_accumulate():
    spacing = 0.1
    lattice = build_stable_renewal_lattice(
        BOX,
        spacing,
        buffer_length=0.2,
        authority_ramp_width=0.2,
        vpm_dead_zone=0.05,
        lattice_anchor=np.zeros(3),
    )
    position = np.array([[0.65, 0.13, -0.07], [0.95, 0.22, 0.11]])
    strength = np.array([[0.2, -0.1, 0.3], [-0.05, 0.07, 0.02]])

    first = renew_stable_overlap(
        position,
        strength,
        lattice,
        fvm_vortex_strength_at_node=_zero_target,
        amplification_cap=1.0,
        compute_diagnostics=False,
    )
    second = renew_stable_overlap(
        first.position,
        first.vortex_strength,
        lattice,
        fvm_vortex_strength_at_node=_zero_target,
        amplification_cap=1.0,
        compute_diagnostics=False,
    )

    assert first.renewed_input_count == 1
    assert first.preserved_outer_count == 1
    assert second.particle_count == first.particle_count
    np.testing.assert_allclose(second.position, first.position, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(second.vortex_strength, first.vortex_strength, atol=2.0e-15)
    np.testing.assert_allclose(first.core_radius, spacing)
    outer = np.flatnonzero(np.all(first.position == position[1], axis=1))
    assert len(outer) == 1
    np.testing.assert_allclose(first.vortex_strength[outer[0]], strength[1])

    before = vortex_invariants(position, strength)
    after = vortex_invariants(first.position, first.vortex_strength)
    np.testing.assert_allclose(
        after.total_vortex_strength, before.total_vortex_strength, atol=2e-14
    )
    np.testing.assert_allclose(after.linear_impulse, before.linear_impulse, atol=2e-14)
    assert first.conservation_raw_mismatch["total_vortex_strength"] < 2.0e-14
    assert first.conservation_applied_correction["total_vortex_strength"] < 2.0e-14
    assert first.conservation_residual["total_vortex_strength"] < 2.0e-14
    assert first.conservation_applied_particle_strength_fraction < 2.0e-14


def test_prune_diagnostics_expose_raw_mismatch_and_applied_closure():
    spacing = 0.1
    lattice = build_stable_renewal_lattice(
        BOX,
        spacing,
        buffer_length=0.2,
        authority_ramp_width=0.2,
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

    result = renew_stable_overlap(
        position,
        strength,
        lattice,
        fvm_vortex_strength_at_node=_zero_target,
        prune_threshold=0.05,
        amplification_cap=1.0,
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
    with pytest.raises(RuntimeError, match="closure correction is excessive"):
        replace_particles_from_buffered_m4_renewal(
            vpm,
            lattice=lattice,
            fvm_vortex_strength_at_node=_zero_target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            prune_threshold=0.05,
            core_radius_ratio=1.0,
            amplification_cap=1.0,
            boundary_prune_multiplier=1.0,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
            maximum_closure_correction_fraction=1.0e-6,
        )
    np.testing.assert_array_equal(vpm.particles.position, position)
    np.testing.assert_array_equal(vpm.particles.vortex_strength, strength)


def test_repeated_fvm_renewal_has_a_fixed_population_and_base_radii():
    spacing = 0.1
    lattice = build_stable_renewal_lattice(
        [-0.3, 0.3, -0.3, 0.3, -0.3, 0.3],
        spacing,
        buffer_length=0.5,
        authority_ramp_width=0.2,
        vpm_dead_zone=0.05,
        lattice_anchor=np.zeros(3),
    )

    def target(points: np.ndarray) -> np.ndarray:
        radius_squared = np.einsum("ij,ij->i", points, points)
        strength = np.zeros_like(points)
        strength[:, 2] = 2.0e-4 * np.exp(-radius_squared / 0.08)
        return strength

    first = renew_stable_overlap(
        np.empty((0, 3)),
        np.empty((0, 3)),
        lattice,
        fvm_vortex_strength_at_node=target,
        prune_threshold=1.0e-12,
        compute_diagnostics=False,
    )
    second = renew_stable_overlap(
        first.position,
        first.vortex_strength,
        lattice,
        fvm_vortex_strength_at_node=target,
        prune_threshold=1.0e-12,
        compute_diagnostics=False,
    )

    assert first.particle_count > 0
    assert second.particle_count == first.particle_count
    np.testing.assert_allclose(second.position, first.position, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(second.core_radius, spacing)


def test_population_cap_preserves_total_strength_and_linear_impulse():
    rng = np.random.default_rng(3)
    position = rng.uniform([1.0, -0.5, -0.5], [2.0, 0.5, 0.5], (8, 3))
    strength = rng.normal(size=(8, 3)) * 0.05
    lattice = build_stable_renewal_lattice(
        BOX,
        0.1,
        buffer_length=0.0,
        authority_ramp_width=0.2,
        lattice_anchor=np.zeros(3),
    )

    result = renew_stable_overlap(
        position,
        strength,
        lattice,
        fvm_vortex_strength_at_node=_zero_target,
        maximum_particle_count=4,
        compute_diagnostics=False,
    )

    assert result.particle_count == 4
    assert result.population_pruned_count == 4
    before = vortex_invariants(position, strength)
    after = vortex_invariants(result.position, result.vortex_strength)
    np.testing.assert_allclose(
        after.total_vortex_strength, before.total_vortex_strength, atol=1e-12
    )
    np.testing.assert_allclose(after.linear_impulse, before.linear_impulse, atol=1e-12)
    assert np.isfinite(result.population_pruned_velocity_bound)
    assert result.population_conservation_raw_mismatch["total_vortex_strength"] > 0.0
    assert result.population_conservation_applied_correction["total_vortex_strength"] > 0.0
    assert result.population_conservation_residual["total_vortex_strength"] < 1.0e-12
    assert result.population_conservation_raw_mismatch["linear_impulse"] > 0.0
    assert result.population_conservation_applied_correction["linear_impulse"] > 0.0
    assert result.population_conservation_residual["linear_impulse"] < 1.0e-12
    assert result.population_conservation_applied_particle_strength_fraction > 0.0

    vpm = _GBDVPM(position, strength)
    vpm.particles.capacity = 4
    with pytest.raises(RuntimeError, match="reached the VPM particle capacity"):
        replace_particles_from_buffered_m4_renewal(
            vpm,
            lattice=lattice,
            fvm_vortex_strength_at_node=_zero_target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            prune_threshold=0.0,
            core_radius_ratio=1.0,
            amplification_cap=1.0,
            boundary_prune_multiplier=1.0,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
            maximum_closure_correction_fraction=1.0e-6,
        )
    np.testing.assert_array_equal(vpm.particles.position, position)
    np.testing.assert_array_equal(vpm.particles.vortex_strength, strength)


def test_production_wrapper_replaces_one_complete_gbd_cloud_without_accumulation():
    spacing = 0.1
    lattice = build_stable_renewal_lattice(
        BOX,
        spacing,
        buffer_length=0.2,
        authority_ramp_width=0.2,
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
        result = replace_particles_from_buffered_m4_renewal(
            vpm,
            lattice=lattice,
            fvm_vortex_strength_at_node=target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            prune_threshold=1.0e-8,
            core_radius_ratio=1.0,
            amplification_cap=1.8,
            boundary_prune_multiplier=10.0,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
        )
        counts.append(result.n_particles_after)
        assert result.transfer_method == "buffered_m4_renewal"
        assert result.preserved_outer_particles == 1
        assert vpm.last_replacement is not None
        assert vpm.last_replacement["report_removal"] is False
        np.testing.assert_allclose(vpm.last_replacement["core_radius"], spacing)
        np.testing.assert_allclose(vpm.last_replacement["particle_volume"], spacing**3)

    assert counts[1:] == [counts[0], counts[0]]
    outer = np.flatnonzero(np.isclose(vpm.particles.position[:, 0], 0.95))
    assert len(outer) == 1
    np.testing.assert_allclose(vpm.particles.vortex_strength[outer[0]], [0.0, 0.0, 2.0e-4])


def test_production_wrapper_certifies_the_actual_float32_particle_state():
    spacing = 0.1
    lattice = build_stable_renewal_lattice(
        BOX,
        spacing,
        buffer_length=0.2,
        authority_ramp_width=0.2,
        lattice_anchor=np.full(3, -0.05),
    )

    def target(points: np.ndarray) -> np.ndarray:
        strength = np.empty_like(points)
        strength[:, 0] = 1.0e-4 * (1.0 + 0.13 * points[:, 1])
        strength[:, 1] = -2.0e-4 * (1.0 - 0.07 * points[:, 2])
        strength[:, 2] = 3.0e-4 * (1.0 + 0.11 * points[:, 0])
        return strength

    numerical = renew_stable_overlap(
        np.empty((0, 3)),
        np.empty((0, 3)),
        lattice,
        fvm_vortex_strength_at_node=target,
        prune_threshold=0.0,
        amplification_cap=1.0,
        compute_diagnostics=False,
    )
    assert numerical.conservation_target_invariants is not None
    vpm = _GBDVPM(np.empty((0, 3)), np.empty((0, 3)), dtype=np.float32)

    transfer = replace_particles_from_buffered_m4_renewal(
        vpm,
        lattice=lattice,
        fvm_vortex_strength_at_node=target,
        particle_fluid_weight=None,
        particle_in_solid=None,
        prune_threshold=0.0,
        core_radius_ratio=1.0,
        amplification_cap=1.0,
        boundary_prune_multiplier=1.0,
        kinematic_viscosity=1.0e-3,
        freestream_speed=1.0,
        time_step_size=0.01,
        compute_diagnostics=False,
        maximum_closure_correction_fraction=0.08,
    )

    actual = vortex_invariants(vpm.particles.position, vpm.particles.vortex_strength)
    expected = numerical.conservation_target_invariants
    strength_error = np.linalg.norm(actual.total_vortex_strength - expected.total_vortex_strength)
    impulse_error = np.linalg.norm(actual.linear_impulse - expected.linear_impulse)
    assert vpm.particles.vortex_strength.dtype == np.float32
    assert transfer.renewal_conservation_error == pytest.approx(strength_error)
    assert transfer.renewal_linear_impulse_error == pytest.approx(impulse_error)
    assert transfer.renewal_conservation_error <= transfer.renewal_vortex_strength_tolerance
    assert transfer.renewal_linear_impulse_error <= transfer.renewal_linear_impulse_tolerance


def test_support_output_coalesces_with_a_persistent_particle_on_the_same_lattice_node():
    spacing = 0.1
    lattice = build_stable_renewal_lattice(
        BOX,
        spacing,
        buffer_length=0.2,
        authority_ramp_width=0.2,
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

    result = renew_stable_overlap(
        position,
        strength,
        lattice,
        fvm_vortex_strength_at_node=_zero_target,
        prune_threshold=0.0,
        amplification_cap=1.0,
        compute_diagnostics=False,
    )

    hits = np.all(np.isclose(result.position, position[1], rtol=0.0, atol=1.0e-14), axis=1)
    assert np.count_nonzero(hits) == 1
    assert result.coalesced_outer_count == 1
    assert result.preserved_outer_count == 0
    before = vortex_invariants(position, strength)
    after = vortex_invariants(result.position, result.vortex_strength)
    np.testing.assert_allclose(
        after.total_vortex_strength,
        before.total_vortex_strength,
        rtol=0.0,
        atol=2.0e-13,
    )

    vpm = _GBDVPM(position, strength)
    transfer = replace_particles_from_buffered_m4_renewal(
        vpm,
        lattice=lattice,
        fvm_vortex_strength_at_node=_zero_target,
        particle_fluid_weight=None,
        particle_in_solid=None,
        prune_threshold=0.0,
        core_radius_ratio=1.0,
        amplification_cap=1.0,
        boundary_prune_multiplier=1.0,
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
    lattice = build_stable_renewal_lattice(
        BOX,
        spacing,
        buffer_length=0.2,
        authority_ramp_width=0.2,
        lattice_anchor=np.full(3, -0.05),
        mesh_weight_at_node=lambda points: np.zeros(len(points)),
    )
    position = np.array([[0.699, -0.05, -0.05], [0.75, -0.05, -0.05]])
    strength = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.2]])
    numerical = renew_stable_overlap(
        position,
        strength,
        lattice,
        fvm_vortex_strength_at_node=_zero_target,
        prune_threshold=0.0,
        amplification_cap=1.0,
        compute_diagnostics=False,
    )
    invalid = replace(
        numerical,
        coalesced_outer_input_indices=np.array([len(position)], dtype=np.int64),
    )
    monkeypatch.setattr(transfer_module, "renew_stable_overlap", lambda *args, **kwargs: invalid)
    vpm = _GBDVPM(position, strength)

    with pytest.raises(RuntimeError, match="invalid coalesced input index"):
        replace_particles_from_buffered_m4_renewal(
            vpm,
            lattice=lattice,
            fvm_vortex_strength_at_node=_zero_target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            prune_threshold=0.0,
            core_radius_ratio=1.0,
            amplification_cap=1.0,
            boundary_prune_multiplier=1.0,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
        )

    assert vpm.last_replacement is None
    np.testing.assert_array_equal(vpm.particles.position, position)
    np.testing.assert_array_equal(vpm.particles.vortex_strength, strength)


def test_support_seam_coalesces_multiple_particles_but_preserves_a_near_miss_across_steps():
    spacing = 0.1
    lattice = build_stable_renewal_lattice(
        BOX,
        spacing,
        buffer_length=0.2,
        authority_ramp_width=0.2,
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
        transfer = replace_particles_from_buffered_m4_renewal(
            vpm,
            lattice=lattice,
            fvm_vortex_strength_at_node=_zero_target,
            particle_fluid_weight=None,
            particle_in_solid=None,
            prune_threshold=0.0,
            core_radius_ratio=1.0,
            amplification_cap=1.0,
            boundary_prune_multiplier=1.0,
            kinematic_viscosity=1.0e-3,
            freestream_speed=1.0,
            time_step_size=0.01,
            compute_diagnostics=False,
            maximum_closure_correction_fraction=1.0,
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
