"""Device-hierarchy and covariance tests for production VPM FMM."""

from __future__ import annotations

import math

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.io.logging import Logging
from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.numerics.rk_tableaux import RK2, RK4, SSPRK3
from source.solvers.vpm.numerics.runge_kutta import RungeKutta
from source.solvers.vpm.particles import Particles
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.induction.fmm import FMMInduction
from source.solvers.vpm.physics.induction.fmm.device import (
    _DERIVATIVE_INDICES,
    _MULTI_INDICES,
    _translation_tables,
)


def _ensure_taichi_cpu() -> None:
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, offline_cache=False, cpu_max_num_threads=2)


def test_fmm_workspace_estimate_is_linear_in_capacity():
    induction = FMMInduction()
    estimates = [induction.estimated_workspace_bytes(capacity) for capacity in (1, 10, 100)]
    assert estimates[1] > estimates[0]
    assert estimates[2] > estimates[1]
    assert estimates[2] - estimates[1] == 10 * (estimates[1] - estimates[0])


def test_particle_capacity_warning_is_emitted_at_eighty_percent(monkeypatch):
    _ensure_taichi_cpu()
    warnings = []
    monkeypatch.setattr(Logging, "warning", lambda text, **kwargs: warnings.append(text))
    particles = Particles(max_n_particles=10, float_dtype="f32")
    particles.add_vortex_particles(
        position=np.zeros((8, 3), dtype=np.float32),
        velocity=np.zeros((8, 3), dtype=np.float32),
        vortex_strength=np.ones((8, 3), dtype=np.float32),
        core_radius=np.ones(8, dtype=np.float32),
        particle_volume=np.ones(8, dtype=np.float32),
        kinematic_viscosity=np.zeros(8, dtype=np.float32),
    )
    assert len(warnings) == 1
    assert "active particle count=8" in warnings[0]


def test_particle_capacity_error_precedes_overflow():
    _ensure_taichi_cpu()
    particles = Particles(max_n_particles=2, float_dtype="f32")
    with pytest.raises(ValueError, match="current particle count=0.*requested new particles=3"):
        particles.add_vortex_particles(
            position=np.zeros((3, 3), dtype=np.float32),
            velocity=np.zeros((3, 3), dtype=np.float32),
            vortex_strength=np.ones((3, 3), dtype=np.float32),
            core_radius=np.ones(3, dtype=np.float32),
            particle_volume=np.ones(3, dtype=np.float32),
            kinematic_viscosity=np.zeros(3, dtype=np.float32),
        )


class _DeviceFMMHarness:
    def __init__(self, capacity: int = 64, kernel_name: str = "GAUSSIAN") -> None:
        _ensure_taichi_cpu()
        self.capacity = capacity
        physics = PhysicsBase(
            particle_kernel=kernel_name,
            max_n_particles=capacity,
            accumulator_dtype=ti.f32,
            max_evaluation_points=capacity,
        )
        self.induction = FMMInduction().bind(physics, kernel=make_vortex_kernel(kernel_name))
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self.strength = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self.radius = ti.field(dtype=ti.f32, shape=capacity)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self.gradient = ti.Matrix.field(3, 3, dtype=ti.f32, shape=capacity)
        self.rate = ti.Vector.field(3, dtype=ti.f32, shape=capacity)

    def evaluate(self, position, strength, radius):
        count = len(position)
        position_buffer = np.zeros((self.capacity, 3), dtype=np.float32)
        strength_buffer = np.zeros((self.capacity, 3), dtype=np.float32)
        radius_buffer = np.ones(self.capacity, dtype=np.float32)
        position_buffer[:count] = position
        strength_buffer[:count] = strength
        radius_buffer[:count] = radius
        self.position.from_numpy(position_buffer)
        self.strength.from_numpy(strength_buffer)
        self.radius.from_numpy(radius_buffer)
        self.induction.evaluate_stage(
            position=self.position,
            vortex_strength=self.strength,
            core_radius=self.radius,
            count=count,
            velocity_out=self.velocity,
            vortex_strength_rate_out=self.rate,
            velocity_gradient_out=self.gradient,
        )
        return (
            self.velocity.to_numpy()[:count].copy(),
            self.gradient.to_numpy()[:count].copy(),
            self.rate.to_numpy()[:count].copy(),
        )


class _RecordingFMMRHS:
    def __init__(self, induction) -> None:
        self.induction = induction
        self.calls = []

    def evaluate(self, stage_state: StageState, stage_time: float, stage_rates: StageRates) -> None:
        self.calls.append(
            (
                stage_time,
                stage_state.position.to_numpy()[: stage_state.count].copy(),
                stage_state.vortex_strength.to_numpy()[: stage_state.count].copy(),
            )
        )
        self.induction.evaluate_stage(
            position=stage_state.position,
            vortex_strength=stage_state.vortex_strength,
            core_radius=stage_state.core_radius,
            count=stage_state.count,
            velocity_out=stage_rates.velocity,
            vortex_strength_rate_out=stage_rates.vortex_strength_rate,
            velocity_gradient_out=stage_rates.velocity_gradient,
            stage_time=stage_time,
        )


def _analytic_inverse_r_derivative(displacement, alpha):
    _, term_count, coefficient, exponent, radial_step = _translation_tables()
    derivative_index = _DERIVATIVE_INDICES.index(alpha)
    radius = np.linalg.norm(displacement)
    value = 0.0
    for term in range(term_count[derivative_index]):
        value += (
            coefficient[derivative_index, term]
            * np.prod(displacement ** exponent[derivative_index, term])
            / radius ** (1 + 2 * radial_step[derivative_index, term])
        )
    return value


def _cartesian_far_field_at_centre(position, strength, target, order):
    centre = np.zeros(3)
    source_offset = position - centre
    moments = {}
    for alpha in _MULTI_INDICES:
        if sum(alpha) <= order:
            moments[alpha] = (
                strength
                * source_offset[:, 0, None] ** alpha[0]
                * source_offset[:, 1, None] ** alpha[1]
                * source_offset[:, 2, None] ** alpha[2]
                / math.factorial(alpha[0])
                / math.factorial(alpha[1])
                / math.factorial(alpha[2])
            ).sum(axis=0)
    local = {}
    for beta in _MULTI_INDICES:
        if sum(beta) > 2:
            continue
        translated = np.zeros(3)
        for alpha, moment in moments.items():
            derivative = tuple(alpha[axis] + beta[axis] for axis in range(3))
            translated += (
                (-1.0) ** sum(alpha)
                * moment
                * _analytic_inverse_r_derivative(
                    target - centre,
                    derivative,
                )
            )
        local[beta] = translated / math.prod(math.factorial(value) for value in beta)

    a_x, a_y, a_z = local[(1, 0, 0)], local[(0, 1, 0)], local[(0, 0, 1)]
    a_xx = 2.0 * local[(2, 0, 0)]
    a_xy = local[(1, 1, 0)]
    a_xz = local[(1, 0, 1)]
    a_yy = 2.0 * local[(0, 2, 0)]
    a_yz = local[(0, 1, 1)]
    a_zz = 2.0 * local[(0, 0, 2)]
    velocity = np.array([a_y[2] - a_z[1], a_z[0] - a_x[2], a_x[1] - a_y[0]])
    gradient = np.array(
        [
            [a_xy[2] - a_xz[1], a_yy[2] - a_yz[1], a_yz[2] - a_zz[1]],
            [a_xz[0] - a_xx[2], a_yz[0] - a_xy[2], a_zz[0] - a_xz[2]],
            [a_xx[1] - a_xy[0], a_xy[1] - a_yy[0], a_xz[1] - a_yz[0]],
        ]
    )
    return velocity, gradient


def test_analytic_m2l_oracle_converges_as_the_private_source_order_increases():
    rng = np.random.default_rng(20260907)
    position = rng.normal(scale=0.08, size=(40, 3))
    strength = rng.normal(scale=0.02, size=(40, 3))
    target = np.array([3.0, 2.0, 4.0])
    displacement = target[None, :] - position
    kernel = make_vortex_kernel("GAUSSIAN")
    radius = np.full(len(position), 1.0e-6)
    reference_velocity = kernel.velocity_pair(
        displacement,
        strength,
        np.array([1.0e-6]),
        radius,
    ).sum(axis=0)
    reference_gradient = kernel.gradient_pair(
        displacement,
        strength,
        np.array([1.0e-6]),
        radius,
    ).sum(axis=0)
    errors = []
    for order in range(4):
        velocity, gradient = _cartesian_far_field_at_centre(position, strength, target, order)
        errors.append(
            np.linalg.norm(velocity - reference_velocity) / np.linalg.norm(reference_velocity)
            + np.linalg.norm(gradient - reference_gradient) / np.linalg.norm(reference_gradient)
        )

    assert errors[3] < errors[2] < errors[1] < errors[0]


def test_device_hierarchy_handles_edge_cases_and_rebuilds_stage_metadata():
    harness = _DeviceFMMHarness()
    rng = np.random.default_rng(20260904)
    cases = (
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.zeros((8, 3), dtype=np.float32),
        np.array(
            [[-1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
            dtype=np.float32,
        ),
        rng.normal(scale=1.0e-4, size=(32, 3)).astype(np.float32),
        np.column_stack(
            (np.linspace(-100.0, 100.0, 32), np.zeros(32), np.linspace(-1.0e-3, 1.0e-3, 32))
        ).astype(np.float32),
    )

    for case_index, position in enumerate(cases):
        count = len(position)
        strength = rng.normal(scale=0.01, size=(count, 3)).astype(np.float32)
        radius = np.linspace(0.005, 0.025, count, dtype=np.float32)
        velocity, gradient, rate = harness.evaluate(position, strength, radius)
        tree = harness.induction.workspace.tree
        root = int(tree._root[None])

        assert int(tree.n_nodes[None]) == 2 * count - 1
        np.testing.assert_array_equal(
            np.sort(tree.sorted_indices.to_numpy()[:count]), np.arange(count)
        )
        np.testing.assert_array_equal(tree.node_particle_start.to_numpy()[:count], np.arange(count))
        np.testing.assert_array_equal(tree.node_particle_count.to_numpy()[:count], np.ones(count))
        assert int(tree.node_particle_count[root]) == count
        np.testing.assert_allclose(float(tree.node_max_radius[root]), radius.max(), rtol=1.0e-6)
        assert np.all(np.isfinite(velocity))
        assert np.all(np.isfinite(gradient))
        assert np.all(np.isfinite(rate))

        if case_index == 2:
            topology = (
                tree.sorted_indices.to_numpy()[:count].copy(),
                tree.node_parent.to_numpy()[: 2 * count - 1].copy(),
                tree.node_left.to_numpy()[: 2 * count - 1].copy(),
                tree.node_right.to_numpy()[: 2 * count - 1].copy(),
            )
            harness.evaluate(position, strength, radius)
            rebuilt = (
                tree.sorted_indices.to_numpy()[:count],
                tree.node_parent.to_numpy()[: 2 * count - 1],
                tree.node_left.to_numpy()[: 2 * count - 1],
                tree.node_right.to_numpy()[: 2 * count - 1],
            )
            for first, second in zip(topology, rebuilt, strict=True):
                np.testing.assert_array_equal(first, second)

    position = rng.uniform(-1.0, 1.0, size=(32, 3)).astype(np.float32)
    strength = rng.normal(scale=0.01, size=(32, 3)).astype(np.float32)
    radius = rng.uniform(0.005, 0.025, size=32).astype(np.float32)
    harness.evaluate(position, strength, radius)
    workspace = harness.induction.workspace
    root = int(workspace.tree._root[None])
    first_centre = np.array(workspace.tree.node_centre[root], dtype=np.float32)
    root_moments = workspace.multipole.to_numpy().reshape(workspace.max_nodes, 20, 3)[root]
    displacement = position - first_centre
    for coefficient, (a, b, c) in enumerate(_MULTI_INDICES):
        expected = (
            strength
            * displacement[:, 0, None] ** a
            * displacement[:, 1, None] ** b
            * displacement[:, 2, None] ** c
            / math.factorial(a)
            / math.factorial(b)
            / math.factorial(c)
        ).sum(axis=0)
        np.testing.assert_allclose(root_moments[coefficient], expected, rtol=2.0e-5, atol=2.0e-7)
    first_moment = root_moments[0]

    changed_position = position.copy()
    changed_position[:, 0] += np.linspace(0.0, 0.5, len(position), dtype=np.float32)
    harness.evaluate(changed_position, 2.0 * strength, radius)
    second_centre = np.array(workspace.tree.node_centre[root], dtype=np.float32)
    second_moment = np.array(workspace.multipole[root * 20], dtype=np.float32)
    assert not np.array_equal(first_centre, second_centre)
    np.testing.assert_allclose(second_moment, 2.0 * first_moment, rtol=2.0e-5, atol=2.0e-7)


def test_device_fmm_is_permutation_translation_and_axis_rotation_covariant():
    harness = _DeviceFMMHarness()
    rng = np.random.default_rng(20260905)
    position = rng.normal(scale=0.08, size=(64, 3)).astype(np.float32)
    position[:32, 0] -= 4.0
    position[32:, 0] += 4.0
    strength = rng.normal(scale=0.01, size=(64, 3)).astype(np.float32)
    radius = rng.uniform(0.008, 0.016, size=64).astype(np.float32)

    velocity, gradient, rate = harness.evaluate(position, strength, radius)
    assert int(harness.induction.workspace._m2l_count[None]) > 0
    assert int(harness.induction.workspace._nonzero_l2l_count[None]) > 0

    permutation = rng.permutation(len(position))
    permuted_velocity, permuted_gradient, permuted_rate = harness.evaluate(
        position[permutation], strength[permutation], radius[permutation]
    )
    restored_velocity = np.empty_like(permuted_velocity)
    restored_gradient = np.empty_like(permuted_gradient)
    restored_rate = np.empty_like(permuted_rate)
    restored_velocity[permutation] = permuted_velocity
    restored_gradient[permutation] = permuted_gradient
    restored_rate[permutation] = permuted_rate
    np.testing.assert_allclose(restored_velocity, velocity, rtol=5.0e-5, atol=2.0e-7)
    np.testing.assert_allclose(restored_gradient, gradient, rtol=5.0e-5, atol=2.0e-7)
    np.testing.assert_allclose(restored_rate, rate, rtol=5.0e-5, atol=2.0e-7)

    translated = harness.evaluate(
        position + np.array([0.25, -0.125, 0.5], dtype=np.float32), strength, radius
    )
    for actual, reference in zip(translated, (velocity, gradient, rate), strict=True):
        relative_error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
        assert relative_error < 2.0e-5

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rotated_velocity, rotated_gradient, rotated_rate = harness.evaluate(
        position @ rotation.T,
        strength @ rotation.T,
        radius,
    )
    expected_gradient = np.einsum("ab,nbc,dc->nad", rotation, gradient, rotation)
    np.testing.assert_allclose(rotated_velocity, velocity @ rotation.T, rtol=4.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(rotated_gradient, expected_gradient, rtol=4.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(rotated_rate, rate @ rotation.T, rtol=4.0e-4, atol=1.0e-5)


@pytest.mark.parametrize(
    "kernel_name",
    ("GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"),
)
def test_device_fmm_meets_all_kernel_gates_with_near_pairs_and_far_clusters(kernel_name):
    count = 512
    harness = _DeviceFMMHarness(capacity=count, kernel_name=kernel_name)
    rng = np.random.default_rng(20260906)
    position = rng.normal(scale=0.06, size=(count, 3)).astype(np.float32)
    position[: count // 2, 0] -= 4.0
    position[count // 2 :, 0] += 4.0
    strength = rng.normal(scale=0.01, size=(count, 3)).astype(np.float32)
    kernel = make_vortex_kernel(kernel_name)

    for radius in (
        np.full(count, 0.012, dtype=np.float32),
        rng.uniform(0.008, 0.016, size=count).astype(np.float32),
    ):
        # Keep an explicitly near-core pair in each otherwise random cluster.
        position[1] = position[0] + np.array([0.25 * radius[0], 0.0, 0.0], dtype=np.float32)
        velocity, gradient, rate = harness.evaluate(position, strength, radius)
        displacement = position[:, None, :] - position[None, :, :]
        reference_velocity = kernel.velocity_pair(
            displacement,
            strength[None, :, :],
            radius[:, None],
            radius[None, :],
        ).sum(axis=1)
        reference_gradient = kernel.gradient_pair(
            displacement,
            strength[None, :, :],
            radius[:, None],
            radius[None, :],
        ).sum(axis=1)
        reference_rate = np.einsum("nji,nj->ni", reference_gradient, strength)
        reference_rate_norm = np.linalg.norm(reference_rate, axis=1)
        denominator_floor = max(
            float(np.sqrt(np.mean(reference_rate_norm**2))) * 1.0e-3,
            1.0e-14,
        )
        particle_rate_error = np.linalg.norm(rate - reference_rate, axis=1) / np.maximum(
            reference_rate_norm,
            denominator_floor,
        )

        assert (
            np.linalg.norm(velocity - reference_velocity) / np.linalg.norm(reference_velocity)
            <= 5.0e-3
        )
        assert (
            np.linalg.norm(gradient - reference_gradient) / np.linalg.norm(reference_gradient)
            <= 1.0e-2
        )
        assert np.linalg.norm(rate - reference_rate) / np.linalg.norm(reference_rate) <= 1.5e-2
        assert np.percentile(particle_rate_error, 95.0) <= 3.0e-2
        assert int(harness.induction.workspace._m2l_count[None]) > 0
        assert int(harness.induction.workspace._near_count[None]) > 0
        assert harness.induction.diagnostics.last_relative_rate_defect <= 1.0e-3
        assert harness.induction.diagnostics.host_particle_transfers == 0


def test_every_coupled_rk_stage_rebuilds_fmm_from_the_common_temporary_state():
    harness = _DeviceFMMHarness(capacity=2)
    position = np.array([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
    strength = np.array([[0.0, 0.1, 0.04], [0.03, -0.05, 0.02]], dtype=np.float32)
    radius = np.full(2, 0.1, dtype=np.float32)

    for tableau in (RK2(), SSPRK3(), RK4()):
        harness.position.from_numpy(position)
        harness.strength.from_numpy(strength)
        harness.radius.from_numpy(radius)
        right_hand_side = _RecordingFMMRHS(harness.induction)
        integrator = RungeKutta(tableau, max_n_particles=2, dtype=ti.f32)
        prior_stages = harness.induction.diagnostics.stage_evaluations
        prior_builds = harness.induction.diagnostics.hierarchy_builds
        integrator.advance(
            position=harness.position,
            vortex_strength=harness.strength,
            core_radius=harness.radius,
            count=2,
            time=1.25,
            time_step_size=0.01,
            right_hand_side=right_hand_side,
        )

        assert len(right_hand_side.calls) == tableau.stages
        np.testing.assert_allclose(
            [call[0] for call in right_hand_side.calls],
            1.25 + 0.01 * np.asarray(tableau.c),
        )
        np.testing.assert_array_equal(right_hand_side.calls[0][1], position)
        np.testing.assert_array_equal(right_hand_side.calls[0][2], strength)
        assert any(
            not np.array_equal(call[1], position) and not np.array_equal(call[2], strength)
            for call in right_hand_side.calls[1:]
        )
        assert harness.induction.diagnostics.stage_evaluations - prior_stages == tableau.stages
        assert harness.induction.diagnostics.hierarchy_builds - prior_builds == tableau.stages
        assert harness.induction.diagnostics.direct_strength_rate_fallbacks == 0
        assert harness.induction.diagnostics.host_particle_transfers == 0
