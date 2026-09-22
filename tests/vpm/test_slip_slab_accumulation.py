"""Reflected image reduction agrees with an independent array sum."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction, _add_reflected_results


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_reflected_batch_reduces_velocity_gradient_and_stretching(dtype, mode):
    array_dtype = np.float32 if dtype == ti.f32 else np.float64
    rng = np.random.default_rng(370 + mode)
    points, images, start, capacity = 5, 7, 2, 9
    raw_velocity = rng.normal(size=(images, points, 3)).astype(array_dtype)
    raw_gradient = rng.normal(size=(images, points, 3, 3)).astype(array_dtype)
    strengths = rng.normal(size=(capacity, 3)).astype(array_dtype)
    odds = np.arange(64, dtype=np.int32) % 2

    image_velocity = ti.Vector.field(3, dtype, shape=images * points)
    image_gradient = ti.Matrix.field(3, 3, dtype, shape=images * points)
    image_velocity.from_numpy(raw_velocity.reshape(-1, 3))
    image_gradient.from_numpy(raw_gradient.reshape(-1, 3, 3))
    odd_flags = ti.field(ti.i32, shape=64)
    odd_flags.from_numpy(odds)
    shifts = ti.field(dtype, shape=64)
    strength = ti.Vector.field(3, dtype, shape=capacity)
    strength.from_numpy(strengths)
    velocity = ti.Vector.field(3, dtype, shape=capacity)
    gradient = ti.Matrix.field(3, 3, dtype, shape=capacity)
    shell_velocity = ti.Vector.field(3, dtype, shape=capacity)
    shell_gradient = ti.Matrix.field(3, 3, dtype, shape=capacity)
    rate = ti.Vector.field(3, dtype, shape=capacity)
    initial_velocity = rng.normal(size=(capacity, 3)).astype(array_dtype)
    initial_gradient = rng.normal(size=(capacity, 3, 3)).astype(array_dtype)
    initial_shell_velocity = rng.normal(size=(capacity, 3)).astype(array_dtype)
    initial_shell_gradient = rng.normal(size=(capacity, 3, 3)).astype(array_dtype)
    initial_rate = rng.normal(size=(capacity, 3)).astype(array_dtype)
    velocity.from_numpy(initial_velocity)
    gradient.from_numpy(initial_gradient)
    shell_velocity.from_numpy(initial_shell_velocity)
    shell_gradient.from_numpy(initial_shell_gradient)
    rate.from_numpy(initial_rate)

    _add_reflected_results(
        image_velocity,
        image_gradient,
        shifts,
        odd_flags,
        strength,
        velocity,
        rate,
        gradient,
        shell_velocity,
        shell_gradient,
        start,
        points,
        images,
        mode,
        1,
        True,
        True,
        True,
    )

    reflected_velocity = raw_velocity.copy()
    reflected_gradient = raw_gradient.copy()
    reflected_velocity[odds[:images] == 1, :, 2] *= -1
    for axis_a in range(3):
        for axis_b in range(3):
            if (axis_a == 2) != (axis_b == 2):
                reflected_gradient[odds[:images] == 1, :, axis_a, axis_b] *= -1
    expected_velocity = reflected_velocity.sum(axis=0)
    expected_gradient = reflected_gradient.sum(axis=0)
    if mode == 0:
        expected_rate = np.einsum(
            "nab,nb->na", expected_gradient, strengths[start : start + points]
        )
    elif mode == 1:
        expected_rate = np.einsum(
            "nba,nb->na", expected_gradient, strengths[start : start + points]
        )
    else:
        symmetric = 0.5 * (expected_gradient + expected_gradient.transpose(0, 2, 1))
        expected_rate = np.einsum("nab,nb->na", symmetric, strengths[start : start + points])

    tolerance = 2e-5 if dtype == ti.f32 else 1e-12
    for actual, initial, contribution in (
        (velocity.to_numpy(), initial_velocity, expected_velocity),
        (shell_velocity.to_numpy(), initial_shell_velocity, expected_velocity),
        (gradient.to_numpy(), initial_gradient, expected_gradient),
        (shell_gradient.to_numpy(), initial_shell_gradient, expected_gradient),
        (rate.to_numpy(), initial_rate, expected_rate),
    ):
        np.testing.assert_allclose(
            actual[start : start + points],
            initial[start : start + points] + contribution,
            atol=tolerance,
            rtol=tolerance,
        )
        np.testing.assert_array_equal(actual[:start], initial[:start])
        np.testing.assert_array_equal(actual[start + points :], initial[start + points :])


def test_disabled_outputs_still_accumulate_shell_only():
    points, images = 3, 2
    image_velocity = ti.Vector.field(3, ti.f32, shape=points * images)
    image_gradient = ti.Matrix.field(3, 3, ti.f32, shape=points * images)
    image_velocity.from_numpy(np.ones((points * images, 3), dtype=np.float32))
    image_gradient.from_numpy(np.ones((points * images, 3, 3), dtype=np.float32))
    shifts = ti.field(ti.f32, shape=64)
    odds = ti.field(ti.i32, shape=64)
    odds.from_numpy(np.zeros(64, dtype=np.int32))
    strength = ti.Vector.field(3, ti.f32, shape=points)
    velocity = ti.Vector.field(3, ti.f32, shape=points)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=points)
    rate = ti.Vector.field(3, ti.f32, shape=points)
    shell_velocity = ti.Vector.field(3, ti.f32, shape=points)
    shell_gradient = ti.Matrix.field(3, 3, ti.f32, shape=points)
    velocity.fill(3.0)
    gradient.fill(4.0)
    rate.fill(5.0)
    shell_velocity.fill(6.0)
    shell_gradient.fill(7.0)

    _add_reflected_results(
        image_velocity,
        image_gradient,
        shifts,
        odds,
        strength,
        velocity,
        rate,
        gradient,
        shell_velocity,
        shell_gradient,
        0,
        points,
        images,
        0,
        1,
        False,
        False,
        False,
    )

    for field, value in (
        (velocity, 3.0),
        (gradient, 4.0),
        (rate, 5.0),
        (shell_velocity, 8.0),
        (shell_gradient, 9.0),
    ):
        actual = field.to_numpy()
        np.testing.assert_array_equal(actual, np.full_like(actual, value))


def test_target_tiling_preserves_full_slab_field_and_tail():
    source = ti.Vector.field(3, ti.f32, shape=2)
    strength = ti.Vector.field(3, ti.f32, shape=2)
    radius = ti.field(ti.f32, shape=2)
    target = ti.Vector.field(3, ti.f32, shape=7)
    source.from_numpy(np.array([[0.1, -0.2, 0.17], [-0.3, 0.05, -0.28]], dtype=np.float32))
    strength.from_numpy(np.array([[0.2, -0.3, 0.7], [-0.4, 0.15, -0.2]], dtype=np.float32))
    radius.from_numpy(np.array([0.12, 0.14], dtype=np.float32))
    target.from_numpy(
        np.array(
            [[0.1 * i, -0.03 * i, -0.48 + 0.16 * i] for i in range(7)],
            dtype=np.float32,
        )
    )

    results = []
    for query_capacity in (6, 64):
        physics = PhysicsBase("GAUSSIAN", 2, ti.f32, max_evaluation_points=query_capacity)
        slab = SlipSlabInduction(
            DirectInduction(),
            z_min=-0.5,
            z_max=0.5,
            tail_tolerance=1e-4,
            max_shells=65,
        ).bind(physics)
        velocity = ti.Vector.field(3, ti.f32, shape=7)
        gradient = ti.Matrix.field(3, 3, ti.f32, shape=7)
        slab.evaluate_targets(
            target_position=target,
            source_position=source,
            source_vortex_strength=strength,
            source_core_radius=radius,
            target_velocity=velocity,
            target_velocity_gradient=gradient,
            target_count=7,
            source_count=2,
            include_freestream=False,
            background_velocity=physics._zero_velocity,
        )
        results.append((velocity.to_numpy(), gradient.to_numpy(), slab.last_tail["shell"]))

    np.testing.assert_allclose(results[0][0], results[1][0], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-6, rtol=2e-6)
    assert results[0][2] == results[1][2]
