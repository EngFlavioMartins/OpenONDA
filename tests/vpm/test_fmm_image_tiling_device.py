"""Unwired target-tile execution check with a real device and exact images."""

import numpy as np
import pytest
import taichi as ti

from tests.vpm._fmm_image_tiling_prototype import image_tiling_factory
from tests.vpm.test_fmm_device import _DeviceFMMHarness
from tests.vpm.test_fmm_target_accuracy import _exact_fields_and_roundoff


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_real_image_tiles_preserve_every_target_and_direct_envelope(kernel):
    rng = np.random.default_rng(76225)
    count, first = 8199, 3
    position = rng.uniform(-0.15, 0.15, (8, 3)).astype(np.float32)
    strength = rng.normal(0, 0.01, position.shape).astype(np.float32)
    radius = np.full(len(position), 0.08, np.float32)
    points = rng.uniform(-0.4, 0.4, (first + count, 3)).astype(np.float32)
    harness = _DeviceFMMHarness(capacity=len(position), kernel_name=kernel)
    harness.evaluate(position, strength, radius)
    query = ti.Vector.field(3, ti.f32, shape=len(points))
    velocity = ti.Vector.field(3, ti.f32, shape=count + 2)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=count + 2)
    query.from_numpy(points)
    images = [(0.0, False), (1.0, True), (7.0, False)]
    expected, allowance = None, None
    # Source reflection maps vorticity as an axial vector. Use f64 direct
    # sources, never transformed-query data from the implementation under test.
    for shift, odd in images:
        reflected_position, reflected_strength = position.copy(), strength.copy()
        if odd:
            reflected_position[:, 2] *= -1
            reflected_strength[:, :2] *= -1
        reflected_position[:, 2] += shift
        fields, errors = _exact_fields_and_roundoff(
            kernel, reflected_position, reflected_strength, radius, points[first:]
        )
        if expected is None:
            expected, allowance = fields, errors
        else:
            expected = [old + new for old, new in zip(expected, fields, strict=True)]
            allowance = [old + new for old, new in zip(allowance, errors, strict=True)]
    results = []
    try:
        for capacity in (8192, 32768):
            velocity.fill(41)
            gradient.fill(43)
            with image_tiling_factory(capacity) as records:
                harness.induction.evaluate_image_block(
                    source_position=harness.position,
                    source_vortex_strength=harness.strength,
                    source_core_radius=harness.radius,
                    source_count=len(position),
                    target_position=query, target_start=first, target_count=count,
                    images=images, target_velocity=velocity,
                    target_velocity_gradient=gradient,
                )
                assert records[-1]["status"] == "complete"
                assert sum(tile["target_count"] for tile in records[-1]["tiles"]) == count
                assert len(records[-1]["tiles"]) == (2 if capacity == 8192 else 1)
            result = velocity.to_numpy(), gradient.to_numpy()
            np.testing.assert_array_equal(result[0][count:], 41)
            np.testing.assert_array_equal(result[1][count:], 43)
            results.append(tuple(field[:count] for field in result))
        for small, large, exact, budget in zip(
            results[0], results[1], expected, allowance, strict=True
        ):
            old_error = np.linalg.norm((small - exact).reshape(count, -1), axis=1)
            new_error = np.linalg.norm((large - exact).reshape(count, -1), axis=1)
            assert np.all(new_error <= old_error + budget)
            difference = np.linalg.norm((small - large).reshape(count, -1), axis=1)
            assert np.all(difference <= 2 * budget)
    finally:
        if harness.induction._target_workspace is not None:
            harness.induction._target_workspace.destroy()
        harness.induction.workspace.destroy()
