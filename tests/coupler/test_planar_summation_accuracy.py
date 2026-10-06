"""Separate cancellation error from the native planar pair calculation."""

from types import SimpleNamespace

import numpy as np
import taichi as ti

from openonda import vpm
from tests.support.cylinder.planar_summation_control import AccuratePlanarSums


def test_cancelled_gaussian_and_channel_sources_retain_small_net_circulation():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=1)
    try:
        count = 8193
        induction = vpm.PlanarChannelInduction(half_width=10).bind(
            SimpleNamespace(accumulator_dtype=ti.f32, max_n_particles=count)
        )
        position = ti.Vector.field(3, ti.f32, shape=count)
        strength = ti.Vector.field(3, ti.f32, shape=count)
        radius = ti.field(ti.f32, shape=count)
        target = ti.Vector.field(3, ti.f32, shape=1)
        velocity = ti.Vector.field(3, ti.f32, shape=1)
        gradient = ti.Matrix.field(3, 3, ti.f32, shape=1)
        position.from_numpy(np.tile(np.array([1, 0.3, 0], dtype=np.float32), (count, 1)))
        positive = np.linspace(1, 1.1, 4096, dtype=np.float32)
        axial = np.r_[positive, -positive, np.float32(0.001)]
        strength.from_numpy(
            np.column_stack((np.zeros(count), np.zeros(count), axial)).astype(np.float32)
        )
        radius.fill(0.04)
        target[0] = [0, 0, 0]
        values = {
            "target_position": target,
            "source_position": position,
            "source_vortex_strength": strength,
            "source_core_radius": radius,
            "target_velocity": velocity,
            "target_velocity_gradient": gradient,
            "target_count": 1,
            "source_count": count,
            "include_freestream": False,
            "background_velocity": (0, 0, 0),
        }
        induction.evaluate_targets(**values)
        original = velocity.to_numpy()[0].copy()
        accurate = AccuratePlanarSums(induction)
        induction._evaluate = accurate.evaluate
        induction._add_images = accurate.add_images
        induction.evaluate_targets(**values)
        actual, jacobian = velocity.to_numpy()[0].copy(), gradient.to_numpy()[0].copy()
        # The paired sources cancel exactly, leaving the final 0.001 m²/s vortex.
        position[0] = [1, 0.3, 0]
        strength[0] = [0, 0, 0.001]
        induction.evaluate_targets(**{**values, "source_count": 1})
        expected = velocity.to_numpy()[0]
        expected_gradient = gradient.to_numpy()[0]
        np.testing.assert_allclose(actual, expected, atol=5e-7)
        np.testing.assert_allclose(jacobian, expected_gradient, atol=5e-7)
        assert np.linalg.norm(actual - expected) < 0.02 * np.linalg.norm(original - expected)
    finally:
        ti.reset()
