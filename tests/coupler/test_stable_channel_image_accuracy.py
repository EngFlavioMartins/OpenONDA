"""Qualify image arithmetic at source spacings used in the cylinder case."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from openonda import vpm
from tests.support.cylinder.audit_slip_channel_induction import channel_image_velocity_gradient
from tests.support.cylinder.stable_channel_image_control import StableChannelImages


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_f32_images_at_small_separations_match_independent_complex_formula(device):
    if device == "cuda" and os.environ.get("OPENONDA_TEST_CUDA") != "1":
        pytest.skip("Set OPENONDA_TEST_CUDA=1 for actual CUDA qualification")
    ti.reset()
    ti.init(arch=getattr(ti, device), default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=1)
    assert ti.cfg.arch == getattr(ti, device)
    try:
        induction = vpm.PlanarChannelInduction(half_width=10).bind(
            SimpleNamespace(accumulator_dtype=ti.f32, max_n_particles=1)
        )
        images = StableChannelImages(induction, 10)
        points = np.column_stack(
            (np.array([0, 0.01, 0.02, 0.04, 0.1, 0.5, 1, 4, 6, 8, 40]), np.zeros(11), np.zeros(11))
        ).astype(np.float32)
        position = ti.Vector.field(3, ti.f32, shape=1)
        strength = ti.Vector.field(3, ti.f32, shape=1)
        target = ti.Vector.field(3, ti.f32, shape=len(points))
        velocity = ti.Vector.field(3, ti.f32, shape=len(points))
        gradient = ti.Matrix.field(3, 3, ti.f32, shape=len(points))
        position[0], strength[0] = [0, 0, 0], [0, 0, 1]
        target.from_numpy(points)
        velocity.fill(0)
        gradient.fill(0)
        images.add_images(
            target, position, strength, velocity, gradient, len(points), 1, True, True
        )
        expected, expected_gradient = channel_image_velocity_gradient(
            points.astype(float), np.zeros((1, 3)), np.array([1.0]), 10
        )
        np.testing.assert_allclose(velocity.to_numpy(), expected, atol=1e-8)
        np.testing.assert_allclose(gradient.to_numpy(), expected_gradient, atol=5e-8)
        assert np.isfinite(velocity.to_numpy()).all()
        assert np.isfinite(gradient.to_numpy()).all()
    finally:
        ti.reset()
