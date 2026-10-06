"""Independent physical checks for the frozen slip-wall image experiment."""

import subprocess
import sys
import textwrap

import numpy as np

from tests.support.cylinder.audit_slip_channel_induction import channel_image_velocity_gradient


def point_vortex_velocity(points, sources, circulation):
    delta = points[:, None, :2] - sources[None, :, :2]
    coefficient = circulation[None, :] / (2 * np.pi * np.sum(delta**2, axis=2))
    return np.column_stack(
        (
            -np.sum(coefficient * delta[:, :, 1], axis=1),
            np.sum(coefficient * delta[:, :, 0], axis=1),
            np.zeros(len(points)),
        )
    )


def test_images_cancel_wall_normal_velocity_without_adding_vorticity():
    half_width = 10.0
    sources = np.array([[-1.0, 0.4, 0.0], [4.0, -0.7, 0.0]])
    circulation = np.array([1.3, -1.0])
    x = np.linspace(-20.0, 30.0, 101)
    points = np.column_stack(
        (np.tile(x, 2), np.repeat([-half_width, half_width], len(x)), np.zeros(2 * len(x)))
    )
    correction, gradient = channel_image_velocity_gradient(points, sources, circulation, half_width)
    full = correction + point_vortex_velocity(points, sources, circulation)
    np.testing.assert_allclose(full[:, 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(gradient[:, 0, 1] - gradient[:, 1, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(gradient[:, 0, 0] + gradient[:, 1, 1], 0.0, atol=1e-14)


def test_harmonic_jacobian_and_free_space_limit_at_vortex_centres():
    sources = np.array([[-1.0, 0.4, 0.0], [4.0, -0.7, 0.0]])
    circulation = np.array([1.3, -1.0])
    points = np.vstack((sources, [[0.2, 0.1, 0.0], [3.0, 1.0, 0.0]]))
    correction, gradient = channel_image_velocity_gradient(points, sources, circulation, 10.0)
    for axis in range(2):
        offset = np.zeros(3)
        offset[axis] = 1e-5
        upper, _ = channel_image_velocity_gradient(points + offset, sources, circulation, 10.0)
        lower, _ = channel_image_velocity_gradient(points - offset, sources, circulation, 10.0)
        np.testing.assert_allclose((upper - lower) / (2e-5), gradient[:, :, axis], atol=1e-11)
    remote, _ = channel_image_velocity_gradient(points, sources, circulation, 1e5)
    assert np.max(np.abs(remote)) < 1e-10
    assert np.max(np.abs(correction)) > 1e-4


def check_device_images(device):
    code = textwrap.dedent("""
        from types import SimpleNamespace
        import numpy as np
        import taichi as ti
        from tests.support.cylinder.audit_slip_channel_induction import channel_image_velocity_gradient
        from tests.support.cylinder.slip_channel_control import SlipChannelImages

        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=1)
        position = np.array([[-1., .4, 0.], [4., -.7, 0.]], dtype=np.float32)
        points = np.vstack((position, [[.2, .1, 0.], [3., 1., 0.], [70., -5., 0.]]))
        circulation = np.array([1.3, -1.0], dtype=np.float32)
        target = ti.Vector.field(3, ti.f64, shape=8)
        source = ti.Vector.field(3, ti.f32, shape=2)
        strength = ti.Vector.field(3, ti.f32, shape=2)
        velocity = ti.Vector.field(3, ti.f64, shape=8)
        gradient = ti.Matrix.field(3, 3, ti.f64, shape=8)
        source.from_numpy(position)
        strength.from_numpy(np.column_stack((np.zeros(2), np.zeros(2), circulation)).astype(np.float32))
        target.from_numpy(np.vstack((points, np.zeros((3, 3)))))
        velocity.from_numpy(np.full((8, 3), 2.))
        gradient.from_numpy(np.full((8, 3, 3), 3.))
        model = SlipChannelImages(SimpleNamespace(_dtype=ti.f64, planar_span=1.), 10.)
        model.add_images(target, source, strength, velocity, gradient, 5, 2, True, True)
        expected_u, expected_j = channel_image_velocity_gradient(points, position.astype(float), circulation.astype(float), 10.)
        np.testing.assert_allclose(velocity.to_numpy()[:5] - 2., expected_u, atol=1e-13)
        np.testing.assert_allclose(gradient.to_numpy()[:5] - 3., expected_j, atol=1e-13)
        np.testing.assert_array_equal(velocity.to_numpy()[5:], np.full((3, 3), 2.))
        np.testing.assert_array_equal(gradient.to_numpy()[5:], np.full((3, 3, 3), 3.))
    """)
    if device not in ("cpu", "cuda"):
        raise ValueError("Device qualification must select CPU or CUDA")
    code = code.replace("arch=ti.cpu", "arch=ti." + device)
    code += f"\nassert ti.cfg.arch == ti.{device}\n"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_device_images_match_independent_complex_sum_and_preserve_capacity_tail():
    check_device_images("cpu")
