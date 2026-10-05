"""Curl convention, release support and fixed subcycling."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.boundary import advance_fvm_substeps
from source.coupler.vorticity_transfer import VorticityTransfer, required_renewal_buffer_length


def test_renewal_buffer_covers_release_travel_plus_complete_m4_support():
    h = 0.03125
    coupling_time_step = 0.01

    buffer_length = required_renewal_buffer_length(
        [1.0, 0.0, 0.0],
        coupling_time_step,
        h,
    )

    assert buffer_length == pytest.approx(1.5 * coupling_time_step + 2.0 * h)
    assert buffer_length == pytest.approx(0.0775)
    assert required_renewal_buffer_length([0.0, 0.0, 0.0], coupling_time_step, h) == pytest.approx(
        2.0 * h
    )


def test_fvm_gradient_curl_matches_cell_vorticity_convention():
    gradient = np.zeros((2, 3, 3))
    gradient[0, 1, 2] = 3.0
    gradient[0, 2, 1] = -2.0
    gradient[1, 2, 0] = 4.0
    gradient[1, 0, 2] = 1.5
    np.testing.assert_array_equal(
        VorticityTransfer._vorticity_from_gradient(gradient),
        [[5.0, 0.0, 0.0], [0.0, 2.5, 0.0]],
    )


def test_vorticity_mixed_substeps_use_both_time_endpoints(monkeypatch):
    coupler = SimpleNamespace(
        n_fvm_substeps=4,
        fvm_time_step_size=0.005,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        setup=SimpleNamespace(),
    )
    recorded = []

    def record_step(
        _coupler,
        _patch,
        velocity,
        pressure_gradient=None,
        normal_velocity=None,
        tangential_gradient=None,
    ):
        recorded.append((velocity.copy(), normal_velocity.copy(), tangential_gradient.copy()))

    monkeypatch.setattr("source.coupler.boundary.apply_fvm_boundary", record_step)
    face_centre = np.zeros((3, 3))
    face_normal = np.tile([1.0, 0.0, 0.0], (3, 1))
    face_area = np.ones(3)
    previous_velocity = np.tile([1.0, 0.0, 0.0], (3, 1))
    next_velocity = np.tile([1.0, 0.4, 0.0], (3, 1))
    previous_normal_velocity = np.full(3, 0.2)
    next_normal_velocity = np.full(3, 0.6)
    previous_tangential_gradient = np.zeros((3, 3))
    next_tangential_gradient = np.full((3, 3), 0.8)

    advance_fvm_substeps(
        coupler,
        "numericalBoundary",
        face_centre,
        face_normal,
        face_area,
        previous_velocity,
        next_velocity,
        previous_normal_velocity=previous_normal_velocity,
        next_normal_velocity=next_normal_velocity,
        previous_tangential_gradient=previous_tangential_gradient,
        next_tangential_gradient=next_tangential_gradient,
    )

    for values, alpha in zip(recorded, (0.25, 0.5, 0.75, 1.0), strict=True):
        np.testing.assert_allclose(
            values[0], (1.0 - alpha) * previous_velocity + alpha * next_velocity
        )
        np.testing.assert_allclose(
            values[1],
            (1.0 - alpha) * previous_normal_velocity + alpha * next_normal_velocity,
        )
        np.testing.assert_allclose(values[2], alpha * next_tangential_gradient)
