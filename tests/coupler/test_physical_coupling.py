"""Curl convention, remeshing extension and fixed sub-cycling."""

from types import SimpleNamespace

import numpy as np
import pytest

from openonda.coupler.boundary import advance_fvm_substeps
from openonda.coupler.vorticity_transfer import VorticityTransfer, remeshing_extension_length


def test_remeshing_extension_covers_one_interval_of_travel_plus_complete_m4_support():
    h = 0.03125
    coupling_time_step_size = 0.01

    extension = remeshing_extension_length(
        [1.0, 0.0, 0.0],
        coupling_time_step_size,
        h,
    )

    assert extension == pytest.approx(1.5 * coupling_time_step_size + 2.0 * h)
    assert extension == pytest.approx(0.0775)
    assert remeshing_extension_length([0.0, 0.0, 0.0], coupling_time_step_size, h) == pytest.approx(
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


def test_mixed_boundary_substeps_use_both_interval_ends(monkeypatch):
    coupler = SimpleNamespace(
        n_fvm_substeps=4,
        fvm_time_step_size=0.005,
        fvm_solver=SimpleNamespace(step=4),
        coupling_time=lambda step: step * 0.02,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        config=SimpleNamespace(),
    )
    recorded = []
    times = []

    def record_step(
        _coupler,
        _patch,
        velocity,
        normal_velocity=None,
        normal_gradient=None,
        *,
        accepted_time,
    ):
        recorded.append((velocity.copy(), normal_velocity.copy(), normal_gradient.copy()))
        times.append(accepted_time)

    monkeypatch.setattr("openonda.coupler.boundary.apply_fvm_boundary", record_step)
    face_centre = np.zeros((3, 3))
    face_normal = np.tile([1.0, 0.0, 0.0], (3, 1))
    face_area = np.ones(3)
    previous_velocity = np.tile([1.0, 0.0, 0.0], (3, 1))
    next_velocity = np.tile([1.0, 0.4, 0.0], (3, 1))
    previous_normal_velocity = np.full(3, 0.2)
    next_normal_velocity = np.full(3, 0.6)
    previous_normal_gradient = np.zeros((3, 3))
    next_normal_gradient = np.full((3, 3), 0.8)

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
        previous_normal_gradient=previous_normal_gradient,
        next_normal_gradient=next_normal_gradient,
    )

    for values, alpha in zip(recorded, (0.25, 0.5, 0.75, 1.0), strict=True):
        np.testing.assert_allclose(
            values[0], (1.0 - alpha) * previous_velocity + alpha * next_velocity
        )
        np.testing.assert_allclose(
            values[1],
            (1.0 - alpha) * previous_normal_velocity + alpha * next_normal_velocity,
        )
        np.testing.assert_allclose(values[2], alpha * next_normal_gradient)
    # Substeps follow the coupling clock and the last one lands exactly on the interval end.
    np.testing.assert_allclose(times[:3], [0.025, 0.03, 0.035], rtol=0.0, atol=1e-15)
    assert times[-1] == 2 * 0.02
