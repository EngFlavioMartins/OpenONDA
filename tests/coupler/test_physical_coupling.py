"""Curl convention, remeshing extension and fixed sub-cycling."""

from types import SimpleNamespace

import numpy as np
import pytest

from openonda.coupler.boundary import advance_fvm_step
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


def test_fvm_step_uses_vpm_boundary_data_at_the_shared_end_time(monkeypatch):
    coupler = SimpleNamespace(
        time=0.035,
        fvm_time_step_size=0.005,
        _normal_velocity_boundary_condition=np.full(3, 0.6),
        _normal_gradient_boundary_condition=np.full((3, 3), 0.8),
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
    next_velocity = np.tile([1.0, 0.4, 0.0], (3, 1))

    advance_fvm_step(
        coupler,
        "numericalBoundary",
        next_velocity,
    )

    assert len(recorded) == 1
    np.testing.assert_array_equal(recorded[0][0], next_velocity)
    np.testing.assert_array_equal(recorded[0][1], coupler._normal_velocity_boundary_condition)
    np.testing.assert_array_equal(recorded[0][2], coupler._normal_gradient_boundary_condition)
    assert times == [0.04]
