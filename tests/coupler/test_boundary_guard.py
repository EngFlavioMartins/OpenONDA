"""Guards on the outer FVM boundary of the coupling.

The 03_naca4412 case ran for 3.5 s after its boundary closure went unstable,
logging a tangential FVM-particle mismatch of 6.8 times the freestream speed.
The coupler now stops such a run, names the face, and warns at start-up when
the outer patch cells are not coupling-grade.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from openonda.coupler.boundary import (
    BoundaryMismatchError,
    _check_boundary_mismatch,
    inflow_transition_speed,
    report_coupling_patch_quality,
)
from openonda.coupler.config import CouplerConfig


def _coupler(**config):
    return SimpleNamespace(
        config=CouplerConfig(**config),
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        time=1.25,
        _is_root=True,
        _comm=None,
    )


def _faces(mismatch):
    centre = np.array([[-1.6, 0.16, 3.6], [2.0, 0.0, 0.0]])
    normal = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    difference = np.array([[0.0, -mismatch, 0.0], [0.01, 0.0, 0.0]])
    return centre, normal, difference, np.linalg.norm(difference, axis=1)


def test_an_unphysical_boundary_mismatch_stops_the_run_with_the_face_location():
    coupler = _coupler()
    centre, normal, difference, magnitude = _faces(1.9)
    message = _check_boundary_mismatch(coupler, centre, normal, difference, magnitude, 1.0)
    assert message is not None
    assert "xmin side" in message and "-1.6" in message and "1.9" in message
    assert issubclass(BoundaryMismatchError, RuntimeError)


def test_a_moderate_mismatch_warns_once_per_doubling(caplog):
    coupler = _coupler()
    with caplog.at_level(logging.WARNING, logger="openonda.coupler"):
        for value in (0.1, 0.25, 0.3, 0.45, 0.9):
            centre, normal, difference, magnitude = _faces(value)
            assert (
                _check_boundary_mismatch(coupler, centre, normal, difference, magnitude, 1.0)
                is None
            )
    warnings = [r for r in caplog.records if "boundary mismatch" in r.getMessage().lower()]
    # 0.25 and 0.3 share the first doubling above 0.2; 0.45 and 0.9 start new ones.
    assert len(warnings) == 3


def test_the_guard_can_be_disabled_and_is_not_part_of_the_restart_configuration():
    coupler = _coupler(boundary_mismatch_warning=None, boundary_mismatch_limit=None)
    centre, normal, difference, magnitude = _faces(50.0)
    assert _check_boundary_mismatch(coupler, centre, normal, difference, magnitude, 1.0) is None
    values = CouplerConfig().to_dict()["coupler"]
    assert "boundary_mismatch_limit" not in values and "boundary_mismatch_warning" not in values
    with pytest.raises(ValueError):
        CouplerConfig(boundary_mismatch_warning=2.0, boundary_mismatch_limit=1.0)
    with pytest.raises(ValueError):
        CouplerConfig(boundary_mismatch_limit=0.0)


def test_the_inflow_transition_speed_scales_with_the_flow():
    coupler = _coupler()
    velocity = np.tile([1.0, 0.0, 0.0], (8, 1))
    assert inflow_transition_speed(coupler, velocity) == pytest.approx(0.1)
    # Without a freestream (a body moving in still fluid) the boundary speed sets the scale.
    coupler.freestream_velocity = np.zeros(3)
    assert inflow_transition_speed(coupler, 0.5 * velocity) == pytest.approx(0.05)
    assert inflow_transition_speed(coupler, np.zeros((8, 3))) > 0.0


def test_skewed_outer_patch_cells_are_reported(caplog):
    coupler = _coupler()
    normal = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # The 03_naca4412 corner cells: normal distance 0.0106 m, tangential offset 0.0038 m.
    skewed = np.array([[-0.0106, 0.0, 0.0038], [0.0, 0.0, 0.02]])
    with caplog.at_level(logging.WARNING, logger="openonda.coupler"):
        quality = report_coupling_patch_quality(coupler, normal, skewed)
    assert quality["max_tangential_offset_ratio"] == pytest.approx(0.0038 / 0.0106)
    assert any("not coupling-grade" in r.getMessage().lower() for r in caplog.records)
    caplog.clear()
    exact = np.array([[-0.02, 0.0, 0.0], [0.0, 0.0, 0.02]])
    with caplog.at_level(logging.WARNING, logger="openonda.coupler"):
        quality = report_coupling_patch_quality(coupler, normal, exact)
    assert quality["max_tangential_offset_ratio"] == 0.0
    assert not caplog.records
