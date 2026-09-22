"""Bounded numerical exclusion at a verified cylindrical FVM wall."""

import numpy as np
import pytest

from source.coupler.solver import _project_inside_verified_cylinder


@pytest.mark.parametrize("centre", [(0.0, 0.0), (0.17, -0.23)])
def test_shallow_crossing_projects_to_fluid_and_preserves_axial_location(centre):
    cx, cy = centre
    positions = np.array(
        [
            [cx + 0.49, cy, 0.1],
            [cx + 0.6, cy, -0.1],
        ],
        dtype=np.float32,
    )
    invalid = np.array([True, False])
    corrected, delta, maximum = _project_inside_verified_cylinder(
        positions, invalid, (cx, cy, 0.5, 0.001), 0.1, 2e-6
    )
    assert maximum == pytest.approx(0.01, abs=1e-6)
    assert np.linalg.norm(corrected[0, :2] - [cx, cy]) > 0.5
    assert corrected[0, 2] == positions[0, 2]
    np.testing.assert_array_equal(corrected[1], positions[1])
    np.testing.assert_allclose(corrected[0] - positions[0], delta[0], atol=1e-7)


def test_deep_crossing_is_rejected():
    positions = np.array([[0.4, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="too deeply"):
        _project_inside_verified_cylinder(
            positions, np.array([True]), (0.0, 0.0, 0.5, 0.001), 0.1, 2e-6
        )
