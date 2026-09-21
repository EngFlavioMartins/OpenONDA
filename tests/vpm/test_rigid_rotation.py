"""Shared rigid-body rotation agrees with an independent rotation-vector map."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from source.solvers.vpm.boundary_elements.vlm.coupling.kinematics import CompositeVLM, ManeuverVLM


@pytest.mark.parametrize("omega", [(0, 0, 0), (1, 0, 0), (0.3, -0.7, 1.2)])
def test_maneuver_and_composite_rotation(omega):
    angular_velocity = np.asarray(omega, dtype=float)
    motion = ManeuverVLM(angular_velocity_function=lambda time: angular_velocity)
    composite = CompositeVLM([motion])
    expected = Rotation.from_rotvec(0.2 * angular_velocity).as_matrix()
    for kinematics in (motion, composite):
        actual = kinematics._rotation_matrix(kinematics.get_angular_velocity(0), 0.2)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-16)
        np.testing.assert_allclose(actual.T @ actual, np.eye(3), rtol=0, atol=5e-16)
