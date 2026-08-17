"""Signed distance for immersed bodies, used by the coupled hand-off taper.

The coupler multiplies the transferred circulation by a C1 wall taper built from
this function.  A binary inside/outside mask cannot be used there: it is a step
function of position, and multiplying a band-limited field by a step injects
energy at wavelengths the particle lattice cannot represent.
"""

import numpy as np
import pytest

from source.solvers.FVM.immersed_boundary import ImmersedBody


def test_sphere_signed_distance_is_exact():
    body = ImmersedBody.sphere([0.2, -0.1, 0.3], diameter=1.0, h=0.1)
    assert body.has_solid_geometry
    centre = np.array([0.2, -0.1, 0.3])
    directions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0] / np.sqrt(3)]
    )
    for radius in (0.1, 0.5, 0.9, 1.7):
        points = centre + radius * directions
        np.testing.assert_allclose(body.signed_distance(points), radius - 0.5, atol=1e-12)


def test_sphere_contains_matches_the_sign_of_the_distance():
    body = ImmersedBody.sphere([0.0, 0.0, 0.0], diameter=1.0, h=0.1)
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.5, 1.5, (500, 3))
    distance = body.signed_distance(points)
    np.testing.assert_array_equal(body.contains(points, include_boundary=False), distance < 0.0)


def test_cylinder_signed_distance_is_exact_in_the_plane():
    body = ImmersedBody.cylinder_z([0.0, 0.0, 0.0], diameter=2.0, h=0.1)
    points = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, -4.0, 2.0]])
    np.testing.assert_allclose(body.signed_distance(points), [-1.0, 0.0, 2.0, 3.0], atol=1e-12)


def test_extruded_cylinder_distance_accounts_for_the_end_caps():
    body = ImmersedBody.extruded_cylinder_z(
        [0.0, 0.0, 0.0], diameter=2.0, z_bounds=[-1.0, 1.0], h=0.2, caps=True
    )
    # Directly off the curved side, directly off an end, and off a corner.
    points = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 4.0], [4.0, 0.0, 4.0]])
    expected = [2.0, 3.0, np.hypot(3.0, 3.0)]
    np.testing.assert_allclose(body.signed_distance(points), expected, atol=1e-12)


def test_uncapped_extrusion_is_infinite_in_z():
    """``caps=False`` means a cylinder passing through the domain, not a puck."""
    body = ImmersedBody.extruded_cylinder_z(
        [0.0, 0.0, 0.0], diameter=2.0, z_bounds=[-1.0, 1.0], h=0.2, caps=False
    )
    points = np.array([[0.0, 0.0, 4.0], [3.0, 0.0, 40.0]])
    np.testing.assert_allclose(body.signed_distance(points), [-1.0, 2.0], atol=1e-12)


def test_signed_distance_is_continuous_across_the_wall():
    """No jump at the surface; that is the whole point of using it for a taper."""
    body = ImmersedBody.sphere([0.0, 0.0, 0.0], diameter=1.0, h=0.05)
    radii = np.linspace(0.3, 0.7, 4001)
    points = np.zeros((len(radii), 3))
    points[:, 0] = radii
    distance = body.signed_distance(points)
    assert np.max(np.abs(np.diff(distance))) < 2.0 * (radii[1] - radii[0])
    assert np.isclose(np.interp(0.0, distance, radii), 0.5, atol=1e-9)


def test_body_without_geometry_metadata_refuses_to_guess():
    body = ImmersedBody.from_points(np.eye(4, 3), name="blob")
    with pytest.raises(ValueError, match="no solid geometry metadata"):
        body.signed_distance(np.zeros((1, 3)))
