"""Wall handling depends on the surface, never on a tutorial or fitted shape."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from source.coupler.geometry import SolidBoundary
from source.coupler.interpolation import FVMVelocityInterpolator
from source.coupler.solid import SolidParticleGuard
from source.solvers.fvm.immersed_boundary.body import ImmersedBody
from tests.coupler._solid_geometry import wall_case


@pytest.mark.parametrize("shape", ["rotated", "concave", "thin", "multiple", "curved"])
def test_generic_projection_follows_first_wall_and_leaves_strength_unchanged(shape):
    boundary, start, end, normal = wall_case(shape)
    assert not boundary.contains(start[None])[0]
    assert boundary.blocks_segments(start[None], end[None])[0]
    if shape == "thin":
        assert not boundary.contains(end[None])[0]  # Endpoint-only checks miss tunnelling.
    positions = end[None].astype(np.float32)
    corrected, selected, displacement, maximum = boundary.constrain_motion(
        positions,
        0.25,
        starts=start[None],
    )
    assert selected.tolist() == [True]
    assert not boundary.contains(corrected).any()
    assert not boundary.blocks_segments(start[None], corrected).any()
    assert 0 < maximum < 0.25 * 0.25
    assert np.dot(displacement[0], normal) > 0
    np.testing.assert_allclose(np.cross(displacement[0], normal), 0, atol=2e-7)


def test_concave_void_is_fluid_and_distinct_from_the_bounding_box():
    boundary, *_ = wall_case("concave")
    points = np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]])
    np.testing.assert_array_equal(boundary.contains(points), [True, True, True, False])
    assert not boundary.blocks_segments([[1.2, 1.2, 0]], [[1.8, 1.8, 0]])[0]


def test_deep_crossings_require_a_smaller_time_step():
    boundary, *_ = wall_case("thin")
    with pytest.raises(RuntimeError, match="Reduce the time step"):
        boundary.constrain_motion(np.array([[-0.2, 0, 0]]), 0.1, starts=np.array([[0.2, 0, 0]]))


@pytest.mark.parametrize("shape", ["rotated", "concave", "thin", "multiple", "curved"])
def test_stage_and_accepted_motion_share_geometry_and_report_impulse(shape):
    boundary, start, end, _ = wall_case(shape)
    physics = SimpleNamespace(
        _download_vector_field=lambda field, count: field[:count].copy(),
        _upload_vector_array=lambda values, field, count: np.copyto(field[:count], values),
    )
    guard = SolidParticleGuard(boundary, physics, 0.25, logging.getLogger(__name__))
    strength = np.array([[0.2, -0.1, 0.5]], dtype=np.float32)
    original_strength = strength.copy()
    state = SimpleNamespace(
        position=start[None].astype(np.float32), vortex_strength=strength, count=1, stage_index=0
    )
    guard.stage(state)
    # A diagnostic evaluation must not replace the physical RK path origin.
    state.stage_index = None
    guard.stage(state)
    state.position[:] = end
    state.stage_index = 1
    guard.stage(state)
    assert not boundary.contains(state.position).any()
    state.position[:] = end
    guard.accepted(state.position, strength, 1)
    np.testing.assert_array_equal(strength, original_strength)
    np.testing.assert_allclose(
        physics.last_solid_projection["accepted_impulse_change"],
        np.cross(state.position.astype(float) - end.astype(np.float32), strength).sum(axis=0),
        atol=1e-9,
    )
    assert physics.last_solid_projection["stage_count"] == 1
    assert physics.last_solid_projection["accepted_count"] == 1


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "sphere", "centre": [0, 0, 0], "radius": 0.5},
        {"type": "cylinder_z", "centre": [0, 0, 0], "radius": 0.5, "z_bounds": [-0.5, 0.5]},
        {"type": "rectangle_z", "centre": [0, 0, 0], "size": [1, 1], "z_bounds": [-0.5, 0.5]},
        {
            "type": "polygon_z",
            "vertex_position": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0], [0, 0], [0, 0.5], [-0.5, 0.5]],
            "z_bounds": [-0.5, 0.5],
        },
    ],
)
def test_immersed_geometry_segments_handle_entries_tangency_and_extrusion(geometry):
    body = ImmersedBody("geometry", [[0, 0, 0]], geometry=geometry)
    boundary = SolidBoundary((body,))
    starts = np.array([[-0.6, -0.1, 0], [-0.6, -0.5, 0], [-0.6, -0.1, 1], [-0.6, -0.6, 0]])
    ends = starts.copy()
    ends[:, 0] = 0.6
    np.testing.assert_array_equal(
        boundary.blocks_segments(starts, ends), [True, False, False, False]
    )
    fraction, normals = boundary.first_intersections(starts[:1], ends[:1])
    assert 0 < fraction[0] < 0.5
    assert normals[0, 0] < 0


def test_velocity_trace_uses_visible_donors_without_mixing_opposite_wall_sides():
    boundary, *_ = wall_case("thin")
    left = np.array([[-0.2, y, z] for y in (-0.1, 0.1) for z in (-0.1, 0.1)])
    right = np.array([[0.02, y, z] for y in (-0.02, 0.02) for z in (-0.02, 0.02)])
    donors = np.vstack((left, right))
    trace = FVMVelocityInterpolator(donors, cKDTree(donors), solid_boundary=boundary)
    targets = np.array([[-0.015, 0, 0], [0.015, 0, 0]])
    # The geometrically nearest four donors to the left target are all
    # across the thin wall; the search must expand to find visible support.
    velocity = np.zeros_like(donors)
    velocity[:4, 0], velocity[4:, 0] = 1, 100
    sampled = trace.sample(targets, velocity, np.zeros((8, 3, 3)))
    np.testing.assert_allclose(sampled[:, 0], [1, 100], atol=1e-13)
    gradient = np.broadcast_to(np.eye(3), (8, 3, 3))
    np.testing.assert_allclose(trace.sample(targets, donors, gradient), targets, atol=1e-13)


def test_unbounded_immersed_extrusion_does_not_invent_a_finite_query_box():
    body = ImmersedBody.cylinder_z([0, 0, 0], diameter=1, grid_spacing=0.1)
    boundary = SolidBoundary((body,))
    assert boundary.bounds is None
    assert boundary.blocks_segments([[-1, 0, 100]], [[1, 0, 100]])[0]
