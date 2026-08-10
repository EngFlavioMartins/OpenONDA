from types import SimpleNamespace

import numpy as np

from source.coupler.core.helpers.continuous_overlap import (
    ContinuousOverlapInjector,
    continuous_handoff,
    cosine_eta,
    max_stable_dt,
    required_buffer_length,
)
from source.solvers.FVM.immersed_boundary import ImmersedBody

BOX = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
H = 0.25


def _zero_target(points):
    return np.zeros((len(points), 3))


def test_cosine_authority_partition():
    points = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.5, 0.0, 0.0]])
    eta = cosine_eta(points, BOX, ramp_width=0.3, dead_zone=0.05)
    np.testing.assert_allclose(eta[[0, 2]], [1.0, 0.0])
    assert 0.0 < eta[1] < 1.0


def test_buffer_dt_inverse():
    length = required_buffer_length(1.2, 0.05, H)
    assert np.isclose(max_stable_dt(1.2, length, H), 0.05)


def test_aligned_handoff_excludes_solid():
    def target(points):
        return np.tile([0.0, 0.0, 1.0e-3], (len(points), 1))

    def solid(points):
        return np.all(np.abs(points) < 0.2, axis=1)

    result = continuous_handoff(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        BOX,
        H,
        circulation_at_node=target,
        inside_mesh_at_node=lambda points: ~solid(points),
        excluded_at_node=solid,
        ramp_width=0.3,
        dead_zone=0.05,
        threshold_abs=1.0e-12,
        lattice_anchor=np.array([-0.375, -0.375, -0.375]),
    )

    assert result.n_total > 0
    assert not solid(result.pos).any()
    assert np.isfinite(result.circ).all()


def test_injector_reuses_native_ibm_solid_geometry():
    config = SimpleNamespace(
        h=0.1,
        nu=0.01,
        buffer_thickness=0.3,
        dead_zone_h=1.0,
        overlap_radius_ratio=1.0,
        u_inf=[1.0, 0.0, 0.0],
        prune_vorticity_min=0.01,
        fvm_box=(-1.0, 1.0, -1.0, 1.0, -0.1, 0.1),
        wall_patch_name=None,
    )
    body = ImmersedBody.cylinder_z([0.0, 0.0, 0.0], diameter=1.0, h=0.1)
    fvm = SimpleNamespace(
        ibm=SimpleNamespace(bodies=[body]),
        get_cell_center_coordinates=lambda: np.array([[-0.95, -0.95, -0.05], [0.95, 0.95, 0.05]]),
    )
    coupler = SimpleNamespace(config=config, dt_vpm=0.1, vpm=None)
    injector = ContinuousOverlapInjector(coupler)
    injector.setup(fvm)

    assert injector._solid_bodies == (body,)
    np.testing.assert_array_equal(
        injector._points_in_solid(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]],
            include_boundary=False,
        ),
        [True, False, False],
    )


def test_free_wake_is_retained():
    pos = np.array([[1.0, 0.0, 0.0], [1.2, 0.1, 0.0]])
    circ = np.array([[0.0, 0.0, 0.2], [0.0, 0.1, 0.0]])
    result = continuous_handoff(
        pos,
        circ,
        BOX,
        H,
        circulation_at_node=_zero_target,
        buffer_length=0.1,
        threshold_abs=1.0e-12,
    )

    np.testing.assert_allclose(result.pos, pos)
    np.testing.assert_allclose(result.circ, circ)


def test_population_cap_preserves_integral_circulation():
    rng = np.random.default_rng(3)
    pos = rng.uniform([1.0, -0.5, -0.5], [2.0, 0.5, 0.5], (8, 3))
    circ = rng.normal(size=(8, 3)) * 0.05
    result = continuous_handoff(
        pos,
        circ,
        BOX,
        H,
        circulation_at_node=_zero_target,
        max_output_particles=4,
        threshold_abs=1.0e-12,
    )

    assert result.n_total == 4
    assert result.n_population_pruned == 4
    np.testing.assert_allclose(result.circ.sum(axis=0), circ.sum(axis=0), atol=1.0e-12)
    before = 0.5 * np.cross(pos, circ).sum(axis=0)
    after = 0.5 * np.cross(result.pos, result.circ).sum(axis=0)
    np.testing.assert_allclose(after, before, atol=1.0e-12)
