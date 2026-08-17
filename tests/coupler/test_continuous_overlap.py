from types import SimpleNamespace

import numpy as np

from source.coupler.core.helpers.continuous_overlap import (
    ContinuousOverlapInjector,
    continuous_handoff,
    cosine_eta,
    max_stable_dt,
    redistribute_locally,
    required_buffer_length,
    smoothstep,
    soft_prune,
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
        return np.all(np.abs(np.asarray(points)) < 0.2, axis=1)

    def fluid_weight(points):
        # C1 taper: 0 inside the solid, 1 one cell outside it.
        points = np.asarray(points)
        depth = 0.2 - np.max(np.abs(points), axis=1)
        return smoothstep(-depth, 0.0, H)

    result = continuous_handoff(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        BOX,
        H,
        circulation_at_node=target,
        mesh_weight_at_node=lambda points: (
            1.0 - smoothstep(np.max(np.abs(np.asarray(points)), axis=1), 0.6, 0.8)
        ),
        fluid_weight_at_node=fluid_weight,
        interior_at_node=solid,
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


def test_injector_uses_separate_handoff_box():
    config = SimpleNamespace(
        h=0.1,
        nu=0.01,
        buffer_thickness=0.3,
        dead_zone_h=1.0,
        overlap_radius_ratio=1.0,
        u_inf=[1.0, 0.0, 0.0],
        prune_vorticity_min=0.01,
        fvm_box=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        handoff_box=(-0.7, 0.7, -0.7, 0.7, -0.7, 0.7),
        wall_patch_name=None,
    )
    fvm = SimpleNamespace(
        ibm=SimpleNamespace(bodies=[]),
        get_cell_center_coordinates=lambda: np.array([[-0.9, 0.0, 0.0], [0.9, 0.0, 0.0]]),
    )
    injector = ContinuousOverlapInjector(SimpleNamespace(config=config, dt_vpm=0.1, vpm=None))

    injector.setup(fvm)

    np.testing.assert_array_equal(injector._box, config.handoff_box)


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


def test_soft_prune_is_continuous_and_barely_biases_strong_nodes():
    """A hard clip is a step function of position; the garrote is not."""
    threshold = 1.0e-3
    magnitudes = np.linspace(0.0, 5.0 * threshold, 4001)
    circ = np.zeros((len(magnitudes), 3))
    circ[:, 2] = magnitudes
    shrunk, removed = soft_prune(circ, threshold)

    kept = shrunk[:, 2]
    # Continuity: no jump larger than the sampling step anywhere, in particular
    # across the threshold, where a hard clip would jump by the full threshold.
    assert np.max(np.abs(np.diff(kept))) < 2.0 * (magnitudes[1] - magnitudes[0])
    # Exactly zero below the threshold, so the node count is still bounded.
    assert np.all(kept[magnitudes <= threshold] == 0.0)
    # Negligible bias on strong nodes: (threshold / |Gamma|)^2.
    strong = magnitudes >= 5.0 * threshold
    np.testing.assert_allclose(kept[strong], magnitudes[strong], rtol=0.05)
    np.testing.assert_allclose(shrunk + removed, circ, atol=1e-18)


def test_hard_clip_would_be_discontinuous():
    """Contrast case documenting what the garrote replaces."""
    threshold = 1.0e-3
    magnitudes = np.linspace(0.0, 5.0 * threshold, 4001)
    clipped = np.where(magnitudes >= threshold, magnitudes, 0.0)
    assert np.max(np.abs(np.diff(clipped))) > 0.9 * threshold


def test_local_redistribution_conserves_circulation_and_impulse_locally():
    shape = (7, 7, 7)
    rng = np.random.default_rng(2)
    field = rng.normal(size=(*shape, 3)) * 1e-3
    # The node at the centre is pruned entirely: shrunk + removed == field.
    removed = np.zeros_like(field)
    removed[3, 3, 3] = field[3, 3, 3]
    shrunk = field.copy()
    shrunk[3, 3, 3] = 0.0

    out = redistribute_locally(removed, shrunk.reshape(-1, 3), shape).reshape(*shape, 3)

    # Total circulation preserved.
    np.testing.assert_allclose(
        out.reshape(-1, 3).sum(axis=0), field.reshape(-1, 3).sum(axis=0), atol=1e-15
    )
    # And it stayed local: only the six face neighbours changed.
    changed = np.linalg.norm(out - shrunk, axis=-1) > 1e-18
    assert changed.sum() == 6
    for index in np.argwhere(changed):
        assert np.abs(index - np.array([3, 3, 3])).sum() == 1


def test_local_redistribution_preserves_linear_impulse():
    shape = (5, 5, 5)
    h = 0.2
    coords = np.stack(
        np.meshgrid(*[np.arange(n) * h for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    field = np.zeros((*shape, 3))
    field[2, 2, 2] = [0.0, 0.0, 1.0e-4]
    removed = field.copy()
    shrunk = np.zeros_like(field)
    # Give the neighbours something to survive on so they are donation targets.
    for axis in range(3):
        for step in (-1, +1):
            index = [2, 2, 2]
            index[axis] += step
            shrunk[tuple(index)] = [0.0, 0.0, 1.0e-8]

    out = redistribute_locally(removed, shrunk.reshape(-1, 3), shape).reshape(-1, 3)
    before = 0.5 * np.cross(coords, (shrunk + field).reshape(-1, 3)).sum(axis=0)
    after = 0.5 * np.cross(coords, out).sum(axis=0)
    np.testing.assert_allclose(after, before, atol=1e-18)


def test_local_redistribution_does_not_resurrect_an_all_weak_region():
    shape = (5, 5, 5)
    removed = np.ones((*shape, 3)) * 1.0e-8
    redistributed = redistribute_locally(removed, np.zeros_like(removed), shape)
    assert not np.any(redistributed)


def test_smoothstep_is_c1_and_bounded():
    x = np.linspace(-1.0, 2.0, 20001)
    y = smoothstep(x, 0.0, 1.0)
    assert y.min() == 0.0 and y.max() == 1.0
    derivative = np.gradient(y, x)
    # Zero slope at both ends is what keeps the taper from adding grid-scale
    # content to whatever it multiplies.
    assert abs(derivative[0]) < 1e-6
    assert abs(derivative[-1]) < 1e-6
    assert np.max(np.abs(np.diff(derivative))) < 1e-2


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
