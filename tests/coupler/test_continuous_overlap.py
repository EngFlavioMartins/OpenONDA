from types import SimpleNamespace

import numpy as np

import source.coupler.vorticity_transfer as transfer_module
from source.coupler.vorticity_transfer import (
    VorticityTransfer,
    build_transfer_lattice,
    continuous_transfer,
    cosine_eta,
    max_stable_time_step_size,
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
    eta = cosine_eta(points, BOX, authority_ramp_width=0.3, overlap_zone_dead_zone=0.05)
    np.testing.assert_allclose(eta[[0, 2]], [1.0, 0.0])
    assert 0.0 < eta[1] < 1.0


def test_buffer_dt_inverse():
    length = required_buffer_length(1.2, 0.05, H)
    assert np.isclose(max_stable_time_step_size(1.2, length, H), 0.05)


def test_static_transfer_lattice_preserves_the_dynamic_transfer():
    """Static mesh/solid masks must not change the active hand-off operator."""

    def mesh_weight(points):
        return 1.0 - smoothstep(np.max(np.abs(np.asarray(points)), axis=1), 0.6, 0.8)

    lattice = build_transfer_lattice(
        BOX,
        H,
        transfer_buffer_length=0.0,
        mesh_weight_at_node=mesh_weight,
        authority_ramp_width=0.3,
        overlap_zone_dead_zone=0.05,
        freestream_velocity=[1.0, 0.0, 0.0],
    )
    target = lambda points: np.tile([0.0, 0.0, 1.0e-3], (len(points), 1))  # noqa: E731
    dynamic = continuous_transfer(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        BOX,
        H,
        vortex_strength_at_node=target,
        mesh_weight_at_node=mesh_weight,
        authority_ramp_width=0.3,
        overlap_zone_dead_zone=0.05,
        transfer_prune_threshold_abs=1.0e-12,
    )
    cached = continuous_transfer(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        BOX,
        H,
        vortex_strength_at_node=target,
        authority_ramp_width=0.3,
        overlap_zone_dead_zone=0.05,
        transfer_prune_threshold_abs=1.0e-12,
        lattice=lattice,
    )

    np.testing.assert_allclose(cached.pos, dynamic.pos)
    np.testing.assert_allclose(cached.circ, dynamic.circ)
    assert cached.spectral_band_ratio == dynamic.spectral_band_ratio


def test_transfer_diagnostics_can_be_deferred_without_changing_particles():
    target = lambda points: np.tile([0.0, 0.0, 1.0e-3], (len(points), 1))  # noqa: E731
    full = continuous_transfer(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        BOX,
        H,
        vortex_strength_at_node=target,
        transfer_prune_threshold_abs=1.0e-12,
    )
    deferred = continuous_transfer(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        BOX,
        H,
        vortex_strength_at_node=target,
        transfer_prune_threshold_abs=1.0e-12,
        compute_diagnostics=False,
    )

    np.testing.assert_allclose(deferred.pos, full.pos)
    np.testing.assert_allclose(deferred.circ, full.circ)
    assert deferred.diagnostics_evaluated is False
    assert deferred.spectral_band_ratio == {}


def test_deferred_transfer_skips_final_gaussian_representation(monkeypatch):
    target = lambda points: np.tile([0.0, 0.0, 1.0e-3], (len(points), 1))  # noqa: E731
    gaussian = transfer_module._gaussian_mollified_vortex_strength
    calls = []

    def record_gaussian(*args, **kwargs):
        calls.append(1)
        return gaussian(*args, **kwargs)

    monkeypatch.setattr(transfer_module, "_gaussian_mollified_vortex_strength", record_gaussian)
    continuous_transfer(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        BOX,
        H,
        vortex_strength_at_node=target,
        transfer_prune_threshold_abs=1.0e-12,
        compute_diagnostics=False,
    )

    # One filter constructs the physical VPM field and the second supplies the
    # bounded residual correction.  The post-correction representation is audit-only.
    assert len(calls) == 2


def test_aligned_transfer_excludes_solid():
    def target(points):
        return np.tile([0.0, 0.0, 1.0e-3], (len(points), 1))

    def solid(points):
        return np.all(np.abs(np.asarray(points)) < 0.2, axis=1)

    def fluid_weight(points):
        # C1 taper: 0 inside the solid, 1 one cell outside it.
        points = np.asarray(points)
        depth = 0.2 - np.max(np.abs(points), axis=1)
        return smoothstep(-depth, 0.0, H)

    result = continuous_transfer(
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        BOX,
        H,
        vortex_strength_at_node=target,
        mesh_weight_at_node=lambda points: (
            1.0 - smoothstep(np.max(np.abs(np.asarray(points)), axis=1), 0.6, 0.8)
        ),
        fluid_weight_at_node=fluid_weight,
        interior_at_node=solid,
        authority_ramp_width=0.3,
        overlap_zone_dead_zone=0.05,
        transfer_prune_threshold_abs=1.0e-12,
        lattice_anchor=np.array([-0.375, -0.375, -0.375]),
    )

    assert result.n_total > 0
    assert not solid(result.pos).any()
    assert np.isfinite(result.circ).all()


def test_transfer_reuses_native_ibm_solid_geometry():
    config = SimpleNamespace(
        vpm_particle_spacing=0.1,
        authority_ramp_width=0.3,
        vpm_only_width=0.1,
        vpm_core_radius_ratio=1.0,
        freestream_velocity=[1.0, 0.0, 0.0],
        transfer_vorticity_cutoff=0.01,
        transfer_region_bounds=None,
        transfer_amplification_cap=2.0,
        transfer_boundary_prune_multiplier=1.0,
        transfer_diagnostic_interval_steps=1,
    )
    body = ImmersedBody.cylinder_z([0.0, 0.0, 0.0], diameter=1.0, h=0.1)
    fvm = SimpleNamespace(
        ibm=SimpleNamespace(bodies=[body]),
        setup=SimpleNamespace(boundaries=[]),
        get_cell_centre_coordinates=lambda: np.array([[-0.95, -0.95, -0.05], [0.95, 0.95, 0.05]]),
    )
    coupler = SimpleNamespace(
        setup=config,
        vpm_time_step_size=0.1,
        kinematic_viscosity=0.01,
        fvm_box=np.array([-1.0, 1.0, -1.0, 1.0, -0.1, 0.1]),
        vpm_solver=None,
    )
    transfer = VorticityTransfer(coupler)
    transfer.setup(fvm)

    assert transfer._solid_bodies == (body,)
    np.testing.assert_array_equal(
        transfer._points_in_solid(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]],
            include_boundary=False,
        ),
        [True, False, False],
    )


def test_transfer_uses_separate_transfer_region_box():
    config = SimpleNamespace(
        vpm_particle_spacing=0.1,
        authority_ramp_width=0.3,
        vpm_only_width=0.1,
        vpm_core_radius_ratio=1.0,
        freestream_velocity=[1.0, 0.0, 0.0],
        transfer_vorticity_cutoff=0.01,
        transfer_region_bounds=(-0.7, 0.7, -0.7, 0.7, -0.7, 0.7),
        transfer_amplification_cap=2.0,
        transfer_boundary_prune_multiplier=1.0,
        transfer_diagnostic_interval_steps=1,
    )
    fvm = SimpleNamespace(
        ibm=SimpleNamespace(bodies=[]),
        setup=SimpleNamespace(boundaries=[]),
        get_cell_centre_coordinates=lambda: np.array([[-0.9, 0.0, 0.0], [0.9, 0.0, 0.0]]),
    )
    transfer = VorticityTransfer(
        SimpleNamespace(
            setup=config,
            vpm_time_step_size=0.1,
            kinematic_viscosity=0.01,
            fvm_box=np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
            vpm_solver=None,
        )
    )

    transfer.setup(fvm)

    np.testing.assert_array_equal(transfer._box, config.transfer_region_bounds)


def test_free_wake_is_retained():
    pos = np.array([[1.0, 0.0, 0.0], [1.2, 0.1, 0.0]])
    circ = np.array([[0.0, 0.0, 0.2], [0.0, 0.1, 0.0]])
    result = continuous_transfer(
        pos,
        circ,
        BOX,
        H,
        vortex_strength_at_node=_zero_target,
        transfer_buffer_length=0.1,
        transfer_prune_threshold_abs=1.0e-12,
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


def test_local_redistribution_conserves_strength_and_impulse_locally():
    shape = (7, 7, 7)
    rng = np.random.default_rng(2)
    field = rng.normal(size=(*shape, 3)) * 1e-3
    # The node at the centre is pruned entirely: shrunk + removed == field.
    removed = np.zeros_like(field)
    removed[3, 3, 3] = field[3, 3, 3]
    shrunk = field.copy()
    shrunk[3, 3, 3] = 0.0

    out = redistribute_locally(removed, shrunk.reshape(-1, 3), shape).reshape(*shape, 3)

    # Total vortex_strength preserved.
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


def test_population_cap_preserves_integral_strength():
    rng = np.random.default_rng(3)
    pos = rng.uniform([1.0, -0.5, -0.5], [2.0, 0.5, 0.5], (8, 3))
    circ = rng.normal(size=(8, 3)) * 0.05
    result = continuous_transfer(
        pos,
        circ,
        BOX,
        H,
        vortex_strength_at_node=_zero_target,
        max_output_particles=4,
        transfer_prune_threshold_abs=1.0e-12,
    )

    assert result.n_total == 4
    assert result.n_population_pruned == 4
    np.testing.assert_allclose(result.circ.sum(axis=0), circ.sum(axis=0), atol=1.0e-12)
    before = 0.5 * np.cross(pos, circ).sum(axis=0)
    after = 0.5 * np.cross(result.pos, result.circ).sum(axis=0)
    np.testing.assert_allclose(after, before, atol=1.0e-12)
