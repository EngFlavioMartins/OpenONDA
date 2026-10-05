"""Renewal must preserve an already matching physical vorticity field."""

import numpy as np
import pytest

from source.coupler.stable_renewal import (
    blend_represented_state,
    build_stable_renewal_lattice,
    gaussian_represented_vortex_strength,
    renew_stable_overlap,
)


@pytest.mark.parametrize("slab", [False, True])
def test_matching_state_is_fixed_point_through_authority_ramp_and_wall(slab):
    h = 0.04
    shape = (35, 27, 6 if slab else 1)
    indices = np.indices(shape)
    strength = np.zeros((*shape, 3))
    strength[..., 2] = np.exp(-((indices[0] - 20) ** 2 + (indices[1] - 13) ** 2) / 35)
    wall_weight = (indices[0] >= 9).astype(float).reshape(-1)
    strength = strength.reshape(-1, 3) * wall_weight[:, None]
    authority = np.broadcast_to(np.linspace(0, 1, shape[0])[:, None, None], shape).ravel()
    options = {"dimensions": 3 if slab else 2}
    if slab:
        options.update(slip_slab_bounds=(-0.12, 0.12), lattice_origin_z=-0.10)
    physical = gaussian_represented_vortex_strength(
        strength, shape, h, core_radius=h, **options
    )
    # Production FVM targets have no support inside the wall. Gaussian tails
    # there are not an error to invert into neighboring fluid coefficients.
    physical *= wall_weight[:, None]

    result = blend_represented_state(
        strength, physical, authority, shape, h, core_radius=h,
        amplification_cap=1.8, output_weight=wall_weight, **options
    )

    np.testing.assert_array_equal(result.vortex_strength, strength)
    assert result.residual_before_correction == 0.0
    assert result.residual_after_correction == 0.0


def test_repeated_matching_renewals_do_not_damp_resolved_wake_mode():
    shape, h = (101, 17, 1), 0.04
    strength = np.zeros((*shape, 3))
    strength[..., 2] = np.sin(2 * np.pi * np.arange(shape[0])[:, None, None] * h / 0.2)
    strength = strength.reshape(-1, 3)
    initial = strength.copy()
    authority = np.full(len(strength), 0.5)
    for _ in range(25):
        physical = gaussian_represented_vortex_strength(
            strength, shape, h, core_radius=h, dimensions=2
        )
        strength = blend_represented_state(
            strength, physical, authority, shape, h, core_radius=h,
            amplification_cap=1.8, dimensions=2,
        ).vortex_strength
    np.testing.assert_array_equal(strength, initial)


def test_repeated_unresolved_wall_target_remains_bounded():
    shape, h = (25, 25, 1), 0.04
    indices = np.indices(shape)
    wall_weight = (indices[0] >= 11).astype(float).ravel()
    target = np.zeros((*shape, 3))
    target[..., 2] = ((-1.0) ** indices[1]) * np.exp(-((indices[0] - 11) / 1.5) ** 2)
    target = target.reshape(-1, 3) * wall_weight[:, None]
    strength = np.zeros_like(target)
    target_maximum = np.linalg.norm(target, axis=1).max()
    for _ in range(100):
        result = blend_represented_state(
            strength, target, np.ones(len(strength)), shape, h, core_radius=h,
            amplification_cap=1.8, dimensions=2, output_weight=wall_weight,
        )
        strength = result.vortex_strength
        assert np.linalg.norm(strength, axis=1).max() <= 1.8 * target_maximum * (1 + 1e-13)
        np.testing.assert_array_equal(strength[wall_weight == 0], 0.0)
        assert np.isfinite(strength).all()
    assert result.maximum_amplification == pytest.approx(1.8)


def test_one_saturated_node_does_not_stop_remote_correction():
    shape, h = (31, 31, 1), 0.04
    strength = np.zeros((*shape, 3))
    strength[8, 15, 0, 2] = 10.0
    strength = strength.reshape(-1, 3)
    physical = gaussian_represented_vortex_strength(
        strength, shape, h, core_radius=h, dimensions=2
    ).reshape(*shape, 3)
    physical[8, 15, 0, 2] += 0.01
    physical[23, 15, 0, 2] += 0.01
    result = blend_represented_state(
        strength, physical.reshape(-1, 3), np.ones(len(strength)), shape, h,
        core_radius=h, amplification_cap=1.8, dimensions=2,
    ).vortex_strength.reshape(*shape, 3)
    assert result[8, 15, 0, 2] == 10.0
    assert result[23, 15, 0, 2] > 0.01


@pytest.mark.parametrize("slab", [False, True])
def test_renewal_ignores_fractional_taper_inside_actual_solid(slab):
    from source.coupler.geometry import SolidBoundary, TriangulatedWall

    h = 0.04
    boundary = SolidBoundary(
        (TriangulatedWall.from_box((-2.0, 0.0, -2.0, 2.0, -2.0, 2.0), (-3.0, 3.0) * 3),)
    )
    lattice = build_stable_renewal_lattice(
        (-0.4, 0.4, -0.4, 0.4, -0.12, 0.12), h,
        buffer_length=0.08, authority_ramp_width=0.12,
        lattice_anchor=np.full(3, 0.02),
        fluid_weight_at_node=lambda points: np.clip(1 + points[:, 0] / h, 0, 1),
        interior_at_node=boundary.contains, solid_boundary=boundary,
        planar_span=None if slab else 0.24, slip_slab=slab,
    )
    partial_solid = lattice.solid_interior & (lattice.fluid_weight > 0)
    assert np.any(partial_solid)
    points = lattice.positions
    active = (
        (points[:, 0] > 0) & (points[:, 0] < 0.24) & (np.abs(points[:, 1]) < 0.24)
        & (lattice.fluid_weight == 1)
    )
    strength = np.zeros_like(points)
    strength[active, 2] = np.exp(
        -((points[active, 0] - 0.08) ** 2 + points[active, 1] ** 2) / 0.08**2
    )
    options = {"dimensions": 3 if slab else 2}
    if slab:
        options.update(slip_slab_bounds=(-0.12, 0.12), lattice_origin_z=lattice.origin[2])
    raw_target = gaussian_represented_vortex_strength(
        strength, lattice.shape, h, core_radius=h, **options
    )
    # The production caller multiplies this field by the smooth taper, which
    # still leaves a nonzero, unrepresentable target in partial_solid nodes.
    assert np.linalg.norm(raw_target[partial_solid]) > 0
    result = renew_stable_overlap(
        points[active], strength[active], lattice,
        fvm_vortex_strength_at_node=lambda _: raw_target,
        amplification_cap=1.8, prune_threshold=0.0, compute_diagnostics=True,
    )
    np.testing.assert_array_equal(result.position, points[active])
    np.testing.assert_allclose(result.vortex_strength, strength[active], rtol=0, atol=1e-14)
    assert result.representation_residual_before_prune < 1e-14
    assert result.representation_residual_after_prune < 1e-14
