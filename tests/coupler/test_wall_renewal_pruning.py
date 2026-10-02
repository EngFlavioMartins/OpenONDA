"""Renewal pruning cannot exchange invariants through an unresolved thin wall."""

from dataclasses import replace

import numpy as np
import pytest

from source.coupler.geometry import SolidBoundary, TriangulatedWall
from source.coupler.stable_renewal import (
    build_stable_renewal_lattice,
    renew_stable_overlap,
    scatter_m4_prime_to_lattice,
    vortex_invariants,
)


def thin_wall_lattice(*, planar=False):
    boundary = SolidBoundary(
        (TriangulatedWall.from_box((-0.01, 0.01, -4.0, 4.0, -4.0, 4.0), (-5.0, 5.0) * 3),)
    )
    lattice = build_stable_renewal_lattice(
        (-1.0, 1.0) * 3,
        1.0,
        buffer_length=0.0,
        authority_ramp_width=0.0,
        lattice_anchor=np.full(3, 0.5),
        interior_at_node=boundary.contains,
        solid_boundary=boundary,
        planar_span=1.0 if planar else None,
    )
    assert not lattice.solid_interior.any()
    assert np.count_nonzero(lattice.wall_links & 1) > 0
    return lattice, boundary


def target_strength(points):
    strength = np.zeros_like(points)
    left = np.all(points == [-0.5, -0.5, -0.5], axis=1)
    right = (points[:, 0] == 0.5) & (np.abs(points[:, 1]) == 0.5) & (np.abs(points[:, 2]) == 0.5)
    strength[left, 2] = 0.01
    strength[right, 2] = np.arange(1.0, 5.0)
    return strength


def renew(lattice, target=target_strength, **kwargs):
    return renew_stable_overlap(
        np.empty((0, 3)),
        np.empty((0, 3)),
        lattice,
        fvm_vortex_strength_at_node=target,
        prune_threshold=0.02,
        # Isolate pruning: cap=1 gives the supplied FVM-owned strengths
        # without a represented-field correction.
        amplification_cap=1.0,
        compute_diagnostics=False,
        **kwargs,
    )


def test_pruning_preserves_each_side_and_reuses_cached_wall_links(monkeypatch):
    lattice, boundary = thin_wall_lattice()

    def unexpected_query(*_args):
        pytest.fail("renewal pruning must reuse its cached wall links")

    monkeypatch.setattr(boundary, "blocks_segments", unexpected_query)
    result = renew(lattice)
    original = target_strength(lattice.positions)
    for left in (False, True):
        source = (lattice.positions[:, 0] < 0) == left
        output = (result.position[:, 0] < 0) == left
        before = vortex_invariants(lattice.positions[source], original[source])
        after = vortex_invariants(result.position[output], result.vortex_strength[output])
        np.testing.assert_allclose(
            after.total_vortex_strength, before.total_vortex_strength, atol=1e-14
        )
        np.testing.assert_allclose(after.linear_impulse, before.linear_impulse, atol=1e-14)
    # The weak disconnected particle cannot be discarded into the strong side.
    left = result.position[:, 0] < 0
    np.testing.assert_array_equal(result.position[left], [[-0.5, -0.5, -0.5]])
    np.testing.assert_array_equal(result.vortex_strength[left], [[0.0, 0.0, 0.01]])


def test_rank_deficient_component_keeps_its_original_support():
    lattice, _ = thin_wall_lattice()

    def line_target(points):
        strength = np.zeros_like(points)
        line = (points[:, 0] == -0.5) & (np.abs(points[:, 1]) == 0.5) & (points[:, 2] == -0.5)
        strength[line, 2] = [0.03, 0.06]
        return strength

    result = renew(lattice, target=line_target)
    original = line_target(lattice.positions)
    keep = np.linalg.norm(original, axis=1) > 0
    np.testing.assert_array_equal(result.position, lattice.positions[keep])
    np.testing.assert_array_equal(result.vortex_strength, original[keep])


def test_wall_renewal_capacity_does_not_trigger_global_recovery():
    lattice, _ = thin_wall_lattice()
    with pytest.raises(RuntimeError, match="capacity.*component-local"):
        renew(lattice, maximum_particle_count=4)


@pytest.mark.parametrize("planar", [False, True])
def test_wall_scatter_snaps_only_storage_roundoff_to_the_cardinal_node(planar):
    lattice, _ = thin_wall_lattice(planar=planar)
    displacement = np.array([1000.0, 0.0, 0.0])
    lattice = replace(
        lattice,
        origin=lattice.origin + displacement,
        positions=lattice.positions + displacement,
        renewal_bounds=lattice.renewal_bounds + np.repeat(displacement, 2),
    )
    exact = np.array([999.5, -0.5, 0.0 if planar else -0.5])
    stored = exact.astype(np.float32)
    stored[0] = np.nextafter(stored[0], np.float32(np.inf))
    assert stored[0] - exact[0] > 1e-5 * lattice.particle_spacing
    field = scatter_m4_prime_to_lattice(stored[None], [[0.0, 0.0, 1.0]], lattice)
    selected = np.linalg.norm(field, axis=1) > 0
    np.testing.assert_array_equal(lattice.positions[selected], exact[None])
    np.testing.assert_array_equal(field[selected], [[0.0, 0.0, 1.0]])
    renewed = renew_stable_overlap(
        stored[None],
        [[0.0, 0.0, 1.0]],
        replace(lattice, fvm_authority=np.zeros_like(lattice.fvm_authority)),
        fvm_vortex_strength_at_node=np.zeros_like,
        amplification_cap=1.0,
        compute_diagnostics=False,
    )
    np.testing.assert_array_equal(renewed.position, exact[None])
    np.testing.assert_array_equal(renewed.vortex_strength, [[0.0, 0.0, 1.0]])


@pytest.mark.parametrize("planar", [False, True])
def test_wall_scatter_rejects_off_lattice_sources_before_wall_crossing(planar):
    lattice, _ = thin_wall_lattice(planar=planar)
    position = np.array([[-0.25, -0.5, 0.0 if planar else -0.5]])
    strength = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="GBD-aligned particles"):
        scatter_m4_prime_to_lattice(position, strength, lattice)
    np.testing.assert_array_equal(position, [[-0.25, -0.5, 0.0 if planar else -0.5]])
    np.testing.assert_array_equal(strength, [[0.0, 0.0, 1.0]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_wall_scatter_uses_actual_position_storage_precision(dtype):
    lattice, _ = thin_wall_lattice()
    displacement = np.array([1e6, 0.0, 0.0])
    lattice = replace(
        lattice,
        origin=lattice.origin + displacement,
        positions=lattice.positions + displacement,
        renewal_bounds=lattice.renewal_bounds + np.repeat(displacement, 2),
    )
    position = np.array([[1e6 - 0.5, -0.5, -0.5]], dtype=dtype)
    if dtype == np.float32:
        with pytest.raises(ValueError, match="float32 position precision"):
            scatter_m4_prime_to_lattice(position, [[0.0, 0.0, 1.0]], lattice)
    else:
        field = scatter_m4_prime_to_lattice(position, [[0.0, 0.0, 1.0]], lattice)
        selected = np.linalg.norm(field, axis=1) > 0
        np.testing.assert_array_equal(lattice.positions[selected], position)
        np.testing.assert_array_equal(field[selected], [[0.0, 0.0, 1.0]])


def test_off_lattice_wall_scatter_reuses_gbd_visible_moment_correction():
    from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin

    lattice, boundary = thin_wall_lattice()
    physics = _GridDiffusionMixin()
    physics._init_grid_diffusion()
    physics.configure_body_classifier(
        boundary.contains,
        revision=boundary.revision,
        query_bounds=boundary.bounds,
        blocks_segments=boundary.blocks_segments,
    )
    lattice = replace(lattice, wall_scatter_correction=physics._m4_wall_corrections)
    position = np.array([[-0.25, -0.4, -0.3]])
    strength = np.array([[0.2, -0.1, 1.0]])
    field = scatter_m4_prime_to_lattice(position, strength, lattice)
    np.testing.assert_allclose(field[lattice.positions[:, 0] > 0], 0.0, atol=2e-8)
    np.testing.assert_allclose(field.sum(axis=0), strength[0], atol=2e-7)
    np.testing.assert_allclose(
        lattice.positions.T @ field, np.outer(position[0], strength[0]), atol=2e-7
    )
