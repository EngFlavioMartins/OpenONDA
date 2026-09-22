"""Solid geometry and sparse wall correction across slip-slab images."""

from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin, _m4_prime_1d


@ti.data_oriented
class _Harness(_GridDiffusionMixin):
    pass


def _cpu() -> None:
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu)


@pytest.mark.parametrize("origin_z,inside_index,mirror_index", [(-3.5, 4, 3), (-3.0, 4, 2)])
def test_generic_body_mask_folds_half_and_node_phase(origin_z, inside_index, mirror_index):
    _cpu()
    physics = _Harness()
    physics._init_grid_diffusion()
    physics._slip_slab_bounds = (0.0, 6.0)
    physics._body_mask_grid = ti.field(dtype=ti.i32, shape=(3, 3, 15))
    calls = []

    def interior(points):
        calls.append(len(points))
        return (
            (points[:, 0] ** 2 + points[:, 1] ** 2 < 0.25)
            & (points[:, 2] > 0.0)
            & (points[:, 2] < 1.5)
        )

    physics.configure_body_classifier(
        interior, revision="varying-z", query_bounds=[-0.5, 0.5, -0.5, 0.5, 0, 6]
    )
    origin = np.array([-1.0, -1.0, origin_z])
    physics._prepare_body_mask_current_grid(origin, 1.0, 3, 3, 15)
    mask = physics._body_mask_grid.to_numpy()
    assert mask[1, 1, inside_index] == mask[1, 1, mirror_index] == 1
    assert mask[0, 1, mirror_index] == 0
    assert sum(calls) == 135
    physics._prepare_body_mask_current_grid(origin, 1.0, 3, 3, 15)
    assert sum(calls) == 135
    old_key = physics._body_mask_cache_key
    physics._slip_slab_bounds = (0.0, 8.0)
    physics._prepare_body_mask_current_grid(origin, 1.0, 3, 3, 15)
    assert sum(calls) == 270
    assert physics._body_mask_cache_key != old_key


def test_slab_analytic_cylinder_uses_same_folded_mask_as_particle_classification():
    _cpu()
    physics = _Harness()
    physics._init_grid_diffusion()
    physics._slip_slab_bounds = (0.0, 6.0)
    physics._body_mask_grid = ti.field(dtype=ti.i32, shape=(3, 3, 11))
    physics.configure_body_cylinder((-0.5, 0.5, -0.5, 0.5, 0, 6), axis="z")
    origin = np.array([-1.0, -1.0, -2.0])
    physics._prepare_body_mask_current_grid(origin, 1.0, 3, 3, 11)
    mask = physics._body_mask_grid.to_numpy()
    assert mask[1, 1, 1] == mask[1, 1, 3] == 1
    assert mask[1, 1, 2] == 0  # strict boundary remains fluid
    points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
    np.testing.assert_array_equal(physics._body_interior_at_particles(points), [1, 1, 0])
    physics._slip_slab_bounds = None
    np.testing.assert_array_equal(physics._body_interior_at_particles(points), [0, 1, 0])


def test_sparse_reflection_matches_large_grid_mirror_and_works_with_three_cell_halo():
    physics = _Harness()
    physics._init_grid_diffusion()
    physics._slip_slab_bounds = (-0.48, 0.48)
    physics.configure_body_classifier(
        lambda p: p[:, 0] ** 2 + p[:, 1] ** 2 < 0.1**2,
        revision="radial-cylinder",
        query_bounds=[-0.1, 0.1, -0.1, 0.1, -0.48, 0.48],
    )
    h = 0.08
    origin = np.array([-0.24, -0.24, -0.72])
    shape = (7, 7, 19)
    strength = np.array([[0.3, -0.2, 1.0]])
    physical = np.array([[0.09, 0.02, -0.272]])
    image = physical.copy()
    image[:, 2] = 2 * (-0.48) - image[:, 2]
    parity = np.array([-1.0, -1.0, 1.0])
    nodes, values, budget = physics._m4_wall_corrections(physical, strength, origin, h, shape)
    assert budget["wall_adjacent_particles"] == 1
    reflected_nodes, reflected_values, image_budget = physics._reflect_sparse_wall_corrections(
        nodes, values, origin[2], h, shape[2], (-0.48, 0.48)
    )
    assert image_budget["mirrored_sparse_correction_nodes"] > 0
    # Recomputing the image on the active three-cell halo needs unavailable
    # stencil nodes; the sparse reflection deliberately clips those nodes.
    physics._body_query_bounds = None
    with pytest.raises(RuntimeError, match="support leaves the diffusion lattice"):
        physics._m4_wall_corrections(image, strength * parity, origin, h, shape)
    large_origin = origin.copy()
    large_origin[2] -= 3 * h
    reference_nodes, reference_values, _ = physics._m4_wall_corrections(
        image, strength * parity, large_origin, h, (7, 7, 22)
    )
    reference_nodes[:, 2] -= 3
    retained = (reference_nodes[:, 2] >= 0) & (reference_nodes[:, 2] < shape[2])
    reference_nodes, reference_values = reference_nodes[retained], reference_values[retained]
    # The helper includes physical nodes first; filter to the lower image.
    mirrored = reflected_nodes[:, 2] < 3
    reflected_nodes, reflected_values = reflected_nodes[mirrored], reflected_values[mirrored]
    order = np.lexsort(reflected_nodes.T[::-1])
    reference_order = np.lexsort(reference_nodes.T[::-1])
    np.testing.assert_array_equal(reflected_nodes[order], reference_nodes[reference_order])
    np.testing.assert_allclose(
        reflected_values[order], reference_values[reference_order], rtol=0, atol=0
    )
    # Cached mask and direct classifier give the same physical correction.
    ix, iy, _ = np.indices(shape)
    radial_mask = (origin[0] + ix * h) ** 2 + (origin[1] + iy * h) ** 2 < 0.1**2
    physics._body_mask_host = radial_mask
    physics._body_mask_cache_key = (
        0,
        0,
        tuple(float(x) for x in origin.astype(np.float32)),
        float(np.float32(h)),
    )
    cached_nodes, cached_values, _ = physics._m4_wall_corrections(
        physical, strength, origin, h, shape
    )
    np.testing.assert_array_equal(cached_nodes, nodes)
    np.testing.assert_allclose(cached_values, values, rtol=0, atol=0)


@pytest.mark.parametrize("plane_phase", [0.0, 0.5])
def test_zero_viscosity_remesh_preserves_normal_strength_at_slip_planes(plane_phase):
    """Full physical+image M4 remesh must not double a node-face source."""
    physics = _Harness()
    physics._init_grid_diffusion()
    physics._slip_slab_bounds = (0.0, 6.0)
    origin_z = -3.0 + plane_phase
    z_nodes = origin_z + np.arange(14)
    if plane_phase == 0.0:
        sources = [(0.0, 1.0), (0.5, 0.7), (5.5, 0.4)]
    else:
        sources = [(0.5, 1.0), (1.0, 0.7), (5.5, 0.4)]
    expected = sum(value for _z, value in sources)
    for _ in range(2):
        grid = np.zeros((1, 1, len(z_nodes), 3), dtype=np.float64)
        for z, gamma in sources:
            for image_z in (z, -z, 12.0 - z):
                grid[0, 0, :, 2] += gamma * _m4_prime_1d(z_nodes - image_z)
        outside = (z_nodes < 0.0) | (z_nodes > 6.0)
        grid[:, :, outside, :] = 0.0
        physics._weight_slip_slab_endpoint_nodes(grid, origin_z, 1.0, (0.0, 6.0))
        assert grid[0, 0, :, 2].sum() == pytest.approx(expected, abs=1e-6)
        iz = np.flatnonzero(grid[0, 0, :, 2] != 0)
        ix = np.zeros(len(iz), dtype=int)
        built = physics._build_diffusion_particle_arrays(
            ix,
            ix,
            iz,
            grid,
            np.array([0.0, 0.0, origin_z]),
            1.0,
            0.0,
            0.0,
            None,
            0,
            np.zeros(grid.shape[:3], dtype=np.int32),
            np.zeros(grid.shape[:3], dtype=np.int32),
            slab_endpoint_volume=True,
        )
        if plane_phase == 0.0:
            assert np.all(built["particle_volume"][built["position"][:, 2] == 0.0] == 0.5)
            dvh_style = physics._build_diffusion_particle_arrays(
                ix,
                ix,
                iz,
                grid,
                np.array([0.0, 0.0, origin_z]),
                1.0,
                0.0,
                0.0,
                None,
                0,
                np.zeros(grid.shape[:3], dtype=np.int32),
                np.zeros(grid.shape[:3], dtype=np.int32),
            )
            assert np.all(dvh_style["particle_volume"] == 1.0)
        sources = list(
            zip(
                built["position"][:, 2].tolist(),
                built["vortex_strength"][:, 2].tolist(),
                strict=True,
            )
        )
