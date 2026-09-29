"""Thin and non-axis-aligned surfaces must block remeshing and diffusion."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin, _m4_prime_1d
from tests.coupler._solid_geometry import wall_case


@ti.data_oriented
class WallGrid(_GridDiffusionMixin):
    pass


@pytest.mark.parametrize("variable_viscosity", [False, True])
def test_thin_wall_blocks_flux_and_remesh_with_two_fluid_endpoints(variable_viscosity):
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu)
    boundary, *_ = wall_case("thin")
    physics = WallGrid()
    physics._init_grid_diffusion()
    physics.configure_body_classifier(
        boundary.contains,
        revision=boundary.revision,
        query_bounds=boundary.bounds,
        blocks_segments=boundary.blocks_segments,
    )
    shape = (7, 7, 7)
    origin = np.array([-0.25, -0.3, -0.3])
    h = 0.1
    physics._grid_a = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    physics._grid_b = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    physics._body_mask_grid = ti.field(dtype=ti.i32, shape=shape)
    physics._prepare_body_mask_current_grid(origin, h, *shape)
    assert not physics._body_mask_grid.to_numpy().any()  # Plate is thinner than a cell.
    links = physics._body_link_grid.to_numpy()
    assert np.all(links[2] & 1)

    positions = np.array([[0.03, 0, 0]], dtype=np.float32)
    strengths = np.array([[0.2, -0.1, 1.0]], dtype=np.float32)
    position = ti.Vector.field(3, dtype=ti.f32, shape=1)
    strength = ti.Vector.field(3, dtype=ti.f32, shape=1)
    position.from_numpy(positions)
    strength.from_numpy(strengths)
    physics._m4_scatter_gpu_kernel(position, strength, physics._grid_a, *origin, h, *shape, 0, 1)
    indices, corrections, budget = physics._m4_wall_corrections(
        positions, strengths, origin, h, shape
    )
    assert budget["wall_adjacent_particles"] == 1
    physics._add_sparse_wall_corrections_kernel(
        physics._grid_a,
        np.ascontiguousarray(indices.reshape(-1)),
        corrections,
        len(indices),
    )
    before = physics._grid_a.to_numpy()
    np.testing.assert_allclose(before[:3], 0, atol=1e-7)
    np.testing.assert_allclose(before.sum(axis=(0, 1, 2)), strengths[0], atol=2e-6)
    if variable_viscosity:
        viscosity = ti.field(dtype=ti.f32, shape=shape)
        viscosity.from_numpy(
            np.broadcast_to(np.linspace(0.01, 0.02, 7)[:, None, None], shape).astype(np.float32)
        )
        physics._laplacian_step_variable_gpu_kernel(
            physics._grid_a,
            physics._grid_b,
            viscosity,
            physics._body_mask_grid,
            0.01,
            h,
            *shape,
        )
    else:
        physics._laplacian_step_gpu_kernel(
            physics._grid_a, physics._grid_b, physics._body_mask_grid, 0.02, *shape
        )
    after = physics._grid_b.to_numpy()
    np.testing.assert_allclose(after[:3], 0, atol=1e-7)
    np.testing.assert_allclose(after.sum(axis=(0, 1, 2)), strengths[0], atol=2e-6)


def test_pruning_cannot_discard_a_sparse_component_across_a_thin_wall():
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu)
    boundary, *_ = wall_case("thin")
    physics = WallGrid()
    physics._init_grid_diffusion()
    physics.configure_body_classifier(
        boundary.contains,
        revision=boundary.revision,
        query_bounds=boundary.bounds,
        blocks_segments=boundary.blocks_segments,
    )
    shape, h = (7, 3, 3), 0.1
    origin = np.array([-0.25, -0.1, -0.1])
    physics._body_mask_grid = ti.field(dtype=ti.i32, shape=shape)
    physics._prepare_body_mask_current_grid(origin, h, *shape)
    grid = np.zeros((*shape, 3), dtype=np.float32)
    grid[:3, :, :, 2] = 1
    grid[3:5, 1, 1, 2] = 1e-4
    magnitude = np.linalg.norm(grid, axis=-1)
    labels = physics._wall_recovery_labels(magnitude, np.zeros(shape, dtype=int))
    assert labels[2, 1, 1] != labels[3, 1, 1]
    retained = np.where(magnitude > 0.1)
    ix, iy, iz, added, strict = physics._augment_moment_recovery_support(
        grid,
        magnitude,
        *retained,
        origin,
        h,
        29,
        labels=labels,
        strict_labels=True,
    )
    assert strict and added == 2
    recovered = physics._redistribute_pruned_moments(
        grid,
        magnitude,
        ix,
        iy,
        iz,
        origin,
        h,
        labels=labels,
        strict_labels=True,
    )
    np.testing.assert_allclose(recovered[ix >= 3].sum(axis=0), [0, 0, 2e-4], atol=1e-10)
    with pytest.raises(RuntimeError, match="regeneration cap"):
        physics._augment_moment_recovery_support(
            grid,
            magnitude,
            *retained,
            origin,
            h,
            27,
            labels=labels,
            strict_labels=True,
        )


@pytest.mark.parametrize("shape_name", ["rotated", "concave", "multiple", "curved", "thin"])
@pytest.mark.parametrize("phase", [0.0, 0.37])
def test_general_wall_remap_conserves_moments_and_refines_induced_velocity(shape_name, phase):
    boundary, start, _, normal = wall_case(shape_name)
    contact, normals = boundary.closest_surface(start[None])
    normal = normals[0]
    kernel = make_vortex_kernel("GAUSSIAN")
    strengths = np.array([[0.25, -0.18, 0.9]])
    errors = []
    spacings = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
    for h in spacings:
        physics = WallGrid()
        physics._init_grid_diffusion()
        physics.configure_body_classifier(
            boundary.contains,
            revision=boundary.revision,
            query_bounds=boundary.bounds,
            blocks_segments=boundary.blocks_segments,
        )
        source = contact + 0.3 * h * normal
        # Hold the wall-relative lattice phase fixed during refinement.
        origin = contact[0] - 20 * h + np.array([phase * h, 0, 0])
        shape = (50, 50, 50)
        fraction = (source[0] - origin) / h
        offsets = np.stack(np.meshgrid(*([np.arange(-1, 3)] * 3), indexing="ij"), axis=-1)
        indices = np.floor(fraction).astype(int) + offsets.reshape(-1, 3)
        weights = np.prod(_m4_prime_1d(fraction - indices), axis=1)
        values = {
            tuple(index): weight * strengths[0]
            for index, weight in zip(indices, weights, strict=True)
        }
        corrected_indices, correction, _ = physics._m4_wall_corrections(
            source, strengths, origin, h, shape
        )
        for index, delta in zip(corrected_indices, correction, strict=True):
            values[tuple(index)] = values.get(tuple(index), np.zeros(3)) + delta
        nodes = origin + h * np.array(list(values))
        gamma = np.array(list(values.values()))
        gamma[boundary.contains(nodes)] = 0
        np.testing.assert_allclose(gamma.sum(axis=0), strengths[0], atol=1e-6)
        np.testing.assert_allclose(
            (nodes[:, :, None] * gamma[:, None, :]).sum(axis=0),
            source[0, :, None] * strengths,
            atol=1e-6,
        )
        hidden = boundary.blocks_segments(np.broadcast_to(source, nodes.shape), nodes)
        np.testing.assert_allclose(gamma[hidden], 0, atol=1e-7)
        targets = source + np.array([[0.4, 0.3, 0.2], [-0.3, 0.4, 0.3], [0.5, -0.2, 0.4]])
        actual = kernel.velocity_pair(
            targets[:, None, :] - nodes[None, :, :], gamma[None, :, :], 0.08, 0.08
        ).sum(axis=1)
        expected = kernel.velocity_pair(targets - source, strengths, 0.08, 0.08)
        errors.append(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    # Changing lattice phase can make individual errors non-monotone.
    # Require convergence over five levels, including the asymptotic range.
    order = np.polyfit(np.log(spacings[-4:]), np.log(errors[-4:]), 1)[0]
    assert order > 1.5, (order, errors)
    assert errors[-1] < 0.1 * errors[0], errors
