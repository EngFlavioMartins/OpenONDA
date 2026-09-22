"""Curved-body exclusion for grid-based VPM diffusion."""

from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin


@ti.data_oriented
class _Harness(_GridDiffusionMixin):
    pass


def _ensure_taichi_cpu() -> None:
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu)


def test_cylinder_body_mask_excludes_only_open_solid_interior():
    _ensure_taichi_cpu()
    physics = _Harness()
    physics._init_grid_diffusion()
    physics._body_mask_grid = ti.field(dtype=ti.i32, shape=(7, 7, 7))
    physics.configure_body_cylinder((-0.5, 0.5, -0.5, 0.5, -1.0, 1.0), axis="z")

    physics._prepare_body_mask_current_grid(np.array([-1.5, -1.5, -1.5]), 0.5, 7, 7, 7)
    mask = physics._body_mask_grid.to_numpy()

    assert mask[3, 3, 3] == 1  # cylinder centre
    assert mask[4, 3, 3] == 0  # radial surface
    assert mask[3, 3, 5] == 0  # end-cap surface
    assert mask[5, 3, 3] == 0  # exterior fluid


def test_cylinder_body_mask_rejects_noncircular_transverse_bounds():
    physics = _Harness()
    physics._init_grid_diffusion()
    with pytest.raises(ValueError, match="circular diameter"):
        physics.configure_body_cylinder((-0.5, 0.5, -0.25, 0.25, -1.0, 1.0), axis=2)


def test_wall_classifier_mask_is_cached_and_follows_lattice_phase():
    _ensure_taichi_cpu()
    physics = _Harness()
    physics._init_grid_diffusion()
    physics._body_mask_grid = ti.field(dtype=ti.i32, shape=(7, 7, 7))
    calls = []

    def interior(points):
        calls.append(len(points))
        return (points[:, 0] ** 2 + points[:, 1] ** 2 < 0.25) & (np.abs(points[:, 2]) < 1.0)

    physics.configure_body_classifier(interior, revision="curved-wall-1")
    origin = np.array([-1.5, -1.5, -1.5])
    physics._prepare_body_mask_current_grid(origin, 0.5, 7, 7, 7)
    mask = physics._body_mask_grid.to_numpy()
    assert mask[3, 3, 3] == 1
    assert mask[4, 3, 3] == 0
    assert calls == [343]

    physics._prepare_body_mask_current_grid(origin, 0.5, 7, 7, 7)
    assert calls == [343]
    physics._prepare_body_mask_current_grid(origin + [0.25, 0.0, 0.0], 0.5, 7, 7, 7)
    assert calls == [343, 343]
    assert physics._body_mask_grid.to_numpy()[3, 3, 3] == 1


def test_wall_classifier_requires_one_flag_per_node():
    _ensure_taichi_cpu()
    physics = _Harness()
    physics._init_grid_diffusion()
    physics._body_mask_grid = ti.field(dtype=ti.i32, shape=(3, 3, 3))
    physics.configure_body_classifier(lambda points: [True], revision="bad")
    with pytest.raises(RuntimeError, match="one flag per GBD node"):
        physics._prepare_body_mask_current_grid(np.zeros(3), 0.5, 3, 3, 3)


def test_cylinder_particle_classification_uses_strict_interior():
    physics = _Harness()
    physics._init_grid_diffusion()
    physics.configure_body_cylinder((-0.5, 0.5, -0.5, 0.5, -1.0, 1.0))
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.51, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(
        physics._body_interior_at_particles(points), [True, False, False, False]
    )


@pytest.mark.parametrize("phase", [0.0, 0.2])
def test_wall_adjacent_m4_preserves_local_strength_and_first_moment(phase):
    physics = _Harness()
    physics._init_grid_diffusion()
    physics.configure_body_cylinder((-0.5, 0.5, -0.5, 0.5, -1.0, 1.0))
    position = np.array([[0.57, 0.13, 0.1]])
    strength = np.array([[0.2, -0.4, 1.0]])
    origin = np.array([-1.5 + phase, -1.5, -1.5])
    indices, delta, diagnostics = physics._m4_wall_corrections(
        position, strength, origin, 0.25, (17, 17, 17)
    )
    assert diagnostics["wall_adjacent_particles"] == 1
    assert diagnostics["excluded_signed_weight_l1"] > 0.0
    assert len(indices) == len(delta)
    fractional = (position[0] - origin) / 0.25
    base = np.floor(fractional).astype(int)
    grid = np.stack(np.meshgrid(*([np.arange(-1, 3)] * 3), indexing="ij"), axis=-1)
    all_indices = base + grid.reshape(-1, 3)
    from source.solvers.vpm.physics.diffusion.grid import _m4_prime_1d

    weights = np.prod(_m4_prime_1d(fractional - all_indices), axis=1)
    nodes = origin + 0.25 * all_indices
    fluid = ~physics._body_interior_at_particles(nodes)
    net = (weights[fluid, None] * strength).sum(axis=0) + delta.sum(axis=0)
    moment = (weights[fluid, None, None] * strength[None, :, :] * nodes[fluid, :, None]).sum(
        axis=0
    ) + (delta[:, None, :] * (origin + 0.25 * indices)[:, :, None]).sum(axis=0)
    np.testing.assert_allclose(net, strength[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(moment, position[0, :, None] * strength, rtol=1e-6, atol=1e-6)


def test_device_wall_scatter_and_zero_flux_diffusion_conserve_fluid_strength():
    _ensure_taichi_cpu()
    physics = _Harness()
    physics._init_grid_diffusion()
    shape = (17, 17, 17)
    physics._grid_a = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    physics._grid_b = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    physics._body_mask_grid = ti.field(dtype=ti.i32, shape=shape)
    physics.configure_body_cylinder((-0.5, 0.5, -0.5, 0.5, -1.0, 1.0))
    positions = np.array([[0.57, 0.13, 0.1]], dtype=np.float32)
    strengths = np.array([[0.2, -0.4, 1.0]], dtype=np.float32)
    source_position = ti.Vector.field(3, dtype=ti.f32, shape=1)
    source_strength = ti.Vector.field(3, dtype=ti.f32, shape=1)
    source_position.from_numpy(positions)
    source_strength.from_numpy(strengths)
    origin = np.array([-1.5, -1.5, -1.5])
    physics._m4_scatter_gpu_kernel(
        source_position,
        source_strength,
        physics._grid_a,
        *origin,
        0.25,
        *shape,
        0,
        1,
    )
    nodes, correction, _ = physics._m4_wall_corrections(positions, strengths, origin, 0.25, shape)
    physics._add_sparse_wall_corrections_kernel(
        physics._grid_a, np.ascontiguousarray(nodes.reshape(-1)), correction, len(nodes)
    )
    physics._prepare_body_mask_current_grid(origin, 0.25, *shape)
    physics._apply_body_mask_current_grid(*shape)
    before = physics._grid_a.to_numpy()
    mask = physics._body_mask_grid.to_numpy().astype(bool)
    assert np.all(before[mask] == 0.0)
    np.testing.assert_allclose(before.sum(axis=(0, 1, 2)), strengths[0], atol=1e-6)

    physics._laplacian_step_gpu_kernel(
        physics._grid_a,
        physics._grid_b,
        physics._body_mask_grid,
        0.1,
        *shape,
    )
    after = physics._grid_b.to_numpy()
    assert np.all(after[mask] == 0.0)
    np.testing.assert_allclose(after.sum(axis=(0, 1, 2)), strengths[0], atol=1e-6)
