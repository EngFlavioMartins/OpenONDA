"""Every body uses the same mesh-derived transfer lattice and wall contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import vorticity_transfer as transfer_module
from source.coupler.config.types import CouplerSetup
from source.coupler.geometry import TriangulatedWall
from source.coupler.interpolation import FVMVelocityInterpolator
from source.coupler.stable_renewal import vortex_strength_from_velocity_trace
from source.coupler.vorticity_transfer import VorticityTransfer
from source.solvers.fvm.immersed_boundary.body import ImmersedBody
from tests.coupler._solid_geometry import voxel_wall


def box_transfer(spacing, centre, seed=0, *, triangles=None, patches=("body",), bodies=()):
    centre = np.asarray(centre, dtype=float)
    axis = np.linspace(-0.875, 0.875, 8)
    donors = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    donors = donors[np.any(np.abs(donors) > 0.5, axis=1)] + centre
    donors = donors[np.random.default_rng(seed).permutation(len(donors))]
    if triangles is None:
        # Unit-box fixture; production receives these triangles from the mesh.
        wall = voxel_wall([(0, 0, 0)], origin=(-0.5, -0.5, -0.5))
        from vtkmodules.util.numpy_support import vtk_to_numpy

        points = vtk_to_numpy(wall._surface.GetPoints().GetData())
        indices = vtk_to_numpy(wall._surface.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        triangles = points[indices[:, ::-1]] + centre
    bounds = np.array([-1.0, 1.0] * 3) + np.repeat(centre, 2)
    config = CouplerSetup(
        transfer_region_bounds=tuple(0.8 * np.array([-1.0, 1.0] * 3) + np.repeat(centre, 2)),
        eta_blend_width=0.18,
    )
    coupler = SimpleNamespace(
        setup=config,
        kinematic_viscosity=0.001,
        fvm_box=bounds,
        vpm_core_radius_ratio=1.1,
        vpm_particle_spacing=spacing,
        vpm_time_step_size=0.01,
        vpm_solver=SimpleNamespace(
            viscous_scheme="GBD",
            setup=SimpleNamespace(
                viscous=SimpleNamespace(
                    gbd_threshold_mode="absolute",
                    gbd_threshold=0.0,
                )
            ),
        ),
    )
    fvm = SimpleNamespace(
        setup=SimpleNamespace(
            boundaries=[SimpleNamespace(name=name, mesh_type="wall") for name in patches]
        ),
        ibm=SimpleNamespace(bodies=bodies),
        get_cell_centre_coordinates=lambda: donors,
        get_cell_volume=lambda: np.full(len(donors), 0.25**3),
        get_wall_surface_triangles=lambda: triangles,
    )
    transfer = VorticityTransfer(coupler)
    transfer.setup(fvm)
    return transfer


@pytest.mark.parametrize("spacing", [0.06, 0.0625, 0.2])
def test_lattice_translates_with_mesh_and_ignores_donor_order(spacing):
    displacement = np.array([1.75, -2.5, 0.375])
    original = box_transfer(spacing, np.zeros(3))._stable_renewal_lattice
    translated = box_transfer(spacing, displacement, seed=59)._stable_renewal_lattice
    np.testing.assert_allclose(translated.positions - displacement, original.positions, atol=2e-14)
    np.testing.assert_allclose(translated.mesh_weight, original.mesh_weight, atol=2e-14)
    np.testing.assert_allclose(translated.fvm_authority, original.fvm_authority, atol=2e-14)


def test_wall_shape_and_patch_partition_do_not_select_a_lattice_phase():
    original = box_transfer(0.18, np.zeros(3))
    from vtkmodules.util.numpy_support import vtk_to_numpy

    wall = voxel_wall([(0, 0, 0), (1, 0, 0), (0, 1, 0)], origin=(-0.5, -0.5, -0.5), spacing=0.25)
    points = vtk_to_numpy(wall._surface.GetPoints().GetData())
    indices = vtk_to_numpy(wall._surface.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    triangles = points[indices[:, ::-1]]
    changed = box_transfer(0.18, np.zeros(3), seed=3, triangles=triangles, patches=("a", "b"))
    np.testing.assert_array_equal(original._lattice_anchor, changed._lattice_anchor)
    np.testing.assert_array_equal(
        original._stable_renewal_lattice.positions, changed._stable_renewal_lattice.positions
    )
    assert isinstance(original._solid_bodies[0], TriangulatedWall)
    assert isinstance(changed._solid_bodies[0], TriangulatedWall)
    # A concavity must remain fluid instead of being replaced by box bounds.
    assert changed._points_in_solid(
        [[-0.125, -0.125, -0.375]], include_boundary=False
    ).tolist() == [False]


def test_native_and_immersed_walls_are_combined():
    body = ImmersedBody(
        "immersed",
        [[0.7, 0, 0]],
        geometry={
            "type": "sphere",
            "centre": [0.7, 0, 0],
            "radius": 0.1,
        },
    )
    transfer = box_transfer(0.2, np.zeros(3), bodies=(body,))
    assert len(transfer._solid_bodies) == 2
    np.testing.assert_array_equal(
        transfer._points_in_solid([[0, 0, 0], [0.7, 0, 0], [0.9, 0, 0]], include_boundary=False),
        [True, True, False],
    )


def test_native_walls_require_surface_triangles_even_with_an_immersed_body():
    body = ImmersedBody(
        "immersed",
        [[0, 0, 0]],
        geometry={
            "type": "sphere",
            "centre": [0, 0, 0],
            "radius": 0.1,
        },
    )
    with pytest.raises(RuntimeError, match="wall surface triangles"):
        box_transfer(0.2, np.zeros(3), triangles=np.empty((0, 3, 3)), bodies=(body,))


def test_marker_only_immersed_body_cannot_silently_disable_wall_queries():
    body = ImmersedBody("markers", [[0, 0, 0]])
    with pytest.raises(ValueError, match="explicit solid geometry"):
        box_transfer(0.2, np.zeros(3), bodies=(body,))


@pytest.mark.parametrize("has_solid", [False, True])
@pytest.mark.parametrize("spacing", [0.18, 0.2])
def test_selective_trace_keeps_weighted_target_and_excluded_circulation(
    monkeypatch, has_solid, spacing
):
    transfer = box_transfer(spacing, np.zeros(3))
    if not has_solid:
        transfer.solid_boundary = None
        transfer._velocity_trace = FVMVelocityInterpolator(
            transfer._cell_centre,
            transfer._cell_tree,
        )
    lattice = transfer._stable_renewal_lattice
    centres = transfer._cell_centre
    velocity = centres**2 + np.roll(centres, 1, axis=1)
    gradient = np.zeros((len(centres), 3, 3))
    gradient[:, range(3), range(3)] = 2 * centres
    gradient[:, [2, 0, 1], [0, 1, 2]] = 1.0
    sample = transfer._velocity_trace.sample

    def full_velocity(points):
        weight = (
            transfer_module._smoothstep(transfer._signed_solid_distance(points), 0.0, spacing)
            if has_solid
            else np.ones(len(points))
        )
        result = np.zeros_like(points)
        fluid = weight > 0
        result[fluid] = sample(points[fluid], velocity, gradient) * weight[fluid, None]
        return result

    expected = vortex_strength_from_velocity_trace(lattice.positions, spacing, full_velocity)
    sampled_points = []

    prepare = transfer._velocity_trace.prepare

    def counted_prepare(points):
        sampled_points.append(points.copy())
        return prepare(points)

    monkeypatch.setattr(transfer._velocity_trace, "prepare", counted_prepare)
    monkeypatch.setattr(
        transfer_module,
        "replace_particles_from_buffered_m4_renewal",
        lambda vpm, **kwargs: kwargs["fvm_vortex_strength_at_node"](lattice.positions),
    )
    actual = transfer._transfer_buffered_m4_renewal(
        None, fvm_velocity=velocity, fvm_velocity_gradient=gradient
    )
    weight = (lattice.fluid_weight * lattice.mesh_weight)[:, None]
    np.testing.assert_array_equal(actual * weight, expected * weight)
    np.testing.assert_array_equal(
        actual * (1 - lattice.fluid_weight)[:, None],
        expected * (1 - lattice.fluid_weight)[:, None],
    )
    assert len(sampled_points) == 6
    assert sum(map(len, sampled_points)) < 6 * len(lattice.positions)
    if has_solid:
        assert all(np.all(transfer._signed_solid_distance(points) > 0) for points in sampled_points)
    updated = transfer._transfer_buffered_m4_renewal(
        None, fvm_velocity=2 * velocity, fvm_velocity_gradient=2 * gradient
    )
    np.testing.assert_array_equal(updated, 2 * actual)
    assert len(sampled_points) == 6
