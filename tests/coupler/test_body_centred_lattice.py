"""A symmetric box must receive reflected transfer stencils on opposite walls."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from source.coupler import vorticity_transfer as transfer_module
from source.coupler.config.types import CouplerSetup
from source.coupler.stable_renewal import vortex_strength_from_velocity_trace
from source.coupler.vorticity_transfer import VorticityTransfer


def box_transfer(spacing, centre, seed=0):
    centre = np.asarray(centre, dtype=float)
    axis = np.linspace(-0.875, 0.875, 8)
    donors = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    donors = donors[np.any(np.abs(donors) > 0.5, axis=1)] + centre
    donors = donors[np.random.default_rng(seed).permutation(len(donors))]
    faces = np.concatenate((0.5 * np.eye(3), -0.5 * np.eye(3)))
    normals = -2 * faces
    bounds = np.array([-1.0, 1.0] * 3) + np.repeat(centre, 2)
    config = CouplerSetup(
        transfer_method="buffered_m4_renewal",
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
        setup=SimpleNamespace(boundaries=[SimpleNamespace(name="cube", mesh_type="wall")]),
        get_cell_centre_coordinates=lambda: donors,
        get_cell_volume=lambda: np.full(len(donors), 0.25**3),
        get_boundary_face_centre_coordinates=lambda patch: faces + centre,
        get_boundary_face_normal=lambda patch: normals,
    )
    transfer = VorticityTransfer(coupler)
    transfer.setup(fvm)
    return transfer


@pytest.mark.parametrize("spacing", [0.06, 0.0625, 0.2])
@pytest.mark.parametrize("centre", [(0, 0, 0), (1.75, -2.5, 0.375)])
def test_box_transfer_lattice_and_weights_preserve_reflections(spacing, centre):
    transfer = box_transfer(spacing, centre)
    lattice = transfer._stable_renewal_lattice
    tree = cKDTree(lattice.positions)
    for axis in range(3):
        reflected = lattice.positions.copy()
        reflected[:, axis] = 2 * centre[axis] - reflected[:, axis]
        distance, rows = tree.query(reflected)
        assert distance.max() < 2e-14
        np.testing.assert_allclose(lattice.mesh_weight[rows], lattice.mesh_weight, atol=2e-14)
        np.testing.assert_allclose(lattice.fvm_authority[rows], lattice.fvm_authority, atol=2e-14)


def test_box_lattice_translates_with_geometry_and_ignores_donor_order():
    displacement = np.array([1.75, -2.5, 0.375])
    original = box_transfer(0.06, np.zeros(3))._stable_renewal_lattice
    translated = box_transfer(0.06, displacement, seed=59)._stable_renewal_lattice
    np.testing.assert_allclose(translated.positions - displacement, original.positions, atol=2e-14)
    np.testing.assert_allclose(translated.mesh_weight, original.mesh_weight, atol=2e-14)
    np.testing.assert_allclose(translated.fvm_authority, original.fvm_authority, atol=2e-14)


@pytest.mark.parametrize("spacing", [0.0625, 0.2])
def test_commensurate_body_retains_previous_particle_lattice(spacing):
    lattice = box_transfer(spacing, np.zeros(3))._stable_renewal_lattice
    previous_anchor = -0.5 - 0.5 * spacing
    previous_coordinates = (lattice.positions - previous_anchor) / spacing
    np.testing.assert_allclose(previous_coordinates, np.rint(previous_coordinates), atol=2e-14)


@pytest.mark.parametrize("spacing", [0.06, 0.0625, 0.08, 0.09, 0.13, 0.2])
def test_no_fluid_bearing_control_cell_has_a_discarded_solid_centre(spacing):
    transfer = box_transfer(spacing, np.zeros(3))
    lattice = transfer._stable_renewal_lattice
    lower = np.maximum(lattice.positions - spacing / 2, -0.5)
    upper = np.minimum(lattice.positions + spacing / 2, 0.5)
    fluid_fraction = 1 - np.prod(np.maximum(upper - lower, 0), axis=1) / spacing**3
    assert fluid_fraction[lattice.solid_interior].max() < 2e-12

    def velocity_at(points):
        # Fully 3D, divergence-free velocity, zero on the complete cube wall.
        x, y, z = points.T
        shear = np.maximum(np.abs(y) - 0.5, 0)
        return np.column_stack((shear * (1 + z**2), np.zeros(len(x)), shear * np.sin(x)))

    circulation = vortex_strength_from_velocity_trace(lattice.positions, spacing, velocity_at)
    assert np.linalg.norm(circulation[lattice.solid_interior], axis=1).max() < 1e-14


@pytest.mark.parametrize("has_solid", [False, True])
@pytest.mark.parametrize("spacing", [0.18, 0.2])
def test_selective_trace_keeps_weighted_target_and_excluded_circulation(
    monkeypatch, has_solid, spacing
):
    transfer = box_transfer(spacing, np.zeros(3))
    if not has_solid:
        transfer._body_bounds = None
    lattice = transfer._stable_renewal_lattice
    centres = transfer._cell_centre
    velocity = centres**2 + np.roll(centres, 1, axis=1)
    gradient = np.zeros((len(centres), 3, 3))
    gradient[:, range(3), range(3)] = 2 * centres
    gradient[:, [2, 0, 1], [0, 1, 2]] = 1.0
    sample = transfer._velocity_trace.sample

    def full_velocity(points):
        result = sample(points, velocity, gradient)
        if has_solid:
            result *= transfer_module._smoothstep(
                transfer._signed_solid_distance(points), 0.0, spacing
            )[:, None]
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
