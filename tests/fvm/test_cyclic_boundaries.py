import copy

import numpy as np
import pytest

from source.solvers.FVM.assemble import convection, diffusion, matrix_assembly
from source.solvers.FVM.fields import gradients
from source.solvers.FVM.mesh import geometry
from source.solvers.FVM.mesh.coupled import configure_cyclic_boundaries
from source.solvers.FVM.mesh.validation import MeshValidationError
from source.solvers.FVM.solve.simple_solver import (
    assemble_pressure_correction_equation_rhie_chow,
    update_scalar_boundaries,
)


def _periodic_x_mesh(hand_built_3d_mesh):
    mesh = copy.deepcopy(hand_built_3d_mesh)
    for patch in mesh["boundary"]:
        if patch["name"] == "xmin":
            patch.update(
                bc_type="cyclic",
                bc_type_velocity="cyclic",
                bc_type_p="cyclic",
                neighbourPatch="xmax",
            )
        elif patch["name"] == "xmax":
            patch.update(
                bc_type="cyclic",
                bc_type_velocity="cyclic",
                bc_type_p="cyclic",
                neighbourPatch="xmin",
            )
        else:
            patch.update(
                bc_type="zeroGradient",
                bc_type_velocity="zeroGradient",
                bc_type_p="zeroGradient",
            )
    geo = geometry.compute_mesh_geometry(mesh, gradient_scheme="lsq")
    configure_cyclic_boundaries(mesh, geo)
    geo.update(gradients.compute_lsq_geometry(mesh, geo))
    return mesh, geo


def test_cyclic_pairing_is_reciprocal_and_translational(hand_built_3d_mesh):
    mesh, geo = _periodic_x_mesh(hand_built_3d_mesh)
    pair = mesh["boundary_pair_faces"]
    coupled = np.flatnonzero(pair >= 0)

    assert coupled.size == 8
    assert np.array_equal(pair[pair[coupled]], coupled)
    assert np.all(mesh["boundary_neighbours"][coupled] >= 0)
    assert np.all(np.sum(geo["face_sf"][coupled] * geo["face_cf_vector"][coupled], axis=1) > 0)
    assert np.all((geo["face_weights"][coupled] > 0) & (geo["face_weights"][coupled] < 1))


def test_cyclic_sparse_coupling_is_symmetric_with_constant_nullspace(hand_built_3d_mesh):
    mesh, _ = _periodic_x_mesh(hand_built_3d_mesh)
    n_faces = mesh["n_faces"]
    coupled = mesh["boundary_neighbours"] >= 0
    flux = {
        "flux_cf": np.where(coupled, 1.0, 0.0),
        "flux_ff": np.where(coupled, -1.0, 0.0),
        "flux_vf": np.zeros(n_faces),
    }
    matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(flux, mesh)

    assert np.allclose(matrix.toarray(), matrix.toarray().T)
    assert np.allclose(matrix @ np.ones(mesh["n_elements"]), 0.0)


def test_cyclic_operators_preserve_a_constant_field(hand_built_3d_mesh):
    mesh, geo = _periodic_x_mesh(hand_built_3d_mesh)
    n_cells = mesh["n_elements"]
    n_total = n_cells + mesh["n_faces"] - mesh["n_interior_faces"]
    scalar = np.ones(n_total)
    update_scalar_boundaries(scalar, mesh, mesh["boundary"], field_name="p")

    grad = gradients.compute_lsq_gradient(scalar, mesh, geo)
    diffusive = diffusion.assemble_diffusion_term(
        scalar, grad, np.ones(n_cells), mesh, geo, mesh["boundary"]
    )
    matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(diffusive, mesh)
    rhs = matrix_assembly.assemble_rhs_from_fluxes_vectorized(diffusive, mesh)
    assert np.allclose(grad[:n_cells], 0.0, atol=1e-13)
    assert np.allclose(matrix @ scalar[:n_cells] - rhs, 0.0, atol=1e-13)

    velocity = np.zeros((n_total, 3))
    velocity[:, 0] = 1.0
    mdot = convection.compute_volumetric_face_flux(velocity, mesh, geo)
    convective = convection.assemble_convection_term(
        scalar, mdot, mesh, geo, mesh["boundary"], scheme="central"
    )
    matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(convective, mesh)
    rhs = matrix_assembly.assemble_rhs_from_fluxes_vectorized(convective, mesh)
    assert np.allclose(matrix @ scalar[:n_cells] - rhs, 0.0, atol=1e-13)


def test_cyclic_pressure_operator_retains_only_constant_nullspace(hand_built_3d_mesh):
    mesh, geo = _periodic_x_mesh(hand_built_3d_mesh)
    n_cells = mesh["n_elements"]
    n_total = n_cells + mesh["n_faces"] - mesh["n_interior_faces"]
    velocity = np.zeros((n_total, 3))
    pressure = np.zeros(n_total)
    momentum_diagonal = np.ones((n_cells, 3))

    matrix, rhs, _ = assemble_pressure_correction_equation_rhie_chow(
        velocity,
        momentum_diagonal,
        pressure,
        1.0,
        mesh,
        geo,
        mesh["boundary"],
        pressure_constraint="nullspace",
    )

    assert np.allclose(matrix.toarray(), matrix.toarray().T, atol=1e-13)
    assert np.allclose(matrix @ np.ones(n_cells), 0.0, atol=1e-13)
    assert abs(np.sum(rhs)) < 1e-13


def test_cyclic_pair_must_be_reciprocal(hand_built_3d_mesh):
    mesh = copy.deepcopy(hand_built_3d_mesh)
    for patch in mesh["boundary"]:
        patch.update(bc_type_velocity="zeroGradient", bc_type_p="zeroGradient")
    mesh["boundary"][0].update(bc_type_velocity="cyclic", bc_type_p="cyclic", neighbourPatch="xmax")
    geo = geometry.compute_mesh_geometry(mesh)

    with pytest.raises(MeshValidationError, match="must use cyclic"):
        configure_cyclic_boundaries(mesh, geo)
