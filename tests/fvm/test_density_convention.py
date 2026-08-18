"""Constant-density and kinematic-pressure convention checks."""

import numpy as np

from source.solvers.FVM.assemble.convection import compute_volumetric_face_flux
from source.solvers.FVM.assemble.momentum import assemble_momentum_equation
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.solve.simple_solver import (
    assemble_pressure_correction_equation_rhie_chow,
    correct_velocity_and_flux,
)

from ._structured_mesh import structured_box


def _case():
    mesh = structured_box(2, 2, 2)
    for boundary in mesh["boundary"]:
        boundary["bc_type_velocity"] = "zeroGradient"
        boundary["bc_type_p"] = "zeroGradient"
    geometry = compute_mesh_geometry(mesh)
    n_cells = mesh["n_elements"]
    n_boundary = mesh["n_faces"] - mesh["n_interior_faces"]
    rng = np.random.default_rng(42)
    velocity = rng.normal(scale=0.1, size=(n_cells + n_boundary, 3))
    pressure = rng.normal(scale=0.05, size=n_cells + n_boundary)
    flux = compute_volumetric_face_flux(velocity, mesh, geometry)
    return mesh, geometry, velocity, pressure, flux


def test_kinematic_momentum_operator_is_density_invariant():
    """Constant density cancels after dividing momentum by the reference density."""
    mesh, geometry, velocity, pressure, flux = _case()
    kwargs = {
        "U": velocity,
        "p": pressure,
        "phi": flux,
        "nu": 0.02,
        "mesh_data": mesh,
        "geo_data": geometry,
        "boundaries": mesh["boundary"],
        "convection_scheme": "central",
        "dt": 0.1,
        "U_old": velocity.copy(),
    }

    reference = assemble_momentum_equation(rho=1.0, **kwargs)
    denser = assemble_momentum_equation(rho=7.5, **kwargs)

    # The segregated components differ only in their RHS. Keeping three CSR
    # matrices/diagonals used to triple the dominant solver storage.
    assert reference["x"]["A"] is reference["y"]["A"] is reference["z"]["A"]
    assert reference["x"]["H"] is reference["y"]["H"] is reference["z"]["H"]

    for component in "xyz":
        np.testing.assert_allclose(
            reference[component]["A"].toarray(), denser[component]["A"].toarray()
        )
        np.testing.assert_allclose(reference[component]["b"], denser[component]["b"])


def test_pressure_correction_keeps_volumetric_flux_density_invariant():
    """Pressure correction returns U·Sf, not density-scaled mass flux."""
    mesh, geometry, velocity, pressure, _ = _case()
    diagonal = np.full((mesh["n_elements"], 3), 2.0)

    ref_matrix, ref_rhs, ref_flux = assemble_pressure_correction_equation_rhie_chow(
        velocity, diagonal, pressure, 1.0, mesh, geometry, mesh["boundary"]
    )
    dense_matrix, dense_rhs, dense_flux = assemble_pressure_correction_equation_rhie_chow(
        velocity, diagonal, pressure, 7.5, mesh, geometry, mesh["boundary"]
    )
    np.testing.assert_allclose(ref_matrix.toarray(), dense_matrix.toarray())
    np.testing.assert_allclose(ref_rhs, dense_rhs)
    np.testing.assert_allclose(ref_flux, dense_flux)

    correction = np.linspace(-0.1, 0.1, mesh["n_elements"])
    ref_velocity, corrected_ref_flux = correct_velocity_and_flux(
        velocity.copy(),
        ref_flux.copy(),
        correction,
        diagonal,
        mesh,
        geometry,
        mesh["boundary"],
        rho=1.0,
    )
    dense_velocity, corrected_dense_flux = correct_velocity_and_flux(
        velocity.copy(),
        dense_flux.copy(),
        correction,
        diagonal,
        mesh,
        geometry,
        mesh["boundary"],
        rho=7.5,
    )
    np.testing.assert_allclose(ref_velocity, dense_velocity)
    np.testing.assert_allclose(corrected_ref_flux, corrected_dense_flux)
