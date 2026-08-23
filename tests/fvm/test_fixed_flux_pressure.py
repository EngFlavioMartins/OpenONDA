"""Discrete fixedFluxPressure compatibility tests."""

import numpy as np

from source.solvers.fvm.coupling.coupler_interface import CouplerInterfaceMixin
from source.solvers.fvm.fields import gradients
from source.solvers.fvm.solve.simple_solver import _update_fixed_flux_pressure_boundaries


def _one_cell_fixed_flux_case():
    # One unit cube cell. Face order: xmin, xmax, ymin, ymax, zmin, zmax.
    sf = np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        dtype=float,
    )
    normals = sf.copy()
    face_centres = 0.5 * normals
    mesh = {
        "n_cells": 1,
        "n_interior_faces": 0,
        "n_faces": 6,
        "owners": np.zeros(6, dtype=np.int32),
        "neighbours": np.zeros(0, dtype=np.int32),
        "boundary": [
            {
                "name": "cut",
                "start_face": 0,
                "n_faces": 6,
                "velocity_type": "fixedValue",
                "pressure_type": "fixedFluxPressure",
            }
        ],
    }
    geo = {
        "face_area_vector": sf,
        "cell_connection_vector": face_centres,
        "face_centre": face_centres,
        "face_interpolation_weight": np.full(6, 0.5),
        "cell_volume": np.ones(1),
        "gradient_scheme": "gauss",
    }
    U_star = np.zeros((7, 3))
    U_star[0] = [1.0, 0.0, 0.0]
    U_star[1:] = [0.5, 0.0, 0.0]
    pressure_velocity_coefficient = np.full((1, 3), 0.25)
    return mesh, geo, U_star, pressure_velocity_coefficient


def test_fixed_flux_pressure_recovers_normal_momentum_balance():
    mesh, geo, U_star, pressure_velocity_coefficient = _one_cell_fixed_flux_case()
    p = np.zeros(7)

    grad = _update_fixed_flux_pressure_boundaries(
        p, U_star, pressure_velocity_coefficient, mesh, geo, mesh["boundary"]
    )

    np.testing.assert_allclose(p[1:], [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(grad[0], [2.0, 0.0, 0.0], atol=1e-14)


def test_fixed_flux_pressure_accepts_explicit_pressure_free_face_flux():
    mesh, geo, U_star, pressure_velocity_coefficient = _one_cell_fixed_flux_case()
    p = np.zeros(7)
    pressure_free_flux = np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    grad = _update_fixed_flux_pressure_boundaries(
        p,
        U_star,
        pressure_velocity_coefficient,
        mesh,
        geo,
        mesh["boundary"],
        pressure_free_face_flux=pressure_free_flux,
    )

    np.testing.assert_allclose(p[1:], [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(grad[0], [2.0, 0.0, 0.0], atol=1e-14)
    # Re-evaluating the public gradient operator must see the same trace.
    got = gradients.compute_gauss_gradient(p, mesh, geo).squeeze(-1)
    np.testing.assert_allclose(got[0], [2.0, 0.0, 0.0], atol=1e-14)


def test_vector_neumann_pressure_gradient_sets_face_increment():
    interface = CouplerInterfaceMixin()
    interface.parallel = None
    interface.mesh_data = {
        "n_cells": 1,
        "n_interior_faces": 0,
        "owners": np.zeros(2, dtype=np.int32),
    }
    interface.geo_data = {
        "face_area_vector": np.array([[-2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        "cell_connection_vector": np.array([[-0.4, 0.0, 0.0], [0.6, 0.0, 0.0]]),
    }
    interface.boundaries = [{"name": "cut", "start_face": 0, "n_faces": 2}]
    interface.kinematic_pressure = np.array([7.0, 0.0, 0.0])

    interface.set_neumann_pressure_boundary_condition(
        np.array([[2.0, 4.0, 0.0], [3.0, -1.0, 0.0]]), "cut"
    )

    boundary = interface.boundaries[0]
    assert boundary["pressure_type"] == "fixedGradient"
    np.testing.assert_allclose(boundary["fixed_gradient_delta"], [-0.8, 1.8])
    np.testing.assert_allclose(interface.kinematic_pressure[1:], [6.2, 8.8])
