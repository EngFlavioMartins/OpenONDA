"""Discrete fixedFluxPressure compatibility tests."""

import numpy as np

from source.solvers.FVM.coupling.coupler_interface import CouplerInterfaceMixin
from source.solvers.FVM.fields import gradients
from source.solvers.FVM.solve.simple_solver import _update_fixed_flux_pressure_boundaries


def _one_cell_fixed_flux_case():
    # One unit cube cell. Face order: xmin, xmax, ymin, ymax, zmin, zmax.
    sf = np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        dtype=float,
    )
    normals = sf.copy()
    face_centres = 0.5 * normals
    mesh = {
        "n_elements": 1,
        "n_interior_faces": 0,
        "n_faces": 6,
        "owners": np.zeros(6, dtype=np.int32),
        "neighbours": np.zeros(0, dtype=np.int32),
        "boundary": [
            {
                "name": "cut",
                "startFace": 0,
                "nFaces": 6,
                "bc_type_U": "fixedValue",
                "bc_type_p": "fixedFluxPressure",
            }
        ],
    }
    geo = {
        "face_sf": sf,
        "face_cf_vector": face_centres,
        "face_centroids": face_centres,
        "face_weights": np.full(6, 0.5),
        "element_volumes": np.ones(1),
        "gradient_scheme": "gauss",
    }
    U_star = np.zeros((7, 3))
    U_star[0] = [1.0, 0.0, 0.0]
    U_star[1:] = [0.5, 0.0, 0.0]
    DU = np.full((1, 3), 0.25)
    return mesh, geo, U_star, DU


def test_fixed_flux_pressure_recovers_normal_momentum_balance():
    mesh, geo, U_star, DU = _one_cell_fixed_flux_case()
    p = np.zeros(7)

    grad = _update_fixed_flux_pressure_boundaries(p, U_star, DU, mesh, geo, mesh["boundary"])

    np.testing.assert_allclose(p[1:], [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(grad[0], [2.0, 0.0, 0.0], atol=1e-14)


def test_fixed_flux_pressure_accepts_explicit_pressure_free_face_flux():
    mesh, geo, U_star, DU = _one_cell_fixed_flux_case()
    p = np.zeros(7)
    pressure_free_flux = np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    grad = _update_fixed_flux_pressure_boundaries(
        p,
        U_star,
        DU,
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
        "n_elements": 1,
        "n_interior_faces": 0,
        "owners": np.zeros(2, dtype=np.int32),
    }
    interface.geo_data = {
        "face_sf": np.array([[-2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        "face_cf_vector": np.array([[-0.4, 0.0, 0.0], [0.6, 0.0, 0.0]]),
    }
    interface.boundaries = [{"name": "cut", "startFace": 0, "nFaces": 2}]
    interface.p = np.array([7.0, 0.0, 0.0])

    interface.set_neumann_pressure_boundary_condition(
        np.array([[2.0, 4.0, 0.0], [3.0, -1.0, 0.0]]), "cut"
    )

    boundary = interface.boundaries[0]
    assert boundary["bc_type_p"] == "fixedGradient"
    np.testing.assert_allclose(boundary["fixed_gradient_delta"], [-0.8, 1.8])
    np.testing.assert_allclose(interface.p[1:], [6.2, 8.8])
