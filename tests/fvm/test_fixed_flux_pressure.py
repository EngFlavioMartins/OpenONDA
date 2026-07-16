"""Discrete fixedFluxPressure compatibility tests."""

import numpy as np

from source.solvers.FVM.fields import gradients
from source.solvers.FVM.solve.simple_solver import _update_fixed_flux_pressure_boundaries


def test_fixed_flux_pressure_recovers_linear_normal_momentum_balance():
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
    p = np.zeros(7)
    # H/A=[1,0,0], D=1/4, desired U=[1/2,0,0] implies grad(p)=[2,0,0].
    U_star = np.zeros((7, 3))
    U_star[0] = [1.0, 0.0, 0.0]
    U_star[1:] = [0.5, 0.0, 0.0]
    DU = np.full((1, 3), 0.25)

    grad = _update_fixed_flux_pressure_boundaries(p, U_star, DU, mesh, geo, mesh["boundary"])

    np.testing.assert_allclose(p[1:], [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(grad[0], [2.0, 0.0, 0.0], atol=1e-14)
    # Re-evaluating the public gradient operator must see the same trace.
    got = gradients.compute_gradient_gauss_linear_vectorized(p, mesh, geo).squeeze(-1)
    np.testing.assert_allclose(got[0], [2.0, 0.0, 0.0], atol=1e-14)
