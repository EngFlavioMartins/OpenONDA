"""Check the production Gauss curl against analytic rotation on anisotropic cells."""

import numpy as np
import pytest

from source.solvers.fvm.fields.diagnostics import compute_vorticity
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d


@pytest.mark.parametrize("stretch", [1.0, 2.0])
def test_gauss_curl_preserves_analytic_rotation_on_anisotropic_cells(stretch):
    mesh = box_mesh_3d(
        np.linspace(-2, 2, 17),
        np.linspace(-2, 2, int(8 * stretch) + 1),
        np.linspace(-2, 2, 17),
        hole_box=(-0.5, 0.5) * 3,
        wall_patch_name="cube",
    )
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "fixedValue"
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    points = np.vstack(
        (geometry["cell_centre"], geometry["face_centre"][mesh["n_interior_faces"] :])
    )
    rotation = np.array([0.4, -0.7, 0.3])
    values = np.cross(rotation, points) + points * [1, 2, -3] + [0.7, 0.2, 0.6]
    actual = compute_vorticity(values, mesh, geometry)
    np.testing.assert_allclose(
        actual, np.tile(2 * rotation, (mesh["n_cells"], 1)), rtol=0, atol=2e-14
    )
