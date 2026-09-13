"""Qualify the compact native-curl measurement against analytic and FVM fields."""

import numpy as np
import pytest

from source.solvers.fvm.fields.diagnostics import compute_vorticity
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.native_curl_3d import native_gauss_curl_stencil


@pytest.mark.parametrize("subset", [False, True])
def test_compact_native_curl_preserves_analytic_rotation_on_anisotropic_cells(subset):
    mesh = box_mesh_3d(np.linspace(-2, 2, 17), np.linspace(-2, 2, 9), np.linspace(-2, 2, 17),
                       hole_box=(-0.5, 0.5) * 3, wall_patch_name="cube")
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "fixedValue"
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    points = np.vstack((geometry["cell_centre"], geometry["face_centre"][mesh["n_interior_faces"]:]))
    rotation = np.array([0.4, -0.7, 0.3])
    values = np.cross(rotation, points) + points * [1, 2, -3] + [0.7, 0.2, 0.6]
    selected = np.arange(mesh["n_cells"])[::3] if subset else np.arange(mesh["n_cells"])
    operator, source_ids = native_gauss_curl_stencil(mesh, geometry, selected)
    actual = (operator @ values[source_ids].ravel()).reshape(-1, 3)
    np.testing.assert_allclose(actual, np.tile(2 * rotation, (len(selected), 1)), rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual, compute_vorticity(values, mesh, geometry)[selected], rtol=0, atol=2e-14)
    rng = np.random.default_rng(305)
    values = rng.normal(size=values.shape)
    actual = (operator @ values[source_ids].ravel()).reshape(-1, 3)
    np.testing.assert_allclose(actual, compute_vorticity(values, mesh, geometry)[selected], rtol=0, atol=2e-14)
