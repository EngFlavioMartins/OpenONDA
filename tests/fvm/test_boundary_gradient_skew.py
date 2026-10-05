"""Boundary gradients and deviatoric stress must preserve affine fields."""

import numpy as np
import pytest

from source.solvers.fvm.assemble.momentum import compute_dev2_stress_source
from source.solvers.fvm.fields.gradients import _resolve_gradient_fn
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from tests.support.fvm_mesh import structured_box


@pytest.mark.parametrize("scheme", ["gauss", "lsq"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("components", [1, 3])
def test_native_skew_boundary_gradient_preserves_affine_tensor(scheme, backend, components):
    mesh = structured_box(4, 3, 3)
    mesh["vertex_position"] = (
        mesh["vertex_position"] @ np.array([[1.0, 0.3, -0.2], [0.1, 1.2, 0.4], [0.2, -0.1, 0.8]]).T
    )
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "normalValueTangentialGradient"
    geometry = compute_mesh_geometry(mesh, gradient_scheme=scheme)
    geometry["_operator_backend"] = backend
    n_cells, n_interior = mesh["n_cells"], mesh["n_interior_faces"]
    # Native convention: first tensor axis is derivative direction.
    exact = np.array([[0.2, 0.6, -0.8], [-0.7, -0.1, 1.2], [1.3, -0.4, -0.1]])
    exact = exact[:, :components]
    locations = np.concatenate((geometry["cell_centre"], geometry["face_centre"][n_interior:]))
    values = locations @ exact + np.arange(components)[None, :] * 0.3
    gradient = _resolve_gradient_fn(geometry)(
        values[:, 0] if components == 1 else values, mesh, geometry
    )

    np.testing.assert_allclose(gradient, np.broadcast_to(exact, gradient.shape), rtol=0, atol=8e-14)
    if components == 3:
        # Constant affine incompressible flow has spatially constant stress.
        # This checks its divergence through the actual momentum source,
        # which consumes the reconstructed boundary tensors.
        source = compute_dev2_stress_source(gradient, 0.037, mesh, geometry)
        assert source.shape == (n_cells, 3)
        np.testing.assert_allclose(source, 0.0, rtol=0, atol=8e-14)
