"""Compiled face accumulation preserves the Gauss gradient in three dimensions."""

import numpy as np
import pytest

from source.solvers.fvm.fields.gradients import compute_gauss_gradient
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d


@pytest.mark.parametrize("components", [1, 3])
def test_compiled_gauss_matches_numpy_on_skewed_mesh(components):
    mesh = box_mesh_3d(np.linspace(-1, 1, 8), np.linspace(-1, 1, 6), np.linspace(-1, 1, 5))
    mesh["vertex_position"] = mesh["vertex_position"] @ np.array(
        [[1, 0.2, -0.1], [0, 1.1, 0.3], [0.1, 0, 0.9]]
    )
    for boundary in mesh["boundary"]:
        boundary["velocity_type"] = "fixedValue"
    geo = compute_mesh_geometry(mesh)
    count = mesh["n_cells"] + mesh["n_faces"] - mesh["n_interior_faces"]
    values = np.random.default_rng(573).normal(size=(count, components))
    if components == 1:
        values = values[:, 0]
    expected = compute_gauss_gradient(values, mesh, geo)
    geo["_operator_backend"] = "numba"
    actual = compute_gauss_gradient(values, mesh, geo)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize("constant_diffusivity", [False, True])
def test_compiled_diffusion_matches_numpy_on_skewed_mesh(constant_diffusivity):
    from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term_interior

    mesh = box_mesh_3d(np.linspace(-1, 1, 8), np.linspace(-1, 1, 6), np.linspace(-1, 1, 5))
    mesh["vertex_position"][:, 0] += 0.3 * mesh["vertex_position"][:, 1]
    geo = compute_mesh_geometry(mesh)
    rng = np.random.default_rng(582)
    count = mesh["n_cells"]
    values, gradient = rng.normal(size=count), rng.normal(size=(count, 3))
    diffusivity = 0.02 if constant_diffusivity else rng.uniform(0.01, 0.05, count)
    expected = assemble_diffusion_term_interior(values, gradient, diffusivity, mesh, geo)
    geo["_operator_backend"] = "numba"
    actual = assemble_diffusion_term_interior(values, gradient, diffusivity, mesh, geo)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=4e-14, atol=4e-14)
