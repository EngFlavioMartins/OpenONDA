import numpy as np

from source.solvers.FVM.fields.diagnostics import compute_enstrophy, compute_kinetic_energy
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


def test_uniform_flow_energy_and_enstrophy(hand_built_3d_mesh):
    mesh = hand_built_3d_mesh
    geo = compute_mesh_geometry(mesh, gradient_scheme="lsq")
    n_total = mesh["n_elements"] + mesh["n_faces"] - mesh["n_interior_faces"]
    velocity = np.tile([2.0, -1.0, 0.5], (n_total, 1))
    volume = np.sum(geo["element_volumes"])

    np.testing.assert_allclose(
        compute_kinetic_energy(velocity, geo, density=3.0),
        0.5 * 3.0 * volume * (2.0**2 + 1.0**2 + 0.5**2),
    )
    assert compute_enstrophy(velocity, mesh, geo) < 1e-24


def test_kinetic_energy_accepts_cellwise_density(hand_built_3d_mesh):
    mesh = hand_built_3d_mesh
    geo = compute_mesh_geometry(mesh)
    n_cells = mesh["n_elements"]
    n_total = n_cells + mesh["n_faces"] - mesh["n_interior_faces"]
    velocity = np.ones((n_total, 3))
    density = np.linspace(1.0, 2.0, n_cells)
    expected = 1.5 * np.sum(density * geo["element_volumes"])

    np.testing.assert_allclose(compute_kinetic_energy(velocity, geo, density), expected)
