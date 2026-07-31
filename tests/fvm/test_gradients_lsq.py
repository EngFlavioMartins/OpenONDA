"""Tests for least-squares gradient computation."""

import numpy as np
import pytest

from source.solvers.FVM.fields.gradients import (
    compute_lsq_gradient,
)
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


def _set_ghost_cells(phi, mesh, geo, func):
    """Set boundary ghost cells to function value at face centroids."""
    n_elem = mesh["n_elements"]
    n_int = mesh["n_interior_faces"]
    fc = geo["face_centroids"]
    for b in mesh["boundary"]:
        start, nf = b["startFace"], b["nFaces"]
        for j in range(nf):
            fi = start + j
            gi = n_elem + (fi - n_int)
            phi[gi] = func(fc[fi])


class TestLSQOnHexMesh:
    """LSQ gradient on a uniform hex mesh — should match Gauss for linear fields."""

    @pytest.fixture(autouse=True)
    def setup(self, hand_built_3d_mesh):
        self.mesh = hand_built_3d_mesh
        self.geo = compute_mesh_geometry(hand_built_3d_mesh, gradient_scheme="lsq")

    def test_linear_scalar(self):
        cents = self.geo["element_centroids"]
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        phi = np.zeros(n_elem + n_bnd)
        phi[:n_elem] = cents[:, 0] + cents[:, 1] + cents[:, 2]
        _set_ghost_cells(phi, self.mesh, self.geo, lambda c: c[0] + c[1] + c[2])
        grad = compute_lsq_gradient(phi, self.mesh, self.geo)
        g = grad[:n_elem].squeeze()
        assert np.allclose(g, 1.0), f"max error = {np.max(np.abs(g - 1.0)):.2e}"

    def test_linear_vector(self):
        cents = self.geo["element_centroids"]
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        phi = np.zeros((n_elem + n_bnd, 3))
        phi[:n_elem, 0] = cents[:, 0] + cents[:, 1] + cents[:, 2]
        phi[:n_elem, 1] = 2 * cents[:, 0] - cents[:, 2]
        phi[:n_elem, 2] = cents[:, 1]
        _set_ghost_cells(phi[:, 0], self.mesh, self.geo, lambda c: c[0] + c[1] + c[2])
        _set_ghost_cells(phi[:, 1], self.mesh, self.geo, lambda c: 2 * c[0] - c[2])
        _set_ghost_cells(phi[:, 2], self.mesh, self.geo, lambda c: c[1])
        grad = compute_lsq_gradient(phi, self.mesh, self.geo)
        assert np.allclose(grad[:n_elem, :, 0], 1.0)
        assert np.allclose(grad[:n_elem, :, 1], [2.0, 0.0, -1.0])
        assert np.allclose(grad[:n_elem, :, 2], [0.0, 1.0, 0.0])

    def test_constant_zero(self):
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        phi = np.ones(n_elem + n_bnd)
        grad = compute_lsq_gradient(phi, self.mesh, self.geo)
        assert np.allclose(grad[:n_elem], 0.0)

    def test_well_conditioned_3d_stencils_select_qr(self):
        assert np.all(self.geo["lsq_rank"] == 3)
        assert np.all(self.geo["lsq_solver_method"] == "qr")


class TestLSQOnTetMesh:
    """LSQ gradient on tetrahedral meshes — where Gauss has O(1) error."""

    @pytest.fixture(autouse=True)
    def setup(self, gmsh_unit_cube):
        self.mesh = gmsh_unit_cube
        self.geo = compute_mesh_geometry(gmsh_unit_cube, gradient_scheme="lsq")

    def test_linear_scalar_exact(self):
        """LSQ recovers exact gradient of linear field even on tets."""
        cents = self.geo["element_centroids"]
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        phi = np.zeros(n_elem + n_bnd)
        phi[:n_elem] = cents[:, 0] + cents[:, 1] + cents[:, 2]
        _set_ghost_cells(phi, self.mesh, self.geo, lambda c: c[0] + c[1] + c[2])
        grad = compute_lsq_gradient(phi, self.mesh, self.geo)
        g = grad[:n_elem].squeeze()
        assert np.allclose(g, 1.0, atol=1e-10), f"max error = {np.max(np.abs(g - 1.0)):.2e}"

    def test_linear_vector_exact(self):
        cents = self.geo["element_centroids"]
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        phi = np.zeros((n_elem + n_bnd, 3))
        phi[:n_elem, 0] = cents[:, 0] + cents[:, 1] + cents[:, 2]
        phi[:n_elem, 1] = 2 * cents[:, 0] - cents[:, 2]
        phi[:n_elem, 2] = cents[:, 1]
        _set_ghost_cells(phi[:, 0], self.mesh, self.geo, lambda c: c[0] + c[1] + c[2])
        _set_ghost_cells(phi[:, 1], self.mesh, self.geo, lambda c: 2 * c[0] - c[2])
        _set_ghost_cells(phi[:, 2], self.mesh, self.geo, lambda c: c[1])
        grad = compute_lsq_gradient(phi, self.mesh, self.geo)
        assert np.allclose(grad[:n_elem, :, 0], 1.0, atol=1e-10)
        assert np.allclose(grad[:n_elem, :, 1], [2.0, 0.0, -1.0], atol=1e-10)
        assert np.allclose(grad[:n_elem, :, 2], [0.0, 1.0, 0.0], atol=1e-10)

    def test_constant_zero(self):
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        phi = np.ones(n_elem + n_bnd)
        grad = compute_lsq_gradient(phi, self.mesh, self.geo)
        assert np.allclose(grad[:n_elem], 0.0, atol=1e-10)

    def test_solver_selection_is_reported(self):
        methods = set(self.geo["lsq_solver_method"])
        assert methods <= {"qr", "svd"}
        assert methods
