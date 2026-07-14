import numpy as np
import pytest

from source.solvers.FVM.fields.gradients import compute_gradient_gauss_linear_vectorized
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


class TestGradientOfLinearField:
    """Gradient of φ = x + y + z on a uniform hex mesh is exactly [1,1,1]."""

    @pytest.fixture(autouse=True)
    def setup(self, hand_built_3d_mesh):
        self.mesh = hand_built_3d_mesh
        self.geo = compute_mesh_geometry(hand_built_3d_mesh)

    def _set_ghost_cells(self, phi, func):
        """Set boundary ghost cells to function value at face centroids."""
        n_elem = self.mesh["n_elements"]
        n_int = self.mesh["n_interior_faces"]
        fc = self.geo["face_centroids"]
        for b in self.mesh["boundary"]:
            start, nf = b["startFace"], b["nFaces"]
            for j in range(nf):
                fi = start + j
                gi = n_elem + (fi - n_int)
                phi[gi] = func(fc[fi])

    def test_gradient_linear_scalar(self):
        cents = self.geo["element_centroids"]
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        phi = np.zeros(n_elem + n_bnd)
        phi[:n_elem] = cents[:, 0] + cents[:, 1] + cents[:, 2]
        self._set_ghost_cells(phi, lambda c: c[0] + c[1] + c[2])
        grad = compute_gradient_gauss_linear_vectorized(phi, self.mesh, self.geo)
        assert grad.ndim == 3 and grad.shape[1] == 3
        g = grad[:n_elem].squeeze()
        assert np.allclose(g, 1.0), f"max error = {np.max(np.abs(g - 1.0)):.2e}"

    def test_gradient_linear_vector(self):
        cents = self.geo["element_centroids"]
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        phi = np.zeros((n_elem + n_bnd, 3))
        phi[:n_elem, 0] = cents[:, 0] + cents[:, 1] + cents[:, 2]
        phi[:n_elem, 1] = 2 * cents[:, 0] - cents[:, 2]
        phi[:n_elem, 2] = cents[:, 1]
        self._set_ghost_cells(phi[:, 0], lambda c: c[0] + c[1] + c[2])
        self._set_ghost_cells(phi[:, 1], lambda c: 2 * c[0] - c[2])
        self._set_ghost_cells(phi[:, 2], lambda c: c[1])
        grad = compute_gradient_gauss_linear_vectorized(phi, self.mesh, self.geo)
        assert grad.ndim == 3 and grad.shape[1] == 3
        assert np.allclose(grad[:n_elem, :, 0], 1.0)
        assert np.allclose(grad[:n_elem, :, 1], [2.0, 0.0, -1.0])
        assert np.allclose(grad[:n_elem, :, 2], [0.0, 1.0, 0.0])

    def test_constant_field_gradient_zero(self):
        """Gradient of φ = 1 should be zero everywhere."""
        n_elem = self.mesh["n_elements"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        phi = np.ones(n_elem + n_bnd)
        grad = compute_gradient_gauss_linear_vectorized(phi, self.mesh, self.geo)
        assert np.allclose(grad[:n_elem], 0.0)
