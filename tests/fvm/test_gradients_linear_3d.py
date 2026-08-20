import numpy as np
import pytest

from source.solvers.FVM.fields.gradients import compute_gauss_gradient
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


class TestGradientOfLinearField:
    """Gradient of φ = x + y + z on a uniform hex mesh is exactly [1,1,1]."""

    @pytest.fixture(autouse=True)
    def setup(self, hand_built_3d_mesh):
        self.mesh = hand_built_3d_mesh
        self.geo = compute_mesh_geometry(hand_built_3d_mesh)
        for boundary in self.mesh["boundary"]:
            boundary["bc_type"] = "fixedValue"

    def _set_ghost_cells(self, face_flux, func):
        """Set boundary ghost cells to function value at face centroids."""
        n_elem = self.mesh["n_cells"]
        n_int = self.mesh["n_interior_faces"]
        fc = self.geo["face_centroids"]
        for b in self.mesh["boundary"]:
            start, nf = b["start_face"], b["n_faces"]
            for j in range(nf):
                fi = start + j
                gi = n_elem + (fi - n_int)
                face_flux[gi] = func(fc[fi])

    def test_gradient_linear_scalar(self):
        cents = self.geo["cell_centroids"]
        n_elem = self.mesh["n_cells"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        face_flux = np.zeros(n_elem + n_bnd)
        face_flux[:n_elem] = cents[:, 0] + cents[:, 1] + cents[:, 2]
        self._set_ghost_cells(face_flux, lambda c: c[0] + c[1] + c[2])
        grad = compute_gauss_gradient(face_flux, self.mesh, self.geo)
        assert grad.ndim == 3 and grad.shape[1] == 3
        g = grad[:n_elem].squeeze()
        assert np.allclose(g, 1.0), f"max error = {np.max(np.abs(g - 1.0)):.2e}"

    def test_gradient_linear_vector(self):
        cents = self.geo["cell_centroids"]
        n_elem = self.mesh["n_cells"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        face_flux = np.zeros((n_elem + n_bnd, 3))
        face_flux[:n_elem, 0] = cents[:, 0] + cents[:, 1] + cents[:, 2]
        face_flux[:n_elem, 1] = 2 * cents[:, 0] - cents[:, 2]
        face_flux[:n_elem, 2] = cents[:, 1]
        self._set_ghost_cells(face_flux[:, 0], lambda c: c[0] + c[1] + c[2])
        self._set_ghost_cells(face_flux[:, 1], lambda c: 2 * c[0] - c[2])
        self._set_ghost_cells(face_flux[:, 2], lambda c: c[1])
        grad = compute_gauss_gradient(face_flux, self.mesh, self.geo)
        assert grad.ndim == 3 and grad.shape[1] == 3
        assert np.allclose(grad[:n_elem, :, 0], 1.0)
        assert np.allclose(grad[:n_elem, :, 1], [2.0, 0.0, -1.0])
        assert np.allclose(grad[:n_elem, :, 2], [0.0, 1.0, 0.0])

    def test_constant_field_gradient_zero(self):
        """Gradient of φ = 1 should be zero everywhere."""
        n_elem = self.mesh["n_cells"]
        n_bnd = self.mesh["n_faces"] - self.mesh["n_interior_faces"]
        face_flux = np.ones(n_elem + n_bnd)
        grad = compute_gauss_gradient(face_flux, self.mesh, self.geo)
        assert np.allclose(grad[:n_elem], 0.0)
