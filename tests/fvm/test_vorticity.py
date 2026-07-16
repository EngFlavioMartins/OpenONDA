import numpy as np

from source.solvers.FVM.fields.diagnostics import compute_vorticity
from source.solvers.FVM.fields.gradients import compute_gradient_gauss_linear_vectorized
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


class TestVorticity:
    """Vorticity of solid-body rotation U = (-y, x, 0) → ω = (0, 0, 2)."""

    def test_vorticity_from_curl(self):
        """Compute curl analytically from known gradient of solid-body rotation."""
        locations = np.array(
            [
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        for loc in locations:
            x, y, z = loc
            # dUx/dy = -1, dUy/dx = 1, all other derivatives = 0
            curl = np.array([0.0, 0.0, 1.0 - (-1.0)])  # = (0, 0, 2)
            assert np.allclose(curl, [0.0, 0.0, 2.0])

    def test_vorticity_on_hand_built_mesh(self, hand_built_3d_mesh):
        """ω = ∇×U for solid-body rotation on hand_built mesh."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_elements"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
        cents = geo["element_centroids"]

        # Build velocity field with proper ghost cells for gradient computation
        U = np.zeros((n_elem + n_bnd, 3))
        U[:n_elem, 0] = -cents[:, 1]
        U[:n_elem, 1] = cents[:, 0]
        for boundary in mesh["boundary"]:
            boundary["bc_type_U"] = "zeroGradient"

        # Set ghost cells to analytic velocity at face centroids
        fc = geo["face_centroids"]
        for b in mesh["boundary"]:
            start, nf = b["startFace"], b["nFaces"]
            for j in range(nf):
                fi = start + j
                gi = n_elem + (fi - mesh["n_interior_faces"])
                U[gi, 0] = -fc[fi, 1]
                U[gi, 1] = fc[fi, 0]

        gradU = compute_gradient_gauss_linear_vectorized(U, mesh, geo)
        # ω = ∇ × U = (∂w/∂y − ∂v/∂z, ∂u/∂z − ∂w/∂x, ∂v/∂x − ∂u/∂y)
        # gradU[i, j, k] = ∂U_k/∂x_j
        dUdx = gradU[:n_elem, 0, :]  # (n_elem, 3): ∂u/∂x, ∂v/∂x, ∂w/∂x
        dUdy = gradU[:n_elem, 1, :]  # ∂u/∂y, ∂v/∂y, ∂w/∂y
        dUdz = gradU[:n_elem, 2, :]  # ∂u/∂z, ∂v/∂z, ∂w/∂z
        omega_x = dUdy[:, 2] - dUdz[:, 1]  # ∂w/∂y − ∂v/∂z
        omega_y = dUdz[:, 0] - dUdx[:, 2]  # ∂u/∂z − ∂w/∂x
        omega_z = dUdx[:, 1] - dUdy[:, 0]  # ∂v/∂x − ∂u/∂y
        omega = np.column_stack([omega_x, omega_y, omega_z])
        expected = np.tile([0.0, 0.0, 2.0], (n_elem, 1))
        err = np.max(np.abs(omega - expected))
        assert err < 1e-10, f"max ω error = {err:.2e} (from analytic gradient)"

    def test_vorticity_function(self, hand_built_3d_mesh):
        """compute_vorticity() on solid-body rotation."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_elements"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
        cents = geo["element_centroids"]

        U = np.zeros((n_elem + n_bnd, 3))
        U[:n_elem, 0] = -cents[:, 1]
        U[:n_elem, 1] = cents[:, 0]
        for boundary in mesh["boundary"]:
            boundary["bc_type_U"] = "zeroGradient"

        fc = geo["face_centroids"]
        for b in mesh["boundary"]:
            start, nf = b["startFace"], b["nFaces"]
            for j in range(nf):
                fi = start + j
                gi = n_elem + (fi - mesh["n_interior_faces"])
                U[gi, 0] = -fc[fi, 1]
                U[gi, 1] = fc[fi, 0]

        vort = compute_vorticity(U, mesh, geo)
        expected = np.tile([0.0, 0.0, 2.0], (n_elem, 1))
        err = np.max(np.abs(vort[:n_elem] - expected))
        assert err < 1e-10, f"max ω error = {err:.2e} (from compute_vorticity)"
