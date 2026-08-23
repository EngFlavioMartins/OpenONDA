import numpy as np

from source.solvers.fvm.fields.diagnostics import compute_vorticity
from source.solvers.fvm.fields.gradients import compute_gauss_gradient
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry


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
        n_elem = mesh["n_cells"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
        cents = geo["cell_centre"]

        # Build velocity field with proper ghost cells for gradient computation
        velocity = np.zeros((n_elem + n_bnd, 3))
        velocity[:n_elem, 0] = -cents[:, 1]
        velocity[:n_elem, 1] = cents[:, 0]
        for boundary in mesh["boundary"]:
            boundary["velocity_type"] = "zeroGradient"

        # Set ghost cells to analytic velocity at face centroids
        fc = geo["face_centre"]
        for b in mesh["boundary"]:
            start, nf = b["start_face"], b["n_faces"]
            for j in range(nf):
                fi = start + j
                gi = n_elem + (fi - mesh["n_interior_faces"])
                velocity[gi, 0] = -fc[fi, 1]
                velocity[gi, 1] = fc[fi, 0]

        velocity_gradient = compute_gauss_gradient(velocity, mesh, geo)
        # vorticity = curl(velocity)
        velocity_derivative_x = velocity_gradient[:n_elem, 0, :]
        velocity_derivative_y = velocity_gradient[:n_elem, 1, :]
        velocity_derivative_z = velocity_gradient[:n_elem, 2, :]
        vorticity_x = velocity_derivative_y[:, 2] - velocity_derivative_z[:, 1]
        vorticity_y = velocity_derivative_z[:, 0] - velocity_derivative_x[:, 2]
        vorticity_z = velocity_derivative_x[:, 1] - velocity_derivative_y[:, 0]
        vorticity = np.column_stack([vorticity_x, vorticity_y, vorticity_z])
        expected = np.tile([0.0, 0.0, 2.0], (n_elem, 1))
        max_vorticity_error = np.max(np.abs(vorticity - expected))
        assert max_vorticity_error < 1e-10, (
            f"max vorticity error = {max_vorticity_error:.2e} (from analytic gradient)"
        )

    def test_vorticity_function(self, hand_built_3d_mesh):
        """compute_vorticity() on solid-body rotation."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
        cents = geo["cell_centre"]

        velocity = np.zeros((n_elem + n_bnd, 3))
        velocity[:n_elem, 0] = -cents[:, 1]
        velocity[:n_elem, 1] = cents[:, 0]
        for boundary in mesh["boundary"]:
            boundary["velocity_type"] = "zeroGradient"

        fc = geo["face_centre"]
        for b in mesh["boundary"]:
            start, nf = b["start_face"], b["n_faces"]
            for j in range(nf):
                fi = start + j
                gi = n_elem + (fi - mesh["n_interior_faces"])
                velocity[gi, 0] = -fc[fi, 1]
                velocity[gi, 1] = fc[fi, 0]

        vort = compute_vorticity(velocity, mesh, geo)
        expected = np.tile([0.0, 0.0, 2.0], (n_elem, 1))
        err = np.max(np.abs(vort[:n_elem] - expected))
        assert err < 1e-10, f"max ω error = {err:.2e} (from compute_vorticity)"
