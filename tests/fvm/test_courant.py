import numpy as np

from source.solvers.fvm.assemble.convection import compute_volumetric_face_flux
from source.solvers.fvm.fields.diagnostics import compute_courant_number
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry


class TestCourant:
    """Courant number for uniform flow on a uniform hex mesh."""

    def test_uniform_courant_on_hand_built(self, hand_built_3d_mesh):
        """U=(1,0,0), dt=0.1 on mesh with dx=1 → Co ≈ 0.1."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]

        velocity = np.tile([1.0, 0.0, 0.0], (n_elem + n_bnd, 1))
        volumetric_face_flux = compute_volumetric_face_flux(velocity, mesh, geo)
        time_step_size = 0.1

        co = compute_courant_number(velocity, volumetric_face_flux, time_step_size, mesh, geo)
        assert co.shape[0] >= n_elem
        co_int = co[:n_elem]
        # On uniform mesh with dx=1, U=1, Co ≈ 0.1
        assert np.allclose(co_int, 0.1, atol=0.02), (
            f"Co range = [{co_int.min():.4f}, {co_int.max():.4f}]"
        )

    def test_courant_scales_with_velocity(self, hand_built_3d_mesh):
        """Doubling U should double Co."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]

        U1 = np.tile([1.0, 0.0, 0.0], (n_elem + n_bnd, 1))
        U2 = np.tile([2.0, 0.0, 0.0], (n_elem + n_bnd, 1))
        phi1 = compute_volumetric_face_flux(U1, mesh, geo)
        phi2 = compute_volumetric_face_flux(U2, mesh, geo)
        time_step_size = 0.1

        co1 = compute_courant_number(U1, phi1, time_step_size, mesh, geo)[:n_elem]
        co2 = compute_courant_number(U2, phi2, time_step_size, mesh, geo)[:n_elem]
        assert np.allclose(co2, 2.0 * co1, atol=1e-12), "Co should scale with velocity"
