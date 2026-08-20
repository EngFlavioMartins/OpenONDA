import numpy as np
from scipy.sparse import diags

from source.solvers.FVM.assemble.time_integration import (
    advance_euler_explicit,
    assemble_transient_term_euler_implicit,
)
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.solve.equation_solver import ScalarEquationSolver


class TestTimeIntegration:
    """dφ/dt = -λφ with Euler implicit → φ¹ = φ⁰ / (1 + λΔt)."""

    def test_single_step_decay(self, hand_built_3d_mesh):
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]

        lam = 2.0
        time_step_size = 0.1
        face_flux_old = np.ones(n_elem) * 10.0

        # Manually assemble: V·(φ - φ_old)/dt + λ·V·φ = 0
        # → (V/dt)·φ + (λ·V)·φ = (V/dt)·φ_old
        # → A·φ = b
        volumes = geo["cell_volumes"]
        A_diag = volumes / time_step_size + lam * volumes
        b = volumes / time_step_size * face_flux_old

        phi_new = b / A_diag
        expected = face_flux_old / (1.0 + lam * time_step_size)
        assert np.allclose(phi_new, expected), (
            f"max error = {np.max(np.abs(phi_new - expected)):.2e}"
        )

    def test_transient_term_assembly(self, hand_built_3d_mesh):
        """Check assemble_transient_term_euler_implicit produces correct diagonal."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]

        rho = 1.0
        time_step_size = 0.1
        face_flux_old = np.ones(n_elem) * 10.0
        result = assemble_transient_term_euler_implicit(face_flux_old, time_step_size, rho, geo)
        volumes = geo["cell_volumes"]
        expected_ac = rho * volumes / time_step_size
        expected_bc = expected_ac * face_flux_old
        assert np.allclose(result["ac"], expected_ac), "transient term ac mismatch"
        assert np.allclose(result["bc"], expected_bc), "transient term bc mismatch"

    def test_explicit_euler_advances_from_spatial_residual(self):
        """Forward Euler uses b - Aφ; it is not a zero-diagonal linear solve."""
        face_flux_old = np.array([10.0, 4.0])
        volumes = np.array([2.0, 0.5])
        rho = np.array([1.0, 3.0])
        decay_rate = 2.0
        time_step_size = 0.1
        matrix = diags(decay_rate * rho * volumes, format="csr")
        rhs = np.zeros_like(face_flux_old)

        phi_new = advance_euler_explicit(face_flux_old, matrix, rhs, time_step_size, rho, volumes)

        assert np.allclose(phi_new, face_flux_old * (1.0 - decay_rate * time_step_size))

    def test_explicit_diffusion_preserves_uniform_field(self, hand_built_3d_mesh):
        """The public scalar path applies the explicit residual update."""
        mesh = hand_built_3d_mesh
        for boundary in mesh["boundary"]:
            boundary["bc_type"] = "zeroGradient"
        geometry = compute_mesh_geometry(mesh)
        n_total = mesh["n_cells"] + mesh["n_faces"] - mesh["n_interior_faces"]
        face_flux = np.full(n_total, 3.0)
        solver = ScalarEquationSolver(mesh, geometry, mesh["boundary"])

        history = solver.solve_transient_diffusion(
            face_flux,
            gamma=0.2,
            density=1.0,
            time_step_size=0.01,
            n_steps=2,
            time_scheme="euler_explicit",
        )

        assert len(history) == 3
        assert np.allclose(history[-1], 3.0)

    def test_three_step_monotonic_decay(self, hand_built_3d_mesh):
        """Over 3 steps, φ should decay monotonically toward zero."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]

        lam = 1.0
        time_step_size = 0.5
        volumes = geo["cell_volumes"]
        face_flux = np.ones(n_elem) * 10.0

        history = [face_flux.copy()]
        for _ in range(3):
            A_diag = volumes / time_step_size + lam * volumes
            b = volumes / time_step_size * face_flux
            face_flux = b / A_diag
            history.append(face_flux.copy())

        for i in range(1, len(history)):
            assert np.all(history[i] <= history[i - 1] * (1 + 1e-12)), (
                f"step {i}: φ did not decay monotonically"
            )
        assert np.all(history[-1] > 0), "φ became negative (unphysical)"
