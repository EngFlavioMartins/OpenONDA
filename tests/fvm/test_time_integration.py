import numpy as np
import pytest

from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.assemble.time_integration import assemble_transient_term_euler_implicit


class TestTimeIntegration:
    """dφ/dt = -λφ with Euler implicit → φ¹ = φ⁰ / (1 + λΔt)."""

    def test_single_step_decay(self, hand_built_3d_mesh):
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_elements"]

        rho = 1.0
        lam = 2.0
        dt = 0.1
        phi_old = np.ones(n_elem) * 10.0

        # Manually assemble: V·(φ - φ_old)/dt + λ·V·φ = 0
        # → (V/dt)·φ + (λ·V)·φ = (V/dt)·φ_old
        # → A·φ = b
        volumes = geo["element_volumes"]
        A_diag = volumes / dt + lam * volumes
        b = volumes / dt * phi_old

        phi_new = b / A_diag
        expected = phi_old / (1.0 + lam * dt)
        assert np.allclose(phi_new, expected), (
            f"max error = {np.max(np.abs(phi_new - expected)):.2e}"
        )

    def test_transient_term_assembly(self, hand_built_3d_mesh):
        """Check assemble_transient_term_euler_implicit produces correct diagonal."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_elements"]

        rho = 1.0
        dt = 0.1
        phi_old = np.ones(n_elem) * 10.0
        phi = np.ones(n_elem) * 8.0

        result = assemble_transient_term_euler_implicit(phi, phi_old, dt, rho, mesh, geo)
        volumes = geo["element_volumes"]
        expected_ac = rho * volumes / dt
        expected_bc = expected_ac * phi_old
        assert np.allclose(result["ac"], expected_ac), "transient term ac mismatch"
        assert np.allclose(result["bc"], expected_bc), "transient term bc mismatch"

    def test_three_step_monotonic_decay(self, hand_built_3d_mesh):
        """Over 3 steps, φ should decay monotonically toward zero."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_elements"]

        lam = 1.0
        dt = 0.5
        volumes = geo["element_volumes"]
        phi = np.ones(n_elem) * 10.0

        history = [phi.copy()]
        for _ in range(3):
            A_diag = volumes / dt + lam * volumes
            b = volumes / dt * phi
            phi = b / A_diag
            history.append(phi.copy())

        for i in range(1, len(history)):
            assert np.all(history[i] <= history[i - 1] * (1 + 1e-12)), (
                f"step {i}: φ did not decay monotonically"
            )
        assert np.all(history[-1] > 0), "φ became negative (unphysical)"
