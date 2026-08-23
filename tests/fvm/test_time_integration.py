import numpy as np
from scipy.sparse import diags

from source.solvers.fvm.assemble.time_integration import (
    advance_euler_explicit,
    assemble_transient_term_euler_implicit,
)
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.solve.equation_solver import ScalarEquationSolver


class TestTimeIntegration:
    """dφ/dt = -λφ with Euler implicit → φ¹ = φ⁰ / (1 + λΔt)."""

    def test_single_step_decay(self, hand_built_3d_mesh):
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]

        lam = 2.0
        time_step_size = 0.1
        scalar_field_old = np.ones(n_elem) * 10.0

        # Manually assemble: V·(φ - φ_old)/dt + λ·V·φ = 0
        # → (V/dt)·φ + (λ·V)·φ = (V/dt)·φ_old
        # → A·φ = b
        cell_volume = geo["cell_volume"]
        A_diag = cell_volume / time_step_size + lam * cell_volume
        b = cell_volume / time_step_size * scalar_field_old

        scalar_field_new = b / A_diag
        expected = scalar_field_old / (1.0 + lam * time_step_size)
        assert np.allclose(scalar_field_new, expected), (
            f"max error = {np.max(np.abs(scalar_field_new - expected)):.2e}"
        )

    def test_transient_term_assembly(self, hand_built_3d_mesh):
        """Check assemble_transient_term_euler_implicit produces correct diagonal."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]

        density = 1.0
        time_step_size = 0.1
        scalar_field_old = np.ones(n_elem) * 10.0
        result = assemble_transient_term_euler_implicit(
            scalar_field_old, time_step_size, density, geo
        )
        cell_volume = geo["cell_volume"]
        expected_ac = density * cell_volume / time_step_size
        expected_bc = expected_ac * scalar_field_old
        assert np.allclose(result["ac"], expected_ac), "transient term ac mismatch"
        assert np.allclose(result["bc"], expected_bc), "transient term bc mismatch"

    def test_explicit_euler_advances_from_spatial_residual(self):
        """Forward Euler uses b - Aφ; it is not a zero-diagonal linear solve."""
        scalar_field_old = np.array([10.0, 4.0])
        cell_volume = np.array([2.0, 0.5])
        density = np.array([1.0, 3.0])
        decay_rate = 2.0
        time_step_size = 0.1
        matrix = diags(decay_rate * density * cell_volume, format="csr")
        rhs = np.zeros_like(scalar_field_old)

        scalar_field_new = advance_euler_explicit(
            scalar_field_old, matrix, rhs, time_step_size, density, cell_volume
        )

        assert np.allclose(
            scalar_field_new,
            scalar_field_old * (1.0 - decay_rate * time_step_size),
        )

    def test_explicit_diffusion_preserves_uniform_field(self, hand_built_3d_mesh):
        """The public scalar path applies the explicit residual update."""
        mesh = hand_built_3d_mesh
        for boundary in mesh["boundary"]:
            boundary["boundary_condition_type"] = "zeroGradient"
        geometry = compute_mesh_geometry(mesh)
        n_total = mesh["n_cells"] + mesh["n_faces"] - mesh["n_interior_faces"]
        scalar_field = np.full(n_total, 3.0)
        solver = ScalarEquationSolver(mesh, geometry, mesh["boundary"])

        history = solver.solve_transient_diffusion(
            scalar_field,
            diffusivity=0.2,
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
        cell_volume = geo["cell_volume"]
        scalar_field = np.ones(n_elem) * 10.0

        history = [scalar_field.copy()]
        for _ in range(3):
            A_diag = cell_volume / time_step_size + lam * cell_volume
            b = cell_volume / time_step_size * scalar_field
            scalar_field = b / A_diag
            history.append(scalar_field.copy())

        for i in range(1, len(history)):
            assert np.all(history[i] <= history[i - 1] * (1 + 1e-12)), (
                f"step {i}: φ did not decay monotonically"
            )
        assert np.all(history[-1] > 0), "φ became negative (unphysical)"
