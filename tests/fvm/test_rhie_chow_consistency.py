import numpy as np

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.solve.simple_solver import (
    _compute_pressure_face_conductance,
    _compute_rhie_chow_coefficients,
    assemble_pressure_correction_equation_rhie_chow,
    correct_velocity_and_flux,
    update_scalar_boundaries,
)


class TestRhieChowConsistency:
    """physical_momentum_diagonal = momentum_diagonal * alpha_u is used identically in assembly and correction."""

    def test_du_coefficient_identity(self, hand_built_3d_mesh):
        """Both assembly and correction produce the same pressure_velocity_coefficient for given momentum_diagonal and alpha_u."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        for b in mesh["boundary"]:
            b["boundary_condition_type"] = "zeroGradient"
            b["velocity_type"] = "zeroGradient"
            b["pressure_type"] = "zeroGradient"

        n_elem = mesh["n_cells"]
        volumes = geo["cell_volume"]
        momentum_diagonal = np.ones((n_elem, 3)) * 10.0  # per-component diagonal
        velocity_relaxation = 0.7

        # Expected DUs (per component)
        A_U_phys = momentum_diagonal * velocity_relaxation  # (n_elem, 3)
        expected_pressure_velocity_coefficient = volumes[:, np.newaxis] / A_U_phys  # (n_elem, 3)

        # From assembly (uses momentum_diagonal * alpha_u internally)
        velocity = np.zeros((n_elem + mesh["n_faces"] - mesh["n_interior_faces"], 3))
        p = np.zeros(n_elem + mesh["n_faces"] - mesh["n_interior_faces"])
        pressure_matrix, pressure_right_hand_side, phi_star = (
            assemble_pressure_correction_equation_rhie_chow(
                velocity,
                momentum_diagonal,
                p,
                1.0,
                mesh,
                geo,
                mesh["boundary"],
                velocity_relaxation=velocity_relaxation,
            )
        )

        # From correction (uses momentum_diagonal * alpha_u internally)
        p_prime = np.zeros(n_elem)
        U_corr, phi_corr = correct_velocity_and_flux(
            velocity.copy(),
            phi_star.copy(),
            p_prime,
            momentum_diagonal,
            mesh,
            geo,
            mesh["boundary"],
            density=1.0,
            velocity_relaxation=velocity_relaxation,
        )

        # Direct computation
        direct_pressure_velocity_coefficient = _compute_rhie_chow_coefficients(volumes, A_U_phys)
        assert np.allclose(
            expected_pressure_velocity_coefficient, direct_pressure_velocity_coefficient
        ), "expected pressure_velocity_coefficient mismatch"

    def test_du_differs_with_alpha(self, hand_built_3d_mesh):
        """pressure_velocity_coefficient(alpha_u=0.5) should be 2× pressure_velocity_coefficient(alpha_u=1.0)."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        volumes = geo["cell_volume"]
        momentum_diagonal = np.ones((mesh["n_cells"], 3)) * 10.0

        pressure_velocity_coefficient_full_relaxation = _compute_rhie_chow_coefficients(
            volumes, momentum_diagonal * 1.0
        )
        pressure_velocity_coefficient_half_relaxation = _compute_rhie_chow_coefficients(
            volumes, momentum_diagonal * 0.5
        )
        assert np.allclose(
            pressure_velocity_coefficient_half_relaxation,
            2.0 * pressure_velocity_coefficient_full_relaxation,
        )

    def test_pressure_solve_removes_flux_defect_on_skewed_mesh(self):
        """Assembly and correction must conserve with the same non-orthogonal metric."""
        from scipy.sparse.linalg import spsolve

        from source.solvers.fvm.fields.diagnostics import compute_continuity_error

        from ._structured_mesh import structured_box

        mesh = structured_box(3, 3, 2)
        # Affine shear makes Sf non-parallel to the owner-neighbour vector while
        # preserving a valid mesh and exact cell connectivity.
        mesh["vertex_position"][:, 0] += 0.35 * mesh["vertex_position"][:, 1]
        geo = compute_mesh_geometry(mesh)

        for patch in mesh["boundary"]:
            patch["velocity_type"] = "zeroGradient"
            patch["pressure_type"] = "fixedValue" if patch["name"] == "xmax" else "zeroGradient"
            patch["kinematic_pressure_value"] = 0.0

        n = mesh["n_cells"]
        nb = mesh["n_faces"] - mesh["n_interior_faces"]
        rng = np.random.default_rng(17)
        velocity = rng.normal(scale=0.1, size=(n + nb, 3))
        p = rng.normal(scale=0.05, size=n + nb)
        update_scalar_boundaries(p, mesh, mesh["boundary"], field_name="kinematic_pressure")
        momentum_diagonal = np.full((n, 3), 4.0)

        pressure_matrix, pressure_right_hand_side, phi_star = (
            assemble_pressure_correction_equation_rhie_chow(
                velocity, momentum_diagonal, p, 1.0, mesh, geo, mesh["boundary"]
            )
        )
        p_prime = spsolve(pressure_matrix, pressure_right_hand_side)
        _, volumetric_face_flux = correct_velocity_and_flux(
            velocity.copy(),
            phi_star.copy(),
            p_prime,
            momentum_diagonal,
            mesh,
            geo,
            mesh["boundary"],
        )

        linear_residual = np.linalg.norm(pressure_matrix @ p_prime - pressure_right_hand_side)
        continuity = compute_continuity_error(volumetric_face_flux, mesh, geo)
        assert linear_residual < 1e-11
        assert np.max(np.abs(continuity)) < 2e-11

    def test_face_conductance_is_positive_on_skewed_mesh(self):
        from ._structured_mesh import structured_box

        mesh = structured_box(2, 2, 2)
        mesh["vertex_position"][:, 0] += 0.25 * mesh["vertex_position"][:, 1]
        geo = compute_mesh_geometry(mesh)
        pressure_velocity_coefficient = np.ones((mesh["n_cells"], 3))
        conductance = _compute_pressure_face_conductance(mesh, geo, pressure_velocity_coefficient)
        assert conductance.shape == (mesh["n_faces"],)
        assert np.all(conductance > 0.0)

    def test_assembly_workspace_reuses_exact_conductance_for_correction(
        self, hand_built_3d_mesh, monkeypatch
    ):
        """The production path must not recalculate Rhie--Chow conductance."""
        import source.solvers.fvm.solve.simple_solver as simple

        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        for boundary in mesh["boundary"]:
            boundary["velocity_type"] = "zeroGradient"
            boundary["pressure_type"] = "zeroGradient"
        n = mesh["n_cells"]
        nb = mesh["n_faces"] - mesh["n_interior_faces"]
        velocity = np.zeros((n + nb, 3))
        p = np.zeros(n + nb)
        momentum_diagonal = np.full((n, 3), 2.0)
        calls = 0
        original = simple._compute_pressure_face_conductance

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(simple, "_compute_pressure_face_conductance", counted)
        _A, _b, volumetric_face_flux, workspace = (
            simple.assemble_pressure_correction_equation_rhie_chow(
                velocity,
                momentum_diagonal,
                p,
                1.0,
                mesh,
                geo,
                mesh["boundary"],
                return_workspace=True,
            )
        )
        simple.correct_velocity_and_flux(
            velocity.copy(),
            np.array(volumetric_face_flux, copy=True),
            np.zeros(n),
            momentum_diagonal,
            mesh,
            geo,
            mesh["boundary"],
            workspace=workspace,
        )
        assert calls == 1
