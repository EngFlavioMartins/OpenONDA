"""Pressure null-space and reference-cell behavior."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from source.solvers.FVM.solve import linear_interface
from source.solvers.FVM.solve.linear_interface import normalized_residual, solve_linear_system
from source.solvers.FVM.solve.simple_solver import (
    _resolve_pressure_constraint,
    assemble_pressure_correction_equation_rhie_chow,
)
from source.solvers.FVM.utils.cavity_utils import (
    fix_pressure_reference,
    needs_pressure_reference,
)


def test_reference_needed_only_without_dirichlet_pressure():
    closed = [{"name": "walls", "nFaces": 100, "bc_type_p": "zeroGradient"}]
    assert needs_pressure_reference(closed)

    # A single Dirichlet patch removes the constant pressure null space,
    # irrespective of its face-count-to-cell-count ratio.
    open_case = closed + [
        {"name": "outlet", "nFaces": 1, "bc_type_p": "fixedValue", "value_p": 0.0}
    ]
    assert not needs_pressure_reference(open_case)


def test_reference_constraint_preserves_symmetry_and_other_equations():
    A = csr_matrix(
        np.array(
            [
                [2.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 2.0],
            ]
        )
    )
    b = np.array([1.0, 0.0, 1.0])
    A_fixed, b_fixed = fix_pressure_reference(A, b, ref_element=0, ref_value=2.0)

    assert np.allclose(A_fixed.toarray(), A_fixed.toarray().T)
    x = spsolve(A_fixed, b_fixed)
    assert np.isclose(x[0], 2.0)
    assert np.allclose((A @ x - b)[1:], 0.0)
    # Inputs are not mutated, which makes repeated correction assembly safe.
    assert np.allclose(A.toarray()[0], [2.0, -1.0, 0.0])
    assert np.allclose(b, [1.0, 0.0, 1.0])


def test_nullspace_pressure_assembly_remains_singular_and_compatible(
    hand_built_3d_mesh,
):
    from source.solvers.FVM.mesh.geometry import compute_mesh_geometry

    mesh = hand_built_3d_mesh
    geo = compute_mesh_geometry(mesh)
    for boundary in mesh["boundary"]:
        boundary["bc_type_velocity"] = "zeroGradient"
        boundary["bc_type_p"] = "zeroGradient"
    n = mesh["n_elements"]
    n_total = n + mesh["n_faces"] - mesh["n_interior_faces"]
    rng = np.random.default_rng(9)
    velocity = rng.normal(scale=0.1, size=(n_total, 3))
    pressure = np.zeros(n_total)
    momentum_diagonal = np.full((n, 3), 2.0)

    matrix, rhs, _ = assemble_pressure_correction_equation_rhie_chow(
        velocity,
        momentum_diagonal,
        pressure,
        1.0,
        mesh,
        geo,
        mesh["boundary"],
        pressure_constraint="nullspace",
    )

    np.testing.assert_allclose(matrix @ np.ones(n), 0.0, atol=1e-13)
    assert np.sum(rhs) == pytest.approx(0.0, abs=1e-13)
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T, atol=1e-13)


def test_pressure_constraint_auto_selects_backend_protocol():
    assert _resolve_pressure_constraint({"_linear_backend": "scipy"}) == "reference"
    assert _resolve_pressure_constraint({"_linear_backend": "petsc"}) == "nullspace"
    with pytest.raises(ValueError, match="requires backend='petsc'"):
        _resolve_pressure_constraint(
            {"_linear_backend": "scipy", "pressure_nullspace_policy": "petsc"}
        )


def test_pressure_iterative_path_solves_current_matrix():
    A = csr_matrix(
        np.array(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 3.0],
            ]
        )
    )
    b = np.array([1.0, 2.0, 3.0])
    x = solve_linear_system(
        A,
        b,
        method="bicgstab",
        equation_type="pressure",
        tol=1e-11,
        maxiter=100,
        amg_tol=1e-11,
        amg_maxiter=100,
    )
    assert normalized_residual(A, x, b) < 1e-9


def test_strict_iterative_policy_does_not_hide_failure(monkeypatch):
    def fail_solve(A, b, **kwargs):
        return np.zeros_like(b), 1

    monkeypatch.setattr(linear_interface, "bicgstab", fail_solve)
    A = sparse.eye(3, format="csr")
    b = np.ones(3)
    with pytest.raises(linear_interface.LinearSolveError, match="did not converge"):
        solve_linear_system(
            A,
            b,
            method="bicgstab",
            equation_type="scalar",
            maxiter=1,
            failure_policy="raise",
        )


def test_bicgstab_breakdown_does_not_switch_method(monkeypatch):
    def broken_bicgstab(A, b, **kwargs):
        return np.zeros_like(b), -10

    monkeypatch.setattr(linear_interface, "bicgstab", broken_bicgstab)
    A = sparse.eye(3, format="csr")
    b = np.ones(3)
    with pytest.raises(linear_interface.LinearSolveError, match="bicgstab did not converge"):
        solve_linear_system(
            A,
            b,
            method="bicgstab",
            equation_type="scalar",
            maxiter=50,
            failure_policy="raise",
        )


def test_generic_iterative_direct_fallback_honors_policy(monkeypatch):
    monkeypatch.setattr(
        linear_interface,
        "cg",
        lambda A, b, **kwargs: (np.zeros_like(b), 1),
    )
    monkeypatch.setattr(linear_interface, "spsolve", lambda A, b: b.copy())

    A = sparse.eye(3, format="csr")
    b = np.ones(3)
    actual = solve_linear_system(
        A,
        b,
        method="cg",
        maxiter=1,
        failure_policy="direct_fallback",
    )

    np.testing.assert_array_equal(actual, b)


def test_direct_fallback_result_is_residual_verified(monkeypatch):
    monkeypatch.setattr(
        linear_interface,
        "cg",
        lambda A, b, **kwargs: (np.zeros_like(b), 1),
    )
    monkeypatch.setattr(linear_interface, "spsolve", lambda A, b: 2.0 * b)

    with pytest.raises(linear_interface.LinearSolveError, match="above the verified limit"):
        solve_linear_system(
            sparse.eye(3, format="csr"),
            np.ones(3),
            method="cg",
            maxiter=1,
            failure_policy="direct_fallback",
        )
