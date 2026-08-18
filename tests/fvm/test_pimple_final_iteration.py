"""Final-corrector semantics for transient PIMPLE relaxation.

PIMPLE applies relaxation factors on every outer corrector *except* the
last: ``fvMatrix::relax`` and ``GeometricField::relax`` look the factor up under
``UFinal`` / ``pFinal``, and the relaxationFactors section does not define
those final-field entries.  The
committed time step is therefore always the result of an *unrelaxed* solve, so
``alpha_u`` / ``alpha_p`` accelerate the outer loop without adding a permanent
``(1-α)/α · diag(A) · ΔU`` lag term that would retard the physical rate of
change (and, in a bluff-body wake, suppress shear-layer rollup and shedding).

The sharp consequence pinned here: with a single outer corrector — which *is*
the final one — the relaxation factors must have no effect whatsoever.
"""

import contextlib
import io

import numpy as np
import pytest

from source.solvers.FVM import (
    BoundaryConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)

from ._structured_mesh import structured_box

DT = 0.02
N_STEPS = 4


def _run(tmp_path, alpha_u, alpha_p, n_outer):
    """March a startup channel flow and return the interior ``(U, p)``."""
    mesh = structured_box(6, 5, 4)
    config = FVMSetup(
        case_name="pimple_final",
        time=TimeConfig.transient(dt=DT, duration=DT * N_STEPS, write_interval=10**9),
        schemes=SchemesConfig(convection_scheme="LUST", time_scheme="backward"),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(
            n_correctors=2,
            n_outer_correctors=n_outer,
            alpha_u=alpha_u,
            alpha_p=alpha_p,
        ),
        transport=TransportConfig(density=1.0, nu=0.01),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.slip("zmin"),
            BoundaryConfig.slip("zmax"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
        initial_p=0.0,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(config, str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
        n_elements = mesh["n_elements"]
        for _ in range(N_STEPS):
            solver.solve_pimple(DT)
            solver.advance_time()
        return solver.U[:n_elements].copy(), solver.p[:n_elements].copy()


@pytest.mark.parametrize("alpha_u, alpha_p", [(0.7, 0.3), (0.5, 0.2)])
def test_single_outer_corrector_ignores_relaxation(tmp_path, alpha_u, alpha_p):
    """The only corrector is the final one, so relaxation must be inert."""
    reference_U, reference_p = _run(tmp_path, 1.0, 1.0, n_outer=1)
    relaxed_U, relaxed_p = _run(tmp_path, alpha_u, alpha_p, n_outer=1)

    assert np.allclose(relaxed_U, reference_U, rtol=0.0, atol=1e-13)
    assert np.allclose(relaxed_p, reference_p, rtol=0.0, atol=1e-13)


def test_relaxation_converges_to_the_unrelaxed_step():
    """More outer correctors must drive a relaxed step onto the unrelaxed one.

    Relaxation may only slow the outer loop down; it may not move the fixed
    point.  Correcting the cell velocity by the *full* ``p_prime`` while the
    pressure only accumulates ``alpha_p * p_prime`` leaves the two describing
    different states, and the loop then settles on an ``alpha_p``-dependent
    answer that no number of outer correctors removes.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as case_dir:
        reference_U, _ = _run(case_dir, 1.0, 1.0, n_outer=16)
        near_U, _ = _run(case_dir, 0.7, 0.3, n_outer=2)
        far_U, _ = _run(case_dir, 0.7, 0.3, n_outer=16)

    scale = np.linalg.norm(reference_U)
    error_two = np.linalg.norm(near_U - reference_U) / scale
    error_converged = np.linalg.norm(far_U - reference_U) / scale

    assert error_converged < 0.05 * error_two
    assert error_converged < 1e-6


def test_relative_linear_tolerances_are_disabled_at_final_stages(tmp_path, monkeypatch):
    """Exercise intermediate/final momentum and pressure solver selection."""
    from source.solvers.FVM.assemble import momentum
    from source.solvers.FVM.solve import pimple_solver

    momentum_tolerances = []
    pressure_tolerances = []
    original_momentum_solve = momentum.solve_linear_system
    original_pressure_solve = pimple_solver.solve_linear_system

    def capture_momentum(*args, **kwargs):
        momentum_tolerances.append((kwargs["tol"], kwargs["rel_tol"]))
        return original_momentum_solve(*args, **kwargs)

    def capture_pressure(*args, **kwargs):
        pressure_tolerances.append((kwargs["tol"], kwargs["rel_tol"]))
        return original_pressure_solve(*args, **kwargs)

    monkeypatch.setattr(momentum, "solve_linear_system", capture_momentum)
    monkeypatch.setattr(pimple_solver, "solve_linear_system", capture_pressure)

    mesh = structured_box(4, 3, 3)
    config = FVMSetup(
        case_name="pimple_linear_final",
        time=TimeConfig.transient(dt=DT, duration=DT, write_interval=10**9),
        schemes=SchemesConfig(convection_scheme="upwind"),
        linear=LinearSolverConfig(
            linear_solver="bicgstab",
            momentum_tol=1e-6,
            momentum_rel_tol=0.1,
            pressure_tol=1e-6,
            pressure_rel_tol=0.01,
        ),
        pimple=PimpleControl(
            n_correctors=2,
            n_outer_correctors=2,
            n_orthogonal_correctors=1,
        ),
        transport=TransportConfig(density=1.0, nu=0.01),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.slip("zmin"),
            BoundaryConfig.slip("zmax"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(config, str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
        solver.evolve()

    assert momentum_tolerances == pytest.approx([(1e-6, 0.1)] * 3 + [(1e-6, 0.0)] * 3)
    assert pressure_tolerances == pytest.approx(
        [(1e-6, 0.01)] * 3 + [(1e-6, 0.0)] + [(1e-6, 0.01)] * 3 + [(1e-6, 0.0)]
    )
