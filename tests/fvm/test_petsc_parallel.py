"""Collective PETSc verification.

Run this module collectively, for example::

    mpiexec -n 2 python -m pytest -q tests/fvm/test_petsc_parallel.py

It is skipped during the ordinary SciPy-only suite.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

mpi4py = pytest.importorskip("mpi4py", reason="parallel FVM test requires mpi4py")
pytest.importorskip("petsc4py", reason="parallel FVM test requires petsc4py")

from source.solvers.FVM import (  # noqa: E402
    BoundaryConfig,
    ExecutionConfig,
    FVMConfig,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.core.parallel import ParallelContext  # noqa: E402
from source.solvers.FVM.fields import diagnostics  # noqa: E402
from source.solvers.FVM.mesh.partition import CellPartition  # noqa: E402
from source.solvers.FVM.solve.linear_interface import (  # noqa: E402
    normalized_residual,
    solve_linear_system,
)
from source.solvers.FVM.solve.petsc_partitioned import (  # noqa: E402
    OwnedRowsCSR,
    solve_owned_rows,
)

from ._structured_mesh import structured_box  # noqa: E402

pytestmark = pytest.mark.mpi


def test_partition_halo_exchange_matches_global_cell_field():
    context = ParallelContext.create(ExecutionConfig.petsc_replicated())
    mesh = structured_box(8, 3, 2)
    partition = CellPartition.from_mesh_data(mesh, context.rank, context.size)
    local = np.full((len(partition.local_global_ids), 2), np.nan)
    n_owned = len(partition.owned_global_ids)
    local[:n_owned, 0] = partition.owned_global_ids
    local[:n_owned, 1] = partition.owned_global_ids**2

    partition.exchange_halo(local, context.comm)

    np.testing.assert_array_equal(local[:, 0], partition.local_global_ids)
    np.testing.assert_array_equal(local[:, 1], partition.local_global_ids**2)


def test_owned_row_petsc_solution_matches_scipy():
    context = ParallelContext.create(ExecutionConfig.petsc_replicated())
    n = 37
    matrix = diags(
        (-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)),
        (-1, 0, 1),
        format="csr",
    )
    rhs = np.linspace(1.0, 2.0, n)
    expected = spsolve(matrix, rhs)
    owned = OwnedRowsCSR.from_global(matrix, rhs, context.rank, context.size)

    local, result = solve_owned_rows(owned, context, tolerance=1e-11)
    gathered = context.comm.allgather(local)
    actual = np.concatenate(gathered)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
    assert result.converged
    assert result.backend == "petsc-partitioned"


def test_collective_petsc_solution_matches_scipy():
    context = ParallelContext.create(ExecutionConfig.petsc_replicated())
    n = 31
    A = diags((-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1), format="csr")
    b = np.linspace(1.0, 2.0, n)
    expected = spsolve(A, b)

    actual, info = solve_linear_system(
        A,
        b,
        method="cg",
        equation_type="pressure",
        tol=1e-11,
        maxiter=200,
        backend="petsc",
        parallel_context=context,
        return_info=True,
        failure_policy="raise",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
    assert info.converged
    assert normalized_residual(A, actual, b) < 1e-10


def test_collective_petsc_constant_pressure_nullspace():
    context = ParallelContext.create(ExecutionConfig.petsc_replicated())
    n = 24
    matrix = diags(
        (-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)),
        (-1, 0, 1),
        format="lil",
    )
    matrix[0, 0] = 1.0
    matrix[-1, -1] = 1.0
    matrix = matrix.tocsr()
    coordinates = np.linspace(0.0, 1.0, n)
    expected = np.cos(2.0 * np.pi * coordinates)
    expected -= np.mean(expected)
    rhs = matrix @ expected

    actual, info = solve_linear_system(
        matrix,
        rhs,
        method="cg",
        equation_type="pressure",
        tol=1e-11,
        maxiter=300,
        backend="petsc",
        parallel_context=context,
        nullspace="constant",
        return_info=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    assert info.nullspace == "constant"
    assert info.converged


def test_collective_pimple_step_is_rank_invariant(tmp_path):
    context = ParallelContext.create(ExecutionConfig.petsc_replicated())
    mesh = structured_box(3, 3, 3)
    config = FVMConfig(
        case_name="petsc_pimple",
        execution=ExecutionConfig.petsc_replicated(),
        time=TimeConfig.transient(dt=0.01, duration=0.01, write_interval=100),
        schemes=SchemesConfig(convection_scheme="upwind"),
        linear=LinearSolverConfig(linear_solver="bicgstab", pressure_tol=1e-10),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, nu=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_U=[1.0, 0.0, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(config, str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
        residuals = solver.solve_pimple(0.01)

    assert residuals["p"] < 1e-8
    assert np.all(np.isfinite(solver.U))
    states = context.comm.allgather(solver.U.copy())
    for state in states[1:]:
        np.testing.assert_allclose(state, states[0], rtol=0.0, atol=1e-12)


def _pimple_config(execution, case_name):
    return FVMConfig(
        case_name=case_name,
        execution=execution,
        time=TimeConfig.transient(dt=0.01, duration=0.01, write_interval=100),
        schemes=SchemesConfig(convection_scheme="upwind", gradient_scheme="gauss"),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="cg",
            momentum_tol=1e-10,
            pressure_tol=1e-10,
        ),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, nu=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_U=[1.0, 0.0, 0.0],
        initial_p=0.0,
    )


def test_partitioned_pimple_matches_replicated_reference(tmp_path):
    replicated_execution = ExecutionConfig.petsc_replicated()
    replicated_context = ParallelContext.create(replicated_execution)
    mesh = structured_box(5, 4, 3)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = Solver(
            _pimple_config(replicated_execution, "replicated-reference"),
            str(tmp_path / "replicated"),
            mesh_data=mesh,
        )
        reference.auto_write = False
        reference_residuals = reference.solve_pimple(0.01)

    partitioned_execution = ExecutionConfig.petsc_partitioned()
    with contextlib.redirect_stdout(io.StringIO()):
        actual = Solver(
            _pimple_config(partitioned_execution, "partitioned"),
            str(tmp_path / "partitioned"),
            mesh_data=mesh if replicated_context.is_root else None,
        )
        actual.auto_write = False
        actual_residuals = actual.solve_pimple(0.01)

    n_owned = actual.parallel.n_owned
    velocity_parts = actual.parallel.comm.allgather(actual.U[:n_owned].copy())
    pressure_parts = actual.parallel.comm.allgather(actual.p[:n_owned].copy())
    velocity = np.concatenate(velocity_parts)
    pressure = np.concatenate(pressure_parts)
    np.testing.assert_allclose(velocity, reference.U[: mesh["n_elements"]], atol=2e-9)
    np.testing.assert_allclose(pressure, reference.p[: mesh["n_elements"]], atol=2e-9)
    assert actual.last_diagnostics.continuity_max == pytest.approx(
        reference.last_diagnostics.continuity_max, rel=2e-7, abs=1e-10
    )
    assert actual_residuals["p"] == pytest.approx(reference_residuals["p"], abs=2e-9)
    force_kwargs = {
        "patch_names": ["ymin", "ymax", "zmin", "zmax"],
        "ref_U": 1.0,
        "ref_area": 1.0,
        "ref_length": 1.0,
    }
    reference_forces = diagnostics.compute_surface_forces(
        reference.U,
        reference.p,
        0.02,
        1.0,
        reference.mesh_data,
        reference.geo_data,
        reference.boundaries,
        **force_kwargs,
    )
    local_forces = diagnostics.compute_surface_forces(
        actual.U,
        actual.p,
        0.02,
        1.0,
        actual.mesh_data,
        actual.geo_data,
        actual.boundaries,
        **force_kwargs,
    )
    partitioned_forces = diagnostics.merge_partition_forces(
        actual.parallel.comm.allgather(local_forces)
    )
    for patch, expected in reference_forces.items():
        np.testing.assert_allclose(
            partitioned_forces[patch]["Ftot"], expected["Ftot"], rtol=1e-7, atol=2e-8
        )


def test_partitioned_checkpoint_restores_complete_pimple_state(tmp_path):
    execution = ExecutionConfig.petsc_partitioned()
    context = ParallelContext.create(execution)
    mesh = structured_box(4, 3, 3)
    shared_root = context.bcast(str(tmp_path) if context.is_root else None)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(
            _pimple_config(execution, "partitioned-restart"),
            str(tmp_path / "run"),
            mesh_data=mesh if context.is_root else None,
        )
        solver.auto_write = False
        solver.solve_pimple(0.01)
        solver.advance_time()
        solver.write_vtk(f"{shared_root}/partitioned-state.vtu")
        solver.save_state(f"{shared_root}/partitioned-checkpoint")

        restored = Solver(
            _pimple_config(execution, "partitioned-restart"),
            str(tmp_path / "restored"),
            mesh_data=mesh if context.is_root else None,
        )
        restored.auto_write = False
        restored.load_state(f"{shared_root}/partitioned-checkpoint")

    for name in ("U", "p", "phi", "U_old", "U_old_old"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(solver, name))
    assert restored.flow_time == solver.flow_time
    assert restored.time_step == solver.time_step
    assert restored._n_committed == solver._n_committed
    if context.is_root:
        assert (Path(shared_root) / "partitioned-state.pvtu").exists()
        for rank in range(context.size):
            assert (Path(shared_root) / f"partitioned-state-rank-{rank:05d}.vtu").exists()
