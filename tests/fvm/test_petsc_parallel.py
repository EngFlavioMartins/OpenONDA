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
    ComputeConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.core.parallel import ParallelContext  # noqa: E402
from source.solvers.FVM.fields import diagnostics  # noqa: E402
from source.solvers.FVM.mesh.partition import CellPartition  # noqa: E402
from source.solvers.FVM.mesh.rectilinear import box_mesh_3d  # noqa: E402
from source.solvers.FVM.solve.linear_interface import (  # noqa: E402
    normalized_residual,
    solve_linear_system,
)
from source.solvers.FVM.solve.petsc_partitioned import (  # noqa: E402
    OwnedRowsCSR,
    PartitionedLinearWorkspace,
    solve_owned_rows,
)

from ._structured_mesh import structured_box  # noqa: E402

pytestmark = pytest.mark.mpi


def test_partition_halo_exchange_matches_global_cell_field():
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
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
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
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


def test_owned_row_relative_tolerance_reduces_entry_residual():
    """The distributed PETSc path applies relTol to its warm-guess residual."""
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
    n = 37
    matrix = diags(
        (-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)),
        (-1, 0, 1),
        format="csr",
    )
    rhs = np.linspace(1.0, 2.0, n)
    exact = spsolve(matrix, rhs)
    guess = exact + 2.0e-3 * np.sin(np.arange(n))
    owned = OwnedRowsCSR.from_global(matrix, rhs, context.rank, context.size)
    local_guess = guess[owned.row_start : owned.row_end]

    _, result = solve_owned_rows(
        owned,
        context,
        tolerance=1.0e-12,
        relative_tolerance=0.5,
        initial_guess=local_guess,
    )

    assert 0.0 < result.initial_residual < 0.1
    assert result.final_residual <= 0.5 * result.initial_residual


def test_partitioned_workspace_reuses_allocations_and_closes_collectively():
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
    n = 19
    matrix = diags((-np.ones(n - 1), 3.0 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1), format="csr")
    rhs = np.linspace(0.5, 1.5, n)
    system = OwnedRowsCSR.from_global(matrix, rhs, context.rank, context.size)
    workspace = PartitionedLinearWorkspace(context)
    try:
        first, _ = workspace.solve(
            system,
            method="cg",
            tolerance=1e-11,
            max_iterations=200,
            constant_nullspace=False,
            initial_guess=None,
        )
        matrix_id = id(workspace.matrix)
        ksp_id = id(workspace.ksp)
        second, _ = workspace.solve(
            system,
            method="cg",
            tolerance=1e-11,
            max_iterations=200,
            constant_nullspace=False,
            initial_guess=first,
        )
        assert id(workspace.matrix) == matrix_id
        assert id(workspace.ksp) == ksp_id
        np.testing.assert_allclose(second, first, rtol=1e-11, atol=1e-12)
    finally:
        workspace.close()
    assert workspace.matrix is None
    assert workspace.ksp is None


def test_collective_petsc_solution_matches_scipy():
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
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
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
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
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
    mesh = structured_box(3, 3, 3)
    config = FVMSetup(
        case_name="petsc_pimple",
        execution=ComputeConfig.petsc_replicated(),
        time=TimeConfig.transient(time_step_size=0.01, duration=0.01, output_interval_steps=100),
        schemes=DiscretizationConfig(convection_scheme="upwind"),
        linear=LinearSolverConfig(linear_solver="bicgstab", pressure_tolerance=1e-10),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(config, str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
        residuals = solver.solve_pimple(0.01)

    assert residuals["p"] < 1e-8
    assert np.all(np.isfinite(solver.velocity))
    states = context.comm.allgather(solver.velocity.copy())
    for state in states[1:]:
        np.testing.assert_allclose(state, states[0], rtol=0.0, atol=1e-12)


def _pimple_config(execution, case_name):
    return FVMSetup(
        case_name=case_name,
        execution=execution,
        time=TimeConfig.transient(time_step_size=0.01, duration=0.01, output_interval_steps=100),
        schemes=DiscretizationConfig(convection_scheme="linearUpwind", gradient_scheme="gauss"),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="cg",
            momentum_tolerance=1e-10,
            pressure_tolerance=1e-10,
        ),
        pimple=PimpleControl(n_correctors=2, n_orthogonal_correctors=1),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )


def test_partitioned_solver_header_is_printed_once(tmp_path):
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
    execution = ComputeConfig.petsc_partitioned()
    mesh = structured_box(2, 2, 2)
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        solver = FVMSolver(
            _pimple_config(execution, "single-header"),
            str(tmp_path / "single-header"),
            mesh_data=mesh if context.is_root else None,
        )
        solver.close()

    local_count = stdout.getvalue().count("FVM Solver: Finite Volume Method")
    counts = context.comm.allgather(local_count)
    assert counts == [1] + [0] * (context.size - 1)


def test_partitioned_progress_and_shared_logs_are_root_owned(tmp_path):
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
    execution = ComputeConfig.petsc_partitioned()
    mesh = structured_box(2, 2, 2)
    case_dir = Path(
        context.bcast(
            str(tmp_path / "root-owned-output") if context.is_root else None,
            root=0,
        )
    )
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        from source.solvers.FVM.sampling.forces import ForceSampler

        config = _pimple_config(execution, "root-owned-output")
        config.logging.mode = "debug"
        config.samplers = (ForceSampler(),)
        solver = FVMSolver(
            config,
            str(case_dir),
            mesh_data=mesh if context.is_root else None,
        )
        solver.auto_write = False
        solver.advance(0.01)
        solver.close()

    outputs = context.comm.allgather(stdout.getvalue())
    markers = (
        "FVM Solver: Finite Volume Method",
        "FVM SOLVER INFO",
        "BOUNDARY CONDITIONS",
        "TIME STEP  (step 1,",
        "Solver Convergence",
        "Conservation",
        "[FVM][Timing]",
    )
    for marker in markers:
        assert marker in outputs[0]
        assert all(marker not in output for output in outputs[1:])

    context.barrier()
    if context.is_root:
        diagnostics = case_dir / "solution" / "diagnostics.jsonl"
        forces = case_dir / "samples" / "forces_history.csv"
        assert len(diagnostics.read_text(encoding="utf-8").splitlines()) == 1
        force_lines = forces.read_text(encoding="utf-8").splitlines()
        assert force_lines[0].startswith("time,step,dt,patch,")
        assert len(force_lines) == 5
        fvm_log = (case_dir / "solution" / "fvm.log").read_text(encoding="utf-8")
        for marker in markers:
            assert fvm_log.count(marker) == 1
        assert "Step completed in" not in fvm_log


def test_partitioned_pimple_matches_replicated_reference(tmp_path):
    replicated_execution = ComputeConfig.petsc_replicated()
    replicated_context = ParallelContext.create(replicated_execution)
    mesh = structured_box(5, 4, 3)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = FVMSolver(
            _pimple_config(replicated_execution, "replicated-reference"),
            str(tmp_path / "replicated"),
            mesh_data=mesh,
        )
        reference.auto_write = False
        reference_residuals = reference.solve_pimple(0.01)

    partitioned_execution = ComputeConfig.petsc_partitioned()
    with contextlib.redirect_stdout(io.StringIO()):
        actual = FVMSolver(
            _pimple_config(partitioned_execution, "partitioned"),
            str(tmp_path / "partitioned"),
            mesh_data=mesh if replicated_context.is_root else None,
        )
        actual.auto_write = False
        actual_residuals = actual.solve_pimple(0.01)

    # Momentum and pressure execute sequentially and must replace the same
    # PETSc matrix/KSP allocation instead of retaining one workspace each.
    assert set(actual.algorithm._partitioned_linear_workspaces) == {"flow"}
    assert actual.algorithm._partitioned_linear_workspaces["flow"].matrix is None

    n_owned = actual.parallel.n_owned
    velocity_parts = actual.parallel.comm.allgather(actual.velocity[:n_owned].copy())
    pressure_parts = actual.parallel.comm.allgather(actual.kinematic_pressure[:n_owned].copy())
    velocity = np.concatenate(velocity_parts)
    pressure = np.concatenate(pressure_parts)
    np.testing.assert_allclose(velocity, reference.velocity[: mesh["n_cells"]], atol=2e-9)
    np.testing.assert_allclose(pressure, reference.kinematic_pressure[: mesh["n_cells"]], atol=2e-9)
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
        reference.velocity,
        reference.kinematic_pressure,
        0.02,
        1.0,
        reference.mesh_data,
        reference.geo_data,
        reference.boundaries,
        **force_kwargs,
    )
    local_forces = diagnostics.compute_surface_forces(
        actual.velocity,
        actual.kinematic_pressure,
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


def test_partitioned_initial_velocity_rebuilds_histories_halos_and_flux(tmp_path):
    context = ParallelContext.create(ComputeConfig.petsc_replicated())
    execution = ComputeConfig.petsc_partitioned()
    mesh = structured_box(5, 4, 3)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(
            _pimple_config(execution, "partitioned-initial-velocity"),
            str(tmp_path / "partitioned-initial-velocity"),
            mesh_data=mesh if context.is_root else None,
        )

    partition = solver.parallel.partition
    n_owned = len(partition.owned_global_ids)
    values = np.full((solver.mesh_data["n_cells"], 3), -999.0)
    owned_ids = partition.owned_global_ids.astype(np.float64)
    values[:n_owned] = np.column_stack((owned_ids, owned_ids**2, -owned_ids))
    solver.set_initial_velocity(values)

    expected_ids = partition.local_global_ids.astype(np.float64)
    expected = np.column_stack((expected_ids, expected_ids**2, -expected_ids))
    np.testing.assert_allclose(solver.velocity[: solver.mesh_data["n_cells"]], expected)
    np.testing.assert_allclose(solver.velocity_old, solver.velocity)
    np.testing.assert_allclose(solver.velocity_older, solver.velocity)
    assert np.all(np.isfinite(solver.face_flux))
    assert np.any(np.abs(solver.face_flux) > 0.0)


def test_partitioned_coupling_interface_gathers_and_scatters_global_fields(tmp_path):
    """The coupler sees one root-owned global field, never rank-local fragments."""
    replicated_execution = ComputeConfig.petsc_replicated()
    context = ParallelContext.create(replicated_execution)
    mesh = structured_box(5, 4, 3)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = FVMSolver(
            _pimple_config(replicated_execution, "coupling-interface-reference"),
            str(tmp_path / "coupling-interface-reference"),
            mesh_data=mesh,
        )
        actual = FVMSolver(
            _pimple_config(
                ComputeConfig.petsc_partitioned(),
                "coupling-interface-partitioned",
            ),
            str(tmp_path / "coupling-interface-partitioned"),
            mesh_data=mesh if context.is_root else None,
        )

    partition = actual.parallel.partition
    n_owned = len(partition.owned_global_ids)
    global_ids = partition.owned_global_ids.astype(np.float64)
    actual.velocity[:n_owned] = np.column_stack((global_ids, global_ids**2, -global_ids))
    actual.parallel.exchange_halo(actual.velocity[: actual.mesh_data["n_cells"]])

    velocity = actual.get_velocity_field()
    centres = actual.get_cell_centre_coordinates()
    volumes = actual.get_cell_volumes()
    reference_centres = reference.get_cell_centre_coordinates()
    reference_volumes = reference.get_cell_volumes()
    expected_n = mesh["n_cells"] if context.is_root else 0
    assert velocity.shape == (expected_n, 3)
    assert centres.shape == (expected_n, 3)
    assert volumes.shape == (expected_n,)
    if context.is_root:
        expected_ids = np.arange(mesh["n_cells"], dtype=np.float64)
        np.testing.assert_array_equal(
            velocity,
            np.column_stack((expected_ids, expected_ids**2, -expected_ids)),
        )
        np.testing.assert_allclose(centres, reference_centres)
        np.testing.assert_allclose(volumes, reference_volumes)

    velocity_buffer = np.empty((expected_n, 3), dtype=np.float64)
    assert actual.get_velocity_field_into(velocity_buffer) is velocity_buffer
    np.testing.assert_array_equal(velocity_buffer, velocity)

    patch = "ymin"
    face_centres = actual.get_boundary_face_centre_coordinates(patch)
    face_normals = actual.get_boundary_face_normals(patch)
    face_areas = actual.get_boundary_face_areas(patch)
    reference_face_centres = reference.get_boundary_face_centre_coordinates(patch)
    if context.is_root:
        np.testing.assert_allclose(face_centres, reference_face_centres)
        assert face_normals.shape == face_centres.shape
        assert face_areas.shape == (len(face_centres),)
    else:
        assert face_centres.shape == (0, 3)
        assert face_normals.shape == (0, 3)
        assert face_areas.shape == (0,)

    # Repeated coupling calls reuse the immutable rank/face layout.
    np.testing.assert_array_equal(actual.get_boundary_face_centre_coordinates(patch), face_centres)

    target = face_centres + np.array([0.25, -0.5, 0.75]) if context.is_root else np.empty((0, 3))
    actual.set_dirichlet_velocity_boundary_condition_vec(target, patch)
    boundary = actual._optional_patch(patch)
    local_face_ids = np.empty(0, dtype=np.int64)
    if boundary is not None:
        start = boundary["start_face"]
        stop = start + boundary["n_faces"]
        local_face_ids = actual.mesh_data["global_face_ids"][start:stop]
    ids_by_rank = actual.parallel.comm.allgather(local_face_ids)
    sorted_face_ids = np.sort(np.concatenate(ids_by_rank))
    expected_patch_values = (
        [target[np.searchsorted(sorted_face_ids, rank_ids)] for rank_ids in ids_by_rank]
        if context.is_root
        else None
    )
    expected_local = actual.parallel.comm.scatter(expected_patch_values, root=0)
    if boundary is not None:
        np.testing.assert_allclose(boundary["velocity_value_field"], expected_local)
    else:
        assert expected_local.shape == (0, 3)

    scalar_global = np.linspace(0.0, 1.0, mesh["n_cells"]) if context.is_root else np.empty(0)
    vector_global = (
        np.column_stack((scalar_global, 2.0 * scalar_global, -scalar_global))
        if context.is_root
        else np.empty((0, 3))
    )
    actual.set_cell_scalar_field("scalarDiagnostic", scalar_global)
    actual.set_cell_vector_field(
        "vectorDiagnostic",
        vector_global[:, 0],
        vector_global[:, 1],
        vector_global[:, 2],
    )
    cell_ids_by_rank = actual.parallel.comm.gather(partition.local_global_ids, root=0)
    expected_scalar_payloads = (
        [scalar_global[rank_ids] for rank_ids in cell_ids_by_rank] if context.is_root else None
    )
    expected_vector_payloads = (
        [vector_global[rank_ids] for rank_ids in cell_ids_by_rank] if context.is_root else None
    )
    expected_scalar = actual.parallel.comm.scatter(expected_scalar_payloads, root=0)
    expected_vector = actual.parallel.comm.scatter(expected_vector_payloads, root=0)
    np.testing.assert_allclose(actual.registered_fields["scalarDiagnostic"], expected_scalar)
    np.testing.assert_allclose(actual.registered_fields["vectorDiagnostic"], expected_vector)


def test_partitioned_lsq_pimple_matches_replicated_reference(tmp_path):
    """The production cube uses LSQ gradients, not the Gauss fallback."""
    replicated_execution = ComputeConfig.petsc_replicated()
    context = ParallelContext.create(replicated_execution)
    mesh = structured_box(5, 4, 3)
    reference_config = _pimple_config(replicated_execution, "replicated-lsq")
    reference_config.schemes.gradient_scheme = "lsq"
    with contextlib.redirect_stdout(io.StringIO()):
        reference = FVMSolver(
            reference_config,
            str(tmp_path / "replicated-lsq"),
            mesh_data=mesh,
        )
        reference.auto_write = False
        reference.solve_pimple(0.01)

    partitioned_config = _pimple_config(ComputeConfig.petsc_partitioned(), "partitioned-lsq")
    partitioned_config.schemes.gradient_scheme = "lsq"
    with contextlib.redirect_stdout(io.StringIO()):
        actual = FVMSolver(
            partitioned_config,
            str(tmp_path / "partitioned-lsq"),
            mesh_data=mesh if context.is_root else None,
        )
        actual.auto_write = False
        actual.solve_pimple(0.01)

    pressure_results = [
        result for result in actual.algorithm.last_linear_results if result.equation == "pressure"
    ]
    assert [result.preconditioner_rebuilt for result in pressure_results] == [
        True,
        False,
        False,
        False,
    ]

    n_owned = actual.parallel.n_owned
    velocity = np.concatenate(actual.parallel.comm.allgather(actual.velocity[:n_owned].copy()))
    pressure = np.concatenate(
        actual.parallel.comm.allgather(actual.kinematic_pressure[:n_owned].copy())
    )
    np.testing.assert_allclose(velocity, reference.velocity[: mesh["n_cells"]], atol=3e-9)
    np.testing.assert_allclose(pressure, reference.kinematic_pressure[: mesh["n_cells"]], atol=3e-9)


def test_partitioned_checkpoint_restores_complete_pimple_state(tmp_path):
    execution = ComputeConfig.petsc_partitioned()
    context = ParallelContext.create(execution)
    mesh = box_mesh_3d(
        np.linspace(0.0, 1.0, 5),
        np.linspace(0.0, 1.0, 4),
        np.linspace(0.0, 1.0, 4),
    )
    mesh["boundary"][0]["name"] = "xmin"
    mesh["boundary"][1]["name"] = "xmax"
    shared_root = context.bcast(str(tmp_path) if context.is_root else None)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(
            _pimple_config(execution, "partitioned-restart"),
            str(tmp_path / "run"),
            mesh_data=mesh if context.is_root else None,
        )
        solver.auto_write = False
        solver.solve_pimple(0.01)
        solver.advance_time()
        solver.write_vtk(f"{shared_root}/partitioned-state.vtu")
        import pyvista as pv

        piece = pv.read(
            Path(shared_root) / f"partitioned-state-rank-{solver.parallel.partition.rank:05d}.vtu"
        )
        partition = solver.parallel.partition
        assert piece.n_cells == len(partition.local_global_ids)
        np.testing.assert_array_equal(
            piece.cell_data["GlobalCellIds"],
            partition.local_global_ids,
        )
        assert np.count_nonzero(piece.cell_data["vtkGhostType"]) == len(partition.ghost_global_ids)
        assert "GlobalPointIds" in piece.point_data
        assert "U" not in piece.point_data
        smooth = piece.cell_data_to_point_data()
        assert np.all(np.isfinite(smooth.point_data["U"]))
        if solver.parallel.is_root:
            parallel = pv.read(Path(shared_root) / "partitioned-state.pvtu")
            assert parallel.n_cells >= mesh["n_cells"]
            assert "U" in parallel.cell_data
        solver.save_state(f"{shared_root}/partitioned-checkpoint")

        restored = FVMSolver(
            _pimple_config(execution, "partitioned-restart"),
            str(tmp_path / "restored"),
            mesh_data=mesh if context.is_root else None,
        )
        restored.auto_write = False
        restored.load_state(f"{shared_root}/partitioned-checkpoint")

    for name in (
        "velocity",
        "kinematic_pressure",
        "face_flux",
        "face_flux_old",
        "face_flux_older",
        "velocity_old",
        "velocity_older",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(solver, name))
    assert restored.time == solver.time
    assert restored.step == solver.step
    assert restored._n_committed == solver._n_committed
    if context.is_root:
        assert (Path(shared_root) / "partitioned-state.pvtu").exists()
        for rank in range(context.size):
            assert (Path(shared_root) / f"partitioned-state-rank-{rank:05d}.vtu").exists()
        collection = tmp_path / "run" / "solution" / "partitioned-restart.pvd"
        assert collection.exists()
        assert "partitioned-state.pvtu" in collection.read_text()
        parallel_text = (Path(shared_root) / "partitioned-state.pvtu").read_text()
        assert 'GhostLevel="1"' in parallel_text
