"""Parity gates for typed field state and accelerated CPU assembly."""

from __future__ import annotations

import contextlib
import io

import numpy as np

from source.solvers.FVM import (
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FieldState,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.assemble.matrix_assembly import (
    MatrixAssemblyWorkspace,
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)

from ._structured_mesh import structured_box


def test_field_state_normalizes_layout_and_copies_independently():
    state = FieldState(np.zeros((4, 3)), np.zeros(4), np.zeros(6))
    checkpoint = state.copy()
    state.velocity[0, 0] = 1.0
    assert state.velocity.flags.c_contiguous
    assert checkpoint.velocity[0, 0] == 0.0


def test_numba_assembly_matches_numpy(hand_built_3d_mesh):
    mesh = hand_built_3d_mesh
    n_faces = mesh["n_faces"]
    flux = {
        "flux_cf": np.linspace(0.5, 1.5, n_faces),
        "flux_ff": np.linspace(-1.0, -0.2, n_faces),
        "flux_vf": np.sin(np.arange(n_faces, dtype=np.float64)),
    }
    numpy_matrix = assemble_matrix_from_fluxes_vectorized(
        flux,
        mesh,
        workspace=MatrixAssemblyWorkspace.create(mesh),
        backend="numpy",
    )
    numba_matrix = assemble_matrix_from_fluxes_vectorized(
        flux,
        mesh,
        workspace=MatrixAssemblyWorkspace.create(mesh),
        backend="numba",
    )
    np.testing.assert_array_equal(numba_matrix.toarray(), numpy_matrix.toarray())
    np.testing.assert_allclose(
        assemble_rhs_from_fluxes_vectorized(flux, mesh, backend="numba"),
        assemble_rhs_from_fluxes_vectorized(flux, mesh, backend="numpy"),
        rtol=0.0,
        atol=5e-16,
    )


def _run_steps(tmp_path, backend, steps=1):
    mesh = structured_box(3, 2, 2)
    config = FVMSetup(
        case_name=f"pimple_{backend}",
        execution=ComputeConfig(operator_backend=backend),
        time=TimeConfig.transient(
            time_step_size=0.01, duration=steps * 0.01, output_interval_steps=100
        ),
        schemes=DiscretizationConfig(convection_scheme="upwind"),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(n_correctors=1),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.01),
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
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(config, str(tmp_path / backend), mesh_data=mesh)
        solver.auto_write = False
        diagnostics = None
        for _ in range(steps):
            diagnostics = solver.solve_pimple(0.01)
            solver.advance_time()
    assert diagnostics is not None
    return (
        solver.velocity.copy(),
        solver.kinematic_pressure.copy(),
        solver.face_flux.copy(),
        diagnostics,
    )


def test_numba_one_step_matches_numpy(tmp_path):
    numpy_result = _run_steps(tmp_path, "numpy")
    numba_result = _run_steps(tmp_path, "numba")
    for numba_values, numpy_values in zip(numba_result[:3], numpy_result[:3], strict=True):
        np.testing.assert_allclose(numba_values, numpy_values, rtol=0.0, atol=1e-13)
    assert numba_result[3].keys() == numpy_result[3].keys()
    for key in numpy_result[3]:
        np.testing.assert_allclose(numba_result[3][key], numpy_result[3][key], atol=1e-13)


def test_taichi_cpu_one_step_matches_numpy(tmp_path):
    numpy_result = _run_steps(tmp_path, "numpy")
    taichi_result = _run_steps(tmp_path, "taichi")
    for taichi_values, numpy_values in zip(taichi_result[:3], numpy_result[:3], strict=True):
        np.testing.assert_allclose(taichi_values, numpy_values, rtol=0.0, atol=1e-12)
    for key in numpy_result[3]:
        np.testing.assert_allclose(taichi_result[3][key], numpy_result[3][key], atol=1e-12)


def test_accelerated_backends_match_numpy_over_bdf2_history(tmp_path):
    reference = _run_steps(tmp_path, "numpy", steps=4)
    for backend in ("numba", "taichi"):
        actual = _run_steps(tmp_path, backend, steps=4)
        for values, expected in zip(actual[:3], reference[:3], strict=True):
            np.testing.assert_allclose(values, expected, rtol=0.0, atol=2e-12)
        for key in reference[3]:
            np.testing.assert_allclose(actual[3][key], reference[3][key], atol=2e-12)
