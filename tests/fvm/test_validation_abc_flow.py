"""Integrated 3D periodic PIMPLE verification with an exact Beltrami flow."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.assemble.convection import compute_volumetric_face_flux
from source.solvers.fvm.solve.simple_solver import update_scalar_boundaries

from ._structured_mesh import structured_box

TWO_PI = 2.0 * np.pi


def _abc_velocity(points: np.ndarray, time: float, kinematic_viscosity: float) -> np.ndarray:
    x, y, z = points.T
    amplitude = np.exp(-kinematic_viscosity * time)
    return amplitude * np.column_stack(
        [np.sin(z) + np.cos(y), np.sin(x) + np.cos(z), np.sin(y) + np.cos(x)]
    )


def _run_abc(level: int, *, time_step_size: float = 0.005, steps: int = 4) -> tuple[float, float]:
    mesh = structured_box(level, level, level, lx=TWO_PI, ly=TWO_PI, lz=TWO_PI)
    boundaries = [
        BoundaryConfig.cyclic("xmin", "xmax"),
        BoundaryConfig.cyclic("xmax", "xmin"),
        BoundaryConfig.cyclic("ymin", "ymax"),
        BoundaryConfig.cyclic("ymax", "ymin"),
        BoundaryConfig.cyclic("zmin", "zmax"),
        BoundaryConfig.cyclic("zmax", "zmin"),
    ]
    kinematic_viscosity = 0.1
    params_schemes = DiscretizationConfig(convection_scheme="central", time_scheme="backward")
    params_linear = LinearSolverConfig(linear_solver="spsolve")
    params_pimple = PimpleControl(n_correctors=2, n_outer_correctors=2)
    config = FVMSetup(
        case_name="abc-periodic-3d",
        time=TimeConfig(
            time_step_size=time_step_size,
            end_time=steps * time_step_size,
            output_interval_steps=10**9,
        ),
        schemes=params_schemes,
        linear=params_linear,
        pimple=params_pimple,
        transport=TransportConfig(density=1.0, kinematic_viscosity=kinematic_viscosity),
        boundaries=boundaries,
        initial_velocity=[0.0, 0.0, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(config, mesh_data=mesh)
        solver.auto_write = False
        n_cells = mesh["n_cells"]
        centres = solver.geo_data["cell_centre"]
        initial_velocity = _abc_velocity(centres, 0.0, kinematic_viscosity)
        solver.set_initial_velocity(initial_velocity)
        solver.kinematic_pressure[:n_cells] = -0.5 * np.sum(initial_velocity**2, axis=1)
        solver.kinematic_pressure[:n_cells] -= np.mean(solver.kinematic_pressure[:n_cells])
        update_scalar_boundaries(
            solver.kinematic_pressure, mesh, solver.boundaries, field_name="kinematic_pressure"
        )
        solver.volumetric_face_flux = compute_volumetric_face_flux(
            solver.velocity, mesh, solver.geo_data
        )

        for _ in range(steps):
            solver.solve_pimple(time_step_size)
            solver.advance_time()

    exact = _abc_velocity(centres, steps * time_step_size, kinematic_viscosity)
    volumes = solver.geo_data["cell_volume"]
    error = np.sqrt(np.sum(volumes[:, None] * (solver.velocity[:n_cells] - exact) ** 2))
    norm = np.sqrt(np.sum(volumes[:, None] * exact**2))
    return float(error / norm), solver.last_diagnostics.max_continuity_error


@pytest.mark.verification
@pytest.mark.slow
def test_periodic_abc_flow_is_second_order_in_three_dimensions():
    levels = np.asarray((6, 8, 12), dtype=float)
    results = [_run_abc(int(level)) for level in levels]
    errors = np.asarray([result[0] for result in results])
    continuity = np.asarray([result[1] for result in results])
    order = np.polyfit(np.log(1.0 / levels), np.log(errors), 1)[0]

    assert np.all(np.diff(errors) < 0.0), f"non-monotone 3D ABC errors: {errors}"
    assert order >= 1.8, f"3D ABC order {order:.3f}; errors={errors}"
    assert np.max(continuity) < 1e-10, f"3D ABC continuity defects: {continuity}"
