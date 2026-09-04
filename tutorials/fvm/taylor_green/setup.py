#!/usr/bin/env python3
"""Periodic Taylor–Green vortex with analytic viscous decay (FVM, PIMPLE).

The velocity decays as exp(-2 nu t), providing exact total_kinetic_energy and error checks.

Run with ``python setup.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent

import openonda.fvm as fvm
from openonda.fvm.mesher import periodic_square_mesh

# ---- Numerics ------------------------------------------------------------
DENSITY = 1.0  # fluid density [kg/m^3]
TIME_SCHEME = "backward"
LINEAR_SOLVER = "spsolve"
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
NUMBER_OF_CELLS = 24
KINEMATIC_VISCOSITY = 0.1
TIME_STEP_SIZE = 0.005
FINAL_TIME = 0.05
MAX_COURANT_NUMBER = 0.9
CONVECTION_SCHEME = "central"


def exact_velocity(centres: np.ndarray, time: float, kinematic_viscosity: float) -> np.ndarray:
    """Return the analytic velocity at cell centres."""
    x = centres[:, 0]
    y = centres[:, 1]
    decay = np.exp(-2.0 * kinematic_viscosity * time)
    return np.column_stack(
        (
            decay * np.sin(x) * np.cos(y),
            -decay * np.cos(x) * np.sin(y),
            np.zeros_like(x),
        )
    )


def relative_l2(numerical: np.ndarray, analytic: np.ndarray, cell_volume: np.ndarray) -> float:
    numerator = np.sum(cell_volume[:, None] * (numerical - analytic) ** 2)
    denominator = np.sum(cell_volume[:, None] * analytic**2)
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    nsteps = int(round(FINAL_TIME / TIME_STEP_SIZE))
    if nsteps < 1 or not np.isclose(nsteps * TIME_STEP_SIZE, FINAL_TIME):
        raise ValueError("end_time must be a positive integer multiple of time_step_size")

    print("\n===== MESH =====")
    print("---- Generating the periodic square mesh ----")
    mesh = periodic_square_mesh(NUMBER_OF_CELLS)
    print(f"  cells: {mesh['n_cells']} ({NUMBER_OF_CELLS} x {NUMBER_OF_CELLS})")

    print("\n===== SIMULATION =====")
    schemes = fvm.DiscretizationConfig(convection_scheme=CONVECTION_SCHEME, time_scheme=TIME_SCHEME)
    linear = fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = fvm.PimpleControl(n_correctors=PISO_CORRECTORS, n_outer_correctors=OUTER_CORRECTORS)
    boundaries = [
        fvm.BoundaryConfig.cyclic("xmin", "xmax"),
        fvm.BoundaryConfig.cyclic("xmax", "xmin"),
        fvm.BoundaryConfig.cyclic("ymin", "ymax"),
        fvm.BoundaryConfig.cyclic("ymax", "ymin"),
        fvm.BoundaryConfig.empty("zmin"),
        fvm.BoundaryConfig.empty("zmax"),
    ]
    fvm_setup = fvm.FVMSetup(
        case_name="taylor_green",
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            end_time=FINAL_TIME,
            output_schedule=fvm.RunSchedule(every_n_steps=nsteps),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=MAX_COURANT_NUMBER,
                # Preserve this verification case's nominal time resolution;
                # CFL control may reduce it but does not coarsen it.
                maximum_time_step_size=TIME_STEP_SIZE,
            ),
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
        boundaries=boundaries,
    )

    solution_dir = CASE_DIR / "solution"
    solution_dir.mkdir(parents=True, exist_ok=True)
    fvm_solver = fvm.create_fvm_solver(fvm_setup, case_dir=CASE_DIR, mesh=mesh)
    centres = fvm_solver.geo_data["cell_centre"]
    cell_volume = fvm_solver.geo_data["cell_volume"]
    fvm_solver.set_initial_velocity(exact_velocity(centres, 0.0, KINEMATIC_VISCOSITY))
    fvm_solver.write_vtk()
    initial_total_kinetic_energy = fvm.compute_kinetic_energy(
        fvm_solver.velocity, fvm_solver.geo_data
    )
    initial_enstrophy = fvm.compute_enstrophy(
        fvm_solver.velocity, fvm_solver.mesh_data, fvm_solver.geo_data
    )

    history_path = solution_dir / "history.csv"
    fields = (
        "step",
        "time",
        "total_kinetic_energy",
        "analytic_total_kinetic_energy",
        "total_kinetic_energy_relative_error",
        "velocity_l2_error",
        "total_enstrophy",
        "analytic_total_enstrophy",
        "total_enstrophy_relative_error",
        "max_continuity_error",
        "max_courant_number",
    )

    def row() -> dict[str, float | int]:
        analytic = exact_velocity(centres, fvm_solver.time, KINEMATIC_VISCOSITY)
        total_kinetic_energy = fvm.compute_kinetic_energy(fvm_solver.velocity, fvm_solver.geo_data)
        analytic_total_kinetic_energy = initial_total_kinetic_energy * np.exp(
            -4.0 * KINEMATIC_VISCOSITY * fvm_solver.time
        )
        total_enstrophy = fvm.compute_enstrophy(
            fvm_solver.velocity, fvm_solver.mesh_data, fvm_solver.geo_data
        )
        analytic_total_enstrophy = initial_enstrophy * np.exp(
            -4.0 * KINEMATIC_VISCOSITY * fvm_solver.time
        )
        continuity = fvm.compute_continuity_error(
            fvm_solver.volumetric_face_flux,
            fvm_solver.mesh_data,
            fvm_solver.geo_data,
        )
        max_continuity_error = np.max(np.abs(continuity) / (cell_volume + 1e-30))
        return {
            "step": fvm_solver.step,
            "time": fvm_solver.time,
            "total_kinetic_energy": total_kinetic_energy,
            "analytic_total_kinetic_energy": analytic_total_kinetic_energy,
            "total_kinetic_energy_relative_error": abs(
                total_kinetic_energy - analytic_total_kinetic_energy
            )
            / analytic_total_kinetic_energy,
            "velocity_l2_error": relative_l2(
                fvm_solver.velocity[: len(cell_volume)], analytic, cell_volume
            ),
            "total_enstrophy": total_enstrophy,
            "analytic_total_enstrophy": analytic_total_enstrophy,
            "total_enstrophy_relative_error": abs(total_enstrophy - analytic_total_enstrophy)
            / analytic_total_enstrophy,
            "max_continuity_error": float(max_continuity_error),
            "max_courant_number": fvm_solver.max_courant_number,
        }

    with history_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row())
        while fvm_solver.time < FINAL_TIME - 1.0e-14:
            fvm_solver.advance()
            writer.writerow(row())

    final = row()
    fvm_solver.write_run_manifest()
    print(f"History written: {history_path}")

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    print(f"Final velocity L2 error: {final['velocity_l2_error']:.6e}")
    print(
        f"Final total_kinetic_energy relative error: {final['total_kinetic_energy_relative_error']:.6e}"
    )
    print(f"Final total_enstrophy relative error: {final['total_enstrophy_relative_error']:.6e}")


if __name__ == "__main__":
    main()
