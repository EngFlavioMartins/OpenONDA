#!/usr/bin/env python3
"""Periodic two-dimensional Taylor–Green vortex, validated against the
analytic decay (FVM, PIMPLE).

The classic Taylor–Green vortex u = (sin x cos y, -cos x sin y) decays
exactly as exp(-2 nu t). Because the solution is known, the numerical decay
of kinetic energy, enstrophy, and the velocity field itself can be compared
with the analytic values step by step (solution/history.csv). This is a
periodic, single-time-step-invariant case; the default run uses the central
convection scheme.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent

import openonda.fvm as fvm

from assets.mesh_periodic import periodic_square_mesh

# ---- Numerics ------------------------------------------------------------
DENSITY = 1.0  # fluid density [kg/m^3]
TIME_SCHEME = "backward"
LINEAR_SOLVER = "spsolve"
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1


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


def relative_l2(numerical: np.ndarray, analytic: np.ndarray, volumes: np.ndarray) -> float:
    numerator = np.sum(volumes[:, None] * (numerical - analytic) ** 2)
    denominator = np.sum(volumes[:, None] * analytic**2)
    return float(np.sqrt(numerator / denominator))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24, help="cells along each periodic direction")
    parser.add_argument(
        "--kinematic-viscosity", type=float, default=0.1, help="kinematic viscosity [m^2/s]"
    )
    parser.add_argument("--time-step-size", type=float, default=0.005, help="time-step size [s]")
    parser.add_argument("--end-time", type=float, default=0.05, help="simulation end time [s]")
    parser.add_argument("--scheme", choices=("central", "upwind"), default="central")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.n < 4
        or args.kinematic_viscosity <= 0.0
        or args.time_step_size <= 0.0
        or args.end_time <= 0.0
    ):
        raise ValueError("n must be at least 4 and nu, dt, and end-time must be positive")
    nsteps = int(round(args.end_time / args.time_step_size))
    if nsteps < 1 or not np.isclose(nsteps * args.time_step_size, args.end_time):
        raise ValueError("end-time must be a positive integer multiple of dt")

    print("\n===== MESH =====")
    print("---- Generating the periodic square mesh ----")
    mesh = periodic_square_mesh(args.n)
    print(f"  cells: {mesh['n_cells']} ({args.n} x {args.n})")

    print("\n===== SIMULATION =====")
    schemes = fvm.DiscretizationConfig(convection_scheme=args.scheme, time_scheme=TIME_SCHEME)
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
            time_step_size=args.time_step_size,
            end_time=args.end_time,
            output_interval_steps=nsteps,
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        transport=fvm.TransportConfig(
            density=DENSITY, kinematic_viscosity=args.kinematic_viscosity
        ),
        boundaries=boundaries,
    )

    solution_dir = CASE_DIR / "solution"
    solution_dir.mkdir(parents=True, exist_ok=True)
    fvm_solver = fvm.FVMSolver(fvm_setup, case_dir=str(CASE_DIR), mesh_data=mesh)
    centres = fvm_solver.geo_data["cell_centroids"]
    volumes = fvm_solver.geo_data["cell_volumes"]
    fvm_solver.set_initial_velocity(exact_velocity(centres, 0.0, args.kinematic_viscosity))
    fvm_solver.write_vtk()
    initial_energy = fvm.compute_kinetic_energy(fvm_solver.velocity, fvm_solver.geo_data)
    initial_enstrophy = fvm.compute_enstrophy(
        fvm_solver.velocity, fvm_solver.mesh_data, fvm_solver.geo_data
    )

    history_path = solution_dir / "history.csv"
    fields = (
        "step",
        "time",
        "kinetic_energy",
        "analytic_energy",
        "energy_relative_error",
        "velocity_l2_error",
        "enstrophy",
        "analytic_enstrophy",
        "enstrophy_relative_error",
        "continuity_max",
        "cfl_max",
    )

    def row() -> dict[str, float | int]:
        analytic = exact_velocity(centres, fvm_solver.time, args.kinematic_viscosity)
        energy = fvm.compute_kinetic_energy(fvm_solver.velocity, fvm_solver.geo_data)
        analytic_energy = initial_energy * np.exp(-4.0 * args.kinematic_viscosity * fvm_solver.time)
        enstrophy = fvm.compute_enstrophy(
            fvm_solver.velocity, fvm_solver.mesh_data, fvm_solver.geo_data
        )
        analytic_enstrophy = initial_enstrophy * np.exp(
            -4.0 * args.kinematic_viscosity * fvm_solver.time
        )
        continuity = fvm.compute_continuity_error(
            fvm_solver.face_flux, fvm_solver.mesh_data, fvm_solver.geo_data
        )
        continuity_max = np.max(np.abs(continuity) / (volumes + 1e-30))
        return {
            "step": fvm_solver.step,
            "time": fvm_solver.time,
            "kinetic_energy": energy,
            "analytic_energy": analytic_energy,
            "energy_relative_error": abs(energy - analytic_energy) / analytic_energy,
            "velocity_l2_error": relative_l2(
                fvm_solver.velocity[: len(volumes)], analytic, volumes
            ),
            "enstrophy": enstrophy,
            "analytic_enstrophy": analytic_enstrophy,
            "enstrophy_relative_error": abs(enstrophy - analytic_enstrophy) / analytic_enstrophy,
            "continuity_max": float(continuity_max),
            "cfl_max": fvm_solver.cfl_max,
        }

    with history_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row())
        for _ in range(nsteps):
            fvm_solver.advance()
            writer.writerow(row())

    final = row()
    fvm_solver.write_run_manifest()
    print(f"History written: {history_path}")

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    print(f"Final velocity L2 error: {final['velocity_l2_error']:.6e}")
    print(f"Final energy relative error: {final['energy_relative_error']:.6e}")
    print(f"Final enstrophy relative error: {final['enstrophy_relative_error']:.6e}")


if __name__ == "__main__":
    main()
