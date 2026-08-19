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

from openonda.fvm import (
    BoundaryConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
)
from openonda.fvm import (
    compute_continuity_error,
    compute_enstrophy,
    compute_kinetic_energy,
)

from assets.mesh_periodic import periodic_square_mesh

# ---- Numerics ------------------------------------------------------------
DENSITY = 1.0  # fluid density [kg/m^3]
TIME_SCHEME = "backward"
LINEAR_SOLVER = "spsolve"
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1


def exact_velocity(centres: np.ndarray, time: float, nu: float) -> np.ndarray:
    """Return the analytic velocity at cell centres."""
    x = centres[:, 0]
    y = centres[:, 1]
    decay = np.exp(-2.0 * nu * time)
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
    parser.add_argument("--nu", type=float, default=0.1, help="kinematic viscosity [m^2/s]")
    parser.add_argument("--dt", type=float, default=0.005, help="time-step size [s]")
    parser.add_argument("--end-time", type=float, default=0.05, help="simulation end time [s]")
    parser.add_argument("--scheme", choices=("central", "upwind"), default="central")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 4 or args.nu <= 0.0 or args.time_step_size <= 0.0 or args.end_time <= 0.0:
        raise ValueError("n must be at least 4 and nu, dt, and end-time must be positive")
    nsteps = int(round(args.end_time / args.time_step_size))
    if nsteps < 1 or not np.isclose(nsteps * args.time_step_size, args.end_time):
        raise ValueError("end-time must be a positive integer multiple of dt")

    print("\n===== MESH =====")
    print("---- Generating the periodic square mesh ----")
    mesh = periodic_square_mesh(args.n)
    print(f"  cells: {mesh['n_elements']} ({args.n} x {args.n})")

    print("\n===== SIMULATION =====")
    schemes = SchemesConfig(convection_scheme=args.scheme, time_scheme=TIME_SCHEME)
    linear = LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = PimpleControl(n_correctors=PISO_CORRECTORS, n_outer_correctors=OUTER_CORRECTORS)
    boundaries = [
        BoundaryConfig.cyclic("xmin", "xmax"),
        BoundaryConfig.cyclic("xmax", "xmin"),
        BoundaryConfig.cyclic("ymin", "ymax"),
        BoundaryConfig.cyclic("ymax", "ymin"),
        BoundaryConfig.empty("zmin"),
        BoundaryConfig.empty("zmax"),
    ]
    config = FVMSetup(
        case_name="taylorGreen",
        time=TimeConfig(
            time_step_size=args.time_step_size,
            end_time=args.end_time,
            write_interval=nsteps,
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        transport=TransportConfig(density=DENSITY, nu=args.nu),
        boundaries=boundaries,
    )

    solution_dir = CASE_DIR / "solution"
    solution_dir.mkdir(parents=True, exist_ok=True)
    solver = FVMSolver(config, case_dir=str(CASE_DIR), mesh_data=mesh)
    centres = solver.geo_data["element_centroids"]
    volumes = solver.geo_data["element_volumes"]
    solver.set_initial_velocity(exact_velocity(centres, 0.0, args.nu))
    solver.write_vtk()
    initial_energy = compute_kinetic_energy(solver.U, solver.geo_data)
    initial_enstrophy = compute_enstrophy(solver.U, solver.mesh_data, solver.geo_data)

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
        analytic = exact_velocity(centres, solver.time, args.nu)
        energy = compute_kinetic_energy(solver.U, solver.geo_data)
        analytic_energy = initial_energy * np.exp(-4.0 * args.nu * solver.time)
        enstrophy = compute_enstrophy(solver.U, solver.mesh_data, solver.geo_data)
        analytic_enstrophy = initial_enstrophy * np.exp(-4.0 * args.nu * solver.time)
        continuity = compute_continuity_error(solver.phi, solver.mesh_data, solver.geo_data)
        continuity_max = np.max(np.abs(continuity) / (volumes + 1e-30))
        return {
            "step": solver.step,
            "time": solver.time,
            "kinetic_energy": energy,
            "analytic_energy": analytic_energy,
            "energy_relative_error": abs(energy - analytic_energy) / analytic_energy,
            "velocity_l2_error": relative_l2(solver.U[: len(volumes)], analytic, volumes),
            "enstrophy": enstrophy,
            "analytic_enstrophy": analytic_enstrophy,
            "enstrophy_relative_error": abs(enstrophy - analytic_enstrophy) / analytic_enstrophy,
            "continuity_max": float(continuity_max),
            "cfl_max": solver.cfl_max,
        }

    with history_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row())
        for _ in range(nsteps):
            solver.advance()
            writer.writerow(row())

    final = row()
    solver.write_run_manifest()
    print(f"History written: {history_path}")

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    print(f"Final velocity L2 error: {final['velocity_l2_error']:.6e}")
    print(f"Final energy relative error: {final['energy_relative_error']:.6e}")
    print(f"Final enstrophy relative error: {final['enstrophy_relative_error']:.6e}")


if __name__ == "__main__":
    main()
