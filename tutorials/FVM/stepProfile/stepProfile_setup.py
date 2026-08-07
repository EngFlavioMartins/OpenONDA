#!/usr/bin/env python3
"""Laminar backward-facing-step flow solved with PIMPLE."""

import argparse
import csv
import os

import numpy as np

from assets.mesh_step import backward_facing_step_mesh

from openonda.fvm import (
    BoundaryConfig,
    FVMSetup,
    Solver,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
)
from openonda.fvm import geometry


def inlet_velocity(mesh_data, geo_data, step_height, mean_velocity):
    """Parabolic inlet profile with the requested bulk velocity."""
    patch = next(item for item in mesh_data["boundary"] if item["name"] == "inlet")
    start = patch["startFace"]
    stop = start + patch["nFaces"]
    y = geo_data["face_centroids"][start:stop, 1]
    eta = np.clip((y - step_height) / step_height, 0.0, 1.0)
    values = np.zeros((patch["nFaces"], 3))
    values[:, 0] = 6.0 * mean_velocity * eta * (1.0 - eta)
    return values


def initial_velocity(geo_data, n_cells, step_height, mean_velocity):
    """Divergence-compatible profile on each side of the expansion."""
    centres = geo_data["element_centroids"][:n_cells]
    x, y = centres[:, 0], centres[:, 1]
    values = np.zeros((n_cells, 3))

    upstream = x < 0.0
    eta_up = np.clip((y[upstream] - step_height) / step_height, 0.0, 1.0)
    values[upstream, 0] = 6.0 * mean_velocity * eta_up * (1.0 - eta_up)

    downstream = ~upstream
    eta_down = np.clip(y[downstream] / (2.0 * step_height), 0.0, 1.0)
    values[downstream, 0] = 3.0 * mean_velocity * eta_down * (1.0 - eta_down)
    return values


def reattachment_location(solver, step_height):
    """Estimate x/h where the first downstream cell row changes to positive u."""
    n_cells = solver.mesh_data["n_elements"]
    centres = solver.geo_data["element_centroids"][:n_cells]
    downstream = centres[:, 0] > 0.0
    y0 = np.min(centres[downstream, 1])
    near_wall = downstream & np.isclose(centres[:, 1], y0, atol=1e-10)
    order = np.argsort(centres[near_wall, 0])
    x = centres[near_wall, 0][order]
    u = solver.U[:n_cells, 0][near_wall][order]

    negative = np.flatnonzero(u < 0.0)
    if not len(negative):
        return np.nan, float(np.min(u))
    last_negative = negative[-1]
    if last_negative + 1 == len(u):
        return np.nan, float(np.min(u))
    x0, x1 = x[last_negative], x[last_negative + 1]
    u0, u1 = u[last_negative], u[last_negative + 1]
    x_re = x0 - u0 * (x1 - x0) / (u1 - u0)
    return float(x_re / step_height), float(np.min(u))


def write_solution_tables(solver, solution_dir, history, step_height):
    """Write cell fields and the reattachment/health history."""
    os.makedirs(solution_dir, exist_ok=True)
    n_cells = solver.mesh_data["n_elements"]
    centres = solver.geo_data["element_centroids"][:n_cells]
    fields_path = os.path.join(solution_dir, "fields.csv")
    with open(fields_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_over_h", "y_over_h", "u", "v", "p"])
        for centre, velocity, pressure in zip(
            centres, solver.U[:n_cells], solver.p[:n_cells], strict=True
        ):
            writer.writerow(
                [
                    centre[0] / step_height,
                    centre[1] / step_height,
                    velocity[0],
                    velocity[1],
                    pressure,
                ]
            )

    history_path = os.path.join(solution_dir, "reattachment_history.csv")
    with open(history_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "x_reattachment_over_h", "min_near_wall_u", "continuity", "cfl"])
        writer.writerows(history)
    print(f"  Cell fields written: {fields_path}")
    print(f"  Reattachment history written: {history_path}")


def build_config(args, inlet_values, nu):
    params_schemes = SchemesConfig(
        convection_scheme=args.convection_scheme,
        gradient_scheme="gauss",
    )
    params_linear = LinearSolverConfig(linear_solver=args.linear_solver)
    params_pimple = PimpleControl(
        n_correctors=args.n_correctors,
        n_outer_correctors=args.n_outer,
    )
    return FVMSetup(
        case_name=args.case_name,
        time=TimeConfig(
            delta_t=args.initial_dt,
            end_time=args.end_time,
            write_interval=10**9,
            write_interval_time=args.write_interval_time,
            adjust_timestep=True,
            max_cfl=args.max_cfl,
            max_delta_t=args.max_dt,
            min_delta_t=1e-6,
        ),
        schemes=params_schemes,
        linear=params_linear,
        pimple=params_pimple,
        transport=TransportConfig(density=args.rho, nu=nu),
        turbulence=None,
        boundaries=[
            BoundaryConfig.inlet("inlet", inlet_values.tolist()),
            BoundaryConfig.outlet("outlet", p=0.0),
            BoundaryConfig.wall("walls"),
            BoundaryConfig.empty("front"),
            BoundaryConfig.empty("back"),
        ],
        initial_U=[0.0, 0.0, 0.0],
        initial_p=0.0,
    )


def main():
    parser = argparse.ArgumentParser(description="Laminar backward-facing-step PIMPLE tutorial")
    parser.add_argument("--Re", type=float, default=100.0, help="Re_h = U_bulk h / nu")
    parser.add_argument("--end-time", type=float, default=12.0)
    parser.add_argument("--step-height", type=float, default=1.0)
    parser.add_argument("--u-mean", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--n-upstream", type=int, default=24)
    parser.add_argument("--n-downstream", type=int, default=120)
    parser.add_argument("--n-height", type=int, default=16)
    parser.add_argument("--initial-dt", type=float, default=0.02)
    parser.add_argument("--max-dt", type=float, default=0.05)
    parser.add_argument("--max-cfl", type=float, default=0.7)
    parser.add_argument("--write-interval-time", type=float, default=2.0)
    parser.add_argument("--n-correctors", type=int, default=2)
    parser.add_argument("--n-outer", type=int, default=1)
    parser.add_argument("--linear-solver", default="bicgstab")
    parser.add_argument("--convection-scheme", default="limitedLinear")
    parser.add_argument("--case-name", default="stepProfile")
    args = parser.parse_args()

    if args.Re <= 0.0:
        parser.error("--Re must be positive")
    case_dir = os.path.dirname(os.path.abspath(__file__))
    solution_dir = os.path.join(case_dir, "solution")

    print("\n--- Backward-facing-step mesh ---")
    mesh_data, depth = backward_facing_step_mesh(
        step_height=args.step_height,
        n_upstream=args.n_upstream,
        n_downstream=args.n_downstream,
        n_height=args.n_height,
    )
    geo_data = geometry.compute_mesh_geometry(mesh_data)
    print(f"  cells: {mesh_data['n_elements']}; extrusion depth: {depth:g}")
    print("  geometry: inlet y/h=1..2, vertical step at x/h=0, outlet height=2h")

    nu = args.u_mean * args.step_height / args.Re
    inlet_values = inlet_velocity(mesh_data, geo_data, args.step_height, args.u_mean)
    config = build_config(args, inlet_values, nu)
    solver = Solver(config, case_dir, mesh_data=mesh_data)
    solver.set_initial_velocity(
        initial_velocity(geo_data, mesh_data["n_elements"], args.step_height, args.u_mean)
    )
    solver.write_vtk()

    history = []
    while solver.flow_time < config.time.end_time:
        solver.evolve()
        x_re, min_u = reattachment_location(solver, args.step_height)
        diagnostics = solver.last_diagnostics
        history.append(
            [solver.flow_time, x_re, min_u, diagnostics.continuity_max, diagnostics.cfl_max]
        )

    write_solution_tables(solver, solution_dir, history, args.step_height)
    x_re, min_u = reattachment_location(solver, args.step_height)
    print("\nSimulation completed successfully.")
    print(f"  Re_h={args.Re:g}; minimum downstream near-wall u={min_u:.6g}")
    print(
        f"  estimated x_reattachment/h={x_re:.6g}"
        if np.isfinite(x_re)
        else "  reattachment not resolved yet"
    )


if __name__ == "__main__":
    main()
