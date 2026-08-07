#!/usr/bin/env python3
"""Laminar flat-plate boundary layer (Blasius validation, FVM).

Flow physics under test: viscous boundary-layer growth on a flat plate.
A uniform stream (U = 1) meets a no-slip plate of length L = 1 whose leading
edge is at x = 0; upstream of it the bottom boundary is a frictionless slip
plane, so the layer starts growing exactly at the leading edge.  At
Re_L = U L / nu = 1e4 the layer stays laminar over the whole plate and the
classical Blasius similarity solution applies (Blasius 1908; Schlichting,
Boundary-Layer Theory):

    u / U       = f'(eta),   eta = y * sqrt(U / (nu x))
    Cf(x)       = 0.664 / sqrt(Re_x)
    delta99(x)  = 5.0 x / sqrt(Re_x)

After the run reaches steady state the script samples wall-normal velocity
profiles at x/L = 0.25, 0.5, 0.75 (solution/profiles.csv) and the skin
friction along the plate (solution/cf.csv); allplot.sh compares both against
the Blasius solution.

Usage:
    python boundaryLayer_setup.py --Re 1e4 --end-time 8
"""

import argparse
import csv
import os

import numpy as np

from assets.mesh_plate import flat_plate_mesh

from openonda.fvm import (
    BoundaryConfig,
    FVMSetup,
    ForceSampler,
    Solver,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
)

STATIONS = (0.25, 0.5, 0.75)  # x/L sampling stations for the Blasius profiles


def build_config(args, nu):
    """FVM configuration for the flat-plate case."""
    solver_schemes = SchemesConfig(
        convection_scheme=args.convection_scheme,
        gradient_scheme="gauss",
    )
    solver_linear = LinearSolverConfig(linear_solver=args.linear_solver)
    solver_pimple = PimpleControl(
        n_correctors=args.n_correctors,
        n_outer_correctors=args.n_outer,
    )
    solver_forces = [
        ForceSampler(
            patch_names=["plate"],
            ref_velocity=args.u_inf,
            ref_length=args.plate_length,
        )
    ]

    return FVMSetup(
        case_name=args.case_name,
        time=TimeConfig(
            delta_t=args.initial_dt,
            start_time=0.0,
            end_time=args.end_time,
            write_interval=10**9,
            write_interval_time=args.write_interval_time,
            adjust_timestep=True,
            max_cfl=args.max_cfl,
            max_delta_t=args.max_dt,
            min_delta_t=1e-6,
        ),
        schemes=solver_schemes,
        linear=solver_linear,
        pimple=solver_pimple,
        samplers=solver_forces,
        transport=TransportConfig(density=args.rho, nu=nu),
        turbulence=None,  # laminar by construction (Re_x <= 1e4 << 5e5)
        boundaries=[
            BoundaryConfig.inlet("inlet", [args.u_inf, 0.0, 0.0]),
            BoundaryConfig.outlet("outlet", p=0.0),
            BoundaryConfig(name="floor", type_U="slip", type_p="zeroGradient"),
            BoundaryConfig.wall("plate"),
            BoundaryConfig(name="top", type_U="slip", type_p="zeroGradient"),
            BoundaryConfig.empty("front"),
            BoundaryConfig.empty("back"),
        ],
        initial_U=[args.u_inf, 0.0, 0.0],
        initial_p=0.0,
    )


def write_profiles(solver, sol_dir, args, nu):
    """Sample u(y) at the stations and Cf(x) along the plate into CSV files."""
    n = solver.mesh_data["n_elements"]
    centroids = solver.geo_data["element_centroids"][:n]
    u = solver.U[:n]
    xc, yc = centroids[:, 0], centroids[:, 1]

    # Column spacing on the plate (uniform x grid there).
    plate_x = np.unique(np.round(xc[xc > 0], 12))
    dx = np.min(np.diff(plate_x))

    with open(os.path.join(sol_dir, "profiles.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["station", "x", "y", "u", "v"])
        for station in STATIONS:
            x_col = plate_x[np.argmin(np.abs(plate_x - station))]
            sel = np.abs(xc - x_col) < 0.5 * dx
            order = np.argsort(yc[sel])
            for y_i, u_i, v_i in zip(
                yc[sel][order], u[sel, 0][order], u[sel, 1][order], strict=True
            ):
                writer.writerow([station, x_col, y_i, u_i, v_i])

    # Skin friction from the wall-adjacent cell row: tau_w ~ mu * u1 / y1.
    # p_wall and the top-row u_e diagnose spurious streamwise pressure
    # gradients / confinement acceleration.
    p = solver.p[:n]
    y1 = yc.min()
    y_top = yc.max()
    wall = (np.abs(yc - y1) < 1e-12) & (xc > 0.0)
    top = np.abs(yc - y_top) < 1e-12
    order = np.argsort(xc[wall])
    x_w = xc[wall][order]
    u_w = u[wall, 0][order]
    p_w = p[wall][order]
    u_e = np.interp(x_w, np.sort(xc[top]), u[top, 0][np.argsort(xc[top])])
    cf = 2.0 * nu * u_w / (y1 * args.u_inf**2)
    rex = args.u_inf * x_w / nu
    with open(os.path.join(sol_dir, "cf.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "Rex", "Cf", "Cf_blasius", "p_wall", "u_top"])
        for row in zip(x_w, rex, cf, 0.664 / np.sqrt(rex), p_w, u_e, strict=True):
            writer.writerow(row)

    print(f"  Profiles written: {os.path.join(sol_dir, 'profiles.csv')}")
    print(f"  Skin friction written: {os.path.join(sol_dir, 'cf.csv')}")


def main():
    parser = argparse.ArgumentParser(description="Laminar flat-plate boundary layer")
    parser.add_argument("--Re", type=float, default=1e4, help="Plate Reynolds number U*L/nu")
    parser.add_argument("--end-time", type=float, default=8.0, help="End time [s]")
    parser.add_argument("--plate-length", type=float, default=1.0, help="Plate length L")
    parser.add_argument("--u-inf", type=float, default=1.0, help="Freestream velocity")
    parser.add_argument("--rho", type=float, default=1.0, help="Density")
    parser.add_argument("--n-plate", type=int, default=72, help="Cells along the plate")
    parser.add_argument("--height", type=float, default=0.35, help="Domain height H")
    parser.add_argument("--dy-wall", type=float, default=0.0015, help="First wall-normal cell")
    parser.add_argument(
        "--dy-ratio", type=float, default=1.12, help="Wall-normal stretching ratio (1 = uniform)"
    )
    parser.add_argument("--initial-dt", type=float, default=0.005, help="Initial dt [s]")
    parser.add_argument("--max-cfl", type=float, default=0.9, help="Target max Courant")
    parser.add_argument("--max-dt", type=float, default=0.02, help="Max dt cap [s]")
    parser.add_argument(
        "--write-interval-time",
        type=float,
        default=2.0,
        help="Write VTK every N seconds of flow time",
    )
    parser.add_argument("--n-correctors", type=int, default=2, help="PISO correctors")
    parser.add_argument("--n-outer", type=int, default=1, help="PIMPLE outer correctors")
    parser.add_argument("--linear-solver", type=str, default="bicgstab")
    parser.add_argument("--convection-scheme", type=str, default="limitedLinear")
    parser.add_argument("--case-name", type=str, default="boundaryLayer")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))
    nu = args.u_inf * args.plate_length / args.Re

    print("\n--- Mesh Generation (rectilinear, in-memory) ---")
    mesh_data, _depth = flat_plate_mesh(
        plate_length=args.plate_length,
        height=args.height,
        n_plate=args.n_plate,
        dy_wall=args.dy_wall,
        ratio=args.dy_ratio,
    )
    delta_L = 5.0 * args.plate_length / np.sqrt(args.Re)
    print(
        f"  cells: {mesh_data['n_elements']}, first wall cell {args.dy_wall} "
        f"(delta99(L)/dy_wall = {delta_L / args.dy_wall:.0f})"
    )

    config = build_config(args, nu)
    solver = Solver(config, case_dir, mesh_data=mesh_data)

    solver.write_vtk()
    while solver.flow_time < config.time.end_time:
        solver.evolve()

    sol_dir = os.path.join(case_dir, "solution")
    os.makedirs(sol_dir, exist_ok=True)
    write_profiles(solver, sol_dir, args, nu)

    print("\nSimulation completed successfully.")
    print("Validation targets (Blasius; Schlichting, Boundary-Layer Theory):")
    print("  u/U = f'(eta) with eta = y*sqrt(U/(nu x));  Cf = 0.664/sqrt(Re_x)")


if __name__ == "__main__":
    main()
