#!/usr/bin/env python3
"""Von Karman vortex street behind a square cylinder (body-fitted FVM).

Flow physics under test: laminar bluff-body vortex shedding.  A square-section
cylinder (an extruded cube, side D) sits in a uniform stream at Re = U*D/nu.
Above Re ~ 50 the wake becomes globally unstable and sheds a periodic von
Karman street whose frequency and mean drag are classical validation data:

  Re = 100, blockage 5%:
    Strouhal St = f*D/U   -> 0.140-0.150
    mean drag  Cd         -> 1.45-1.58
  (Okajima, J. Fluid Mech. 123, 1982: exp. St ~ 0.14; Sohankar et al.,
   Int. J. Numer. Meth. Fluids 26, 1998: St = 0.146, Cd = 1.48;
   Sen, Mittal & Biswas, Int. J. Numer. Meth. Fluids 67, 2011:
   St = 0.145, Cd = 1.53.)

The case is quasi-2D (one cell thick, ``empty`` front/back) on a rectilinear
mesh generated in-memory (assets/mesh_square.py).  Cd(t), Cl(t) are logged
every step to solution/forces_history.csv; allplot.sh extracts the Strouhal
number from the lift signal and compares both against the bands above.

Usage:
    python cubeFlow_setup.py --Re 100 --end-time 120
"""

import argparse
import os

from assets.mesh_square import square_cylinder_mesh

from openonda.fvm import (
    BoundaryConfig,
    FVMConfig,
    Solver,
    ForcesConfig,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
)


def build_config(args, depth):
    """FVM configuration for the square-cylinder shedding case."""
    nu = args.u_inf * args.D / args.Re

    schemes = SchemesConfig(
        convection_scheme=args.convection_scheme,
        gradient_scheme="gauss",
    )
    linear = LinearSolverConfig(linear_solver=args.linear_solver)
    pimple = PimpleControl(
        n_correctors=args.n_correctors,
        n_outer_correctors=args.n_outer,
    )
    forces = ForcesConfig(
        force_patches=["cube"],
        ref_velocity=args.u_inf,
        ref_area=args.D * depth,  # frontal area of the extruded square
        ref_length=args.D,
        moment_centre=[0.0, 0.0, 0.5 * depth],
        force_log_interval=1,  # every step: St comes from the Cl signal
    )

    return FVMConfig(
        case_name=args.case_name,
        time=TimeConfig(
            delta_t=args.initial_dt,
            start_time=0.0,
            end_time=args.end_time,
            write_interval=10**9,  # step-based writing off; time-based below
            write_interval_time=args.write_interval_time,
            adjust_timestep=True,
            max_cfl=args.max_cfl,
            max_delta_t=args.max_dt,
            min_delta_t=1e-5,
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        forces=forces,
        transport=TransportConfig(density=args.rho, nu=nu),
        turbulence=None,  # laminar validation case
        boundaries=[
            BoundaryConfig.inlet("inlet", [args.u_inf, 0.0, 0.0]),
            BoundaryConfig.outlet("outlet", p=0.0),
            # Slip lateral boundaries: 20 D apart (5% blockage), no wall BL.
            BoundaryConfig(
                name="bottom", type_U="slip", type_p="zeroGradient", type_nut="zeroGradient"
            ),
            BoundaryConfig(
                name="top", type_U="slip", type_p="zeroGradient", type_nut="zeroGradient"
            ),
            BoundaryConfig.wall("cube"),
            BoundaryConfig.empty("front"),
            BoundaryConfig.empty("back"),
        ],
        # Small cross-stream component breaks the wake symmetry so shedding
        # starts within a few advective times instead of waiting for round-off.
        initial_U=[args.u_inf, 0.05 * args.u_inf, 0.0],
        initial_p=0.0,
    )


def main():
    parser = argparse.ArgumentParser(description="Square-cylinder von Karman street")
    parser.add_argument("--Re", type=float, default=100.0, help="Reynolds number U*D/nu")
    parser.add_argument("--end-time", type=float, default=120.0, help="End time [s]")
    parser.add_argument("--h", type=float, default=0.0625, help="Core grid spacing (in D units)")
    parser.add_argument("--D", type=float, default=1.0, help="Cylinder side length")
    parser.add_argument("--u-inf", type=float, default=1.0, help="Freestream velocity")
    parser.add_argument("--rho", type=float, default=1.0, help="Density")
    parser.add_argument("--initial-dt", type=float, default=0.02, help="Initial dt [s]")
    parser.add_argument("--max-cfl", type=float, default=0.9, help="Target max Courant")
    parser.add_argument("--max-dt", type=float, default=0.05, help="Max dt cap [s]")
    parser.add_argument(
        "--write-interval-time",
        type=float,
        default=5.0,
        help="Write VTK every N seconds of flow time",
    )
    parser.add_argument("--n-correctors", type=int, default=2, help="PISO correctors")
    parser.add_argument("--n-outer", type=int, default=1, help="PIMPLE outer correctors")
    parser.add_argument("--linear-solver", type=str, default="bicgstab")
    parser.add_argument("--convection-scheme", type=str, default="limitedLinear")
    parser.add_argument("--case-name", type=str, default="cubeFlow")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n--- Mesh Generation (rectilinear, in-memory) ---")
    mesh_data, depth = square_cylinder_mesh(h=args.h, D=args.D)
    print(
        f"  cells: {mesh_data['n_elements']}, core spacing h = {args.h} "
        f"(D/h = {args.D / args.h:.0f}), blockage 5%"
    )

    config = build_config(args, depth)
    solver = Solver(config, case_dir, mesh_data=mesh_data)

    solver.write_vtk()
    while solver.flow_time < config.time.end_time:
        solver.evolve()

    print("\nSimulation completed successfully.")
    print("Reference values at Re = 100 (square cylinder, 5% blockage):")
    print("  St = 0.140-0.150   (Okajima 1982; Sohankar et al. 1998; Sen et al. 2011)")
    print("  mean Cd = 1.45-1.58")


if __name__ == "__main__":
    main()
