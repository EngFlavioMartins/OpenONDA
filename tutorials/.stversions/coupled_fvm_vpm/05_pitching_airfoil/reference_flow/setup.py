#!/usr/bin/env python3
"""Body-fitted FVM reference for the coupled pitching NACA 0012 airfoil at Re = 1000.

The whole mesh turns with the airfoil about its quarter chord. The distant
inlet and side boundaries hold the undisturbed stream, which the turning
boundaries pass through, as in the unbounded stream that the coupled case
models.
"""

import argparse
from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

from assets.naca_surface import write_wing_stl

CASE_DIR = Path(__file__).resolve().parent

# Airfoil, centred on the origin, the leading edge at x = -c/2
CHORD = 1.0  # m
SPAN = 1.0  # m
NACA_CODE = "0012"
N_SECTION_POINTS = 161  # per surface
AIRFOIL_STL = CASE_DIR / "assets/naca0012_long.stl"

# Flow
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)  # m/s
REYNOLDS_NUMBER = 1000.0
DENSITY = 1.0  # kg/m^3
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY[0] * CHORD / REYNOLDS_NUMBER  # m^2/s

# Prescribed pitch, nose up for positive angles
PITCH_AMPLITUDE = 10.0  # degrees
FREQUENCY = 0.2  # Hz
PITCH_AXIS = (-0.25 * CHORD, 0.0, 0.0)  # m, the quarter chord

# Outer FVM domain, in the frame of the airfoil
DOMAIN = (-20.0, 40.0, -25.0, 25.0)  # m, x and y bounds

# Mesh refinement from the airfoil into the wake; the near-body cells hold
# the coupled FVM box at every pitch angle
CELL_SIZE = 0.025  # m
NEAR_BODY = (-2.0, 3.0, -1.8, 1.8)  # m, cells of CELL_SIZE
NEAR_WAKE = (-2.5, 7.0, -2.5, 2.5)  # m, cells of 2 CELL_SIZE
WAKE = (-3.0, 14.0, -3.5, 3.5)  # m, cells of 4 CELL_SIZE
BACKGROUND = 16  # background cells in units of CELL_SIZE

# Time integration
TIME_STEP = 0.005  # s
END_TIME = 30.0  # s

# Output
FORCE_INTERVAL = 0.025  # s
PROFILE_INTERVAL = 0.2  # s, transverse wake profiles at x = 2 and 4 m
FIELD_INTERVAL = 0.2  # s
FIELD_START = 15.0  # s, periodic wake
BACKUP_INTERVAL = 1.0  # s


def build_case(end_time: float = END_TIME) -> fvm.FVMCase:
    """Return the FVM case of the pitching-airfoil flow."""
    half_span = 0.5 * SPAN

    # Requested sizes select octree levels of one, two and four cell widths.
    request = 4.0 * CELL_SIZE / 3.0
    slab = 16 * request

    # Mesh a slab that the airfoil crosses and retain one periodic spanwise cell.
    write_wing_stl(AIRFOIL_STL, NACA_CODE, CHORD, 4.0 * slab, N_SECTION_POINTS, 1, 0.0)
    patches = msh.BoxPatches(
        xmin="inlet", xmax="outlet", ymin="ymin", ymax="ymax", zmin="zmin", zmax="zmax"
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=msh.CartesianMesher(
            domain=msh.BoxDomain(bounds=(*DOMAIN, -slab, slab), patches=patches),
            surfaces=(msh.STLSurface(AIRFOIL_STL, patch="airfoil"),),
            max_cell_size=BACKGROUND * CELL_SIZE,
            refinements=(
                msh.BoxRefinement("nearBody", (*NEAR_BODY, -slab, slab), request),
                msh.BoxRefinement("nearWake", (*NEAR_WAKE, -slab, slab), 2 * request),
                msh.BoxRefinement("wake", (*WAKE, -slab, slab), 4 * request),
            ),
            patch_refinements=(msh.PatchRefinement("airfoil", request),),
            surface_may_cross_domain_boundary=True,
        ),
        domain=msh.BoxDomain(bounds=(*DOMAIN, -half_span, half_span), patches=patches),
        levels=(-half_span, half_span),
    )

    samplers = (
        # Drag, lift and quarter-chord moment coefficients per unit span.
        fvm.ForceSampler(
            patch_names=["airfoil"],
            reference_velocity=FREESTREAM_VELOCITY[0],
            reference_area=CHORD * SPAN,
            reference_length=CHORD,
            moment_centre=PITCH_AXIS,
            schedule=fvm.EveryTime(FORCE_INTERVAL),
        ),
        # Transverse wake profiles at fixed inertial positions.
        *(
            fvm.LineSampler(
                (x, -2.0, 0.0),
                (x, 2.0, 0.0),
                n_points=41,
                n_neighbours=12,
                reconstruction="affine",
                file_name=f"transverse_x{x:g}",
                schedule=fvm.EveryTime(PROFILE_INTERVAL),
            )
            for x in (2.0, 4.0)
        ),
    )
    numerics = fvm.Numerics(
        time_step_size=TIME_STEP,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
        schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        linear=fvm.LinearSolverConfig(pressure_tolerance=1.0e-7, pressure_relative_tolerance=0.005),
        # Three outer passes converge the steps of the accelerating mesh
        pimple=fvm.PimpleConfig(
            n_outer_correctors=3, velocity_relaxation=0.7, pressure_relaxation=0.3
        ),
        execution=fvm.ComputeConfig(operator_backend="numba"),
    )
    return fvm.FVMCase(
        name="reference_pitching_airfoil",
        directory=CASE_DIR,
        numerics=numerics,
        initial_conditions=fvm.InitialFields(velocity=FREESTREAM_VELOCITY),
        run=fvm.RunPlan(end_time=end_time),
        output=fvm.Output(schedule=fvm.EveryTime(FIELD_INTERVAL, start_time=FIELD_START)),
        samplers=fvm.Samplers(samplers),
        backup=fvm.Backup(schedule=fvm.EveryTime(BACKUP_INTERVAL)),
        logging=fvm.LoggingConfig(detailed_timing=True),  # wall time per step
        mesh=mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            # Slip walls would drag the fluid round as they turn; the sides
            # hold the inertial-frame stream instead.
            fvm.BoundaryConfig.inlet("ymin", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.inlet("ymax", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.cyclic("zmin", "zmax"),
            fvm.BoundaryConfig.cyclic("zmax", "zmin"),
            fvm.BoundaryConfig.wall("airfoil"),
        ),
        # A positive angle about -z lifts the nose
        mesh_motion=fvm.MeshMotion(
            fvm.PitchingMotion(
                PITCH_AMPLITUDE,
                FREQUENCY,
                rotation_axis=(0.0, 0.0, -1.0),
                rotation_centre=PITCH_AXIS,
            )
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float, default=END_TIME, help="end time [s]")
    arguments = parser.parse_args()
    with fvm.FVMSolver(build_case(arguments.end_time)) as solver:
        solver.run(start_from="latest")
