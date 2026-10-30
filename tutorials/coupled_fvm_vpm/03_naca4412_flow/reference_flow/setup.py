#!/usr/bin/env python3
"""Body-fitted FVM reference for the coupled NACA 4412 wing at 10 degrees and Re = 1000.

The same pitched wing as in the coupled case sits in a large box. The inlet and
the freestream faces carry the stream along x; distant slip faces bound the span.
"""

import math
from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

from assets.naca_surface import write_wing_stl

CASE_DIR = Path(__file__).resolve().parent

# Wing geometry
NACA_CODE = "4412"
CHORD = 1.0  # m
SPAN = 5.0  # m
N_SECTION_POINTS = 161
WING_STL = CASE_DIR / "assets/naca4412_wing.stl"

# Flow
ANGLE_OF_ATTACK = 10.0  # degrees
FREESTREAM_SPEED = 1.0  # m/s
REYNOLDS_NUMBER = 1000.0
DENSITY = 1.0  # kg/m^3
KINEMATIC_VISCOSITY = FREESTREAM_SPEED * CHORD / REYNOLDS_NUMBER  # m^2/s
ALPHA = math.radians(ANGLE_OF_ATTACK)
FREESTREAM_VELOCITY = (FREESTREAM_SPEED, 0.0, 0.0)

# Outer FVM domain
DOMAIN = (-10.24, 30.72, -12.8, 12.8, -12.8, 12.8)  # m

# Mesh refinement from the wing into the wake
CELL_SIZE = 0.04  # m
NEAR_BODY = (-1.6, 2.72, -1.28, 1.24, -3.6, 3.6)  # m, finest spacing
NEAR_WAKE = (-2.0, 6.0, -1.84, 1.84, -4.0, 4.0)  # m, cells of 2 CELL_SIZE
WAKE = (-2.4, 14.0, -3.2, 3.2, -4.8, 4.8)  # m, cells of 4 CELL_SIZE
BACKGROUND = 16  # background cells in units of CELL_SIZE

# Time integration
TIME_STEP = 0.008  # s
STATISTICS_START = 12.0  # s
END_TIME = 24.0  # s
CORES = 4

# Output
FORCE_INTERVAL = 0.032  # s
PROFILE_INTERVAL = 0.192  # s
SLICE_INTERVAL = 0.192  # s
FIELD_INTERVAL = 1.92  # s
BACKUP_INTERVAL = 0.96  # s
WAKE_STATIONS = (2.0, 5.0)  # m
PROFILE_SPACING = 2.0 * CELL_SIZE  # m
# Trailing-edge height and the force moment centre
WAKE_HEIGHT = -0.5 * CHORD * math.sin(ALPHA)  # m
QUARTER_CHORD = (-0.25 * CHORD * math.cos(ALPHA), 0.25 * CHORD * math.sin(ALPHA), 0.0)  # m
SLICE_Z0 = (-1.6, 8.0, -2.4, 2.4)  # m, x and y bounds of the mid-span plane
SLICE_CROSS = (-2.4, 2.4, -4.0, 4.0)  # m, y and z bounds of the cross-flow planes


write_wing_stl(WING_STL, NACA_CODE, CHORD, SPAN, N_SECTION_POINTS, 1, ANGLE_OF_ATTACK)

MESH = msh.CartesianMesher(
    domain=msh.BoxDomain(
        bounds=DOMAIN,
        patches=msh.BoxPatches(
            xmin="inlet", xmax="outlet", ymin="ymin", ymax="ymax", zmin="zmin", zmax="zmax"
        ),
    ),
    surfaces=(msh.STLSurface(WING_STL, patch="wing"),),
    max_cell_size=BACKGROUND * CELL_SIZE,
    cell_size_anchor=CELL_SIZE,
    refinements=(
        msh.BoxRefinement("nearBody", NEAR_BODY, CELL_SIZE),
        msh.BoxRefinement("nearWake", NEAR_WAKE, 2 * CELL_SIZE),
        msh.BoxRefinement("wake", WAKE, 4 * CELL_SIZE),
    ),
    patch_refinements=(msh.PatchRefinement("wing", CELL_SIZE),),
)


def probe(start, end, name):
    """Velocity and vorticity along a line, every profile interval."""
    return fvm.LineSampler(
        start,
        end,
        spacing=PROFILE_SPACING,
        n_neighbours=12,
        reconstruction="affine",
        file_name=name,
        schedule=fvm.EveryTime(PROFILE_INTERVAL),
    )


def plane(point, normal, bounds, name):
    """Velocity and vorticity on a plane, every slice interval of the statistics window."""
    return fvm.SurfaceSampler(
        point,
        normal,
        bounds,
        PROFILE_SPACING,
        n_neighbours=12,
        reconstruction="affine",
        file_name=name,
        schedule=fvm.EveryTime(SLICE_INTERVAL, start_time=STATISTICS_START),
        initial=False,
    )


SAMPLERS = (
    # Drag and lift coefficients on the planform area c b; the moment about the quarter chord
    fvm.ForceSampler(
        patch_names=["wing"],
        reference_velocity=FREESTREAM_SPEED,
        reference_area=CHORD * SPAN,
        reference_length=CHORD,
        moment_centre=QUARTER_CHORD,
        schedule=fvm.EveryTime(FORCE_INTERVAL),
    ),
    *(
        probe((x, SLICE_CROSS[0], 0.0), (x, SLICE_CROSS[1], 0.0), f"transverse_x{x:g}")
        for x in WAKE_STATIONS
    ),
    *(
        probe(
            (x, WAKE_HEIGHT, SLICE_CROSS[2]),
            (x, WAKE_HEIGHT, SLICE_CROSS[3]),
            f"spanwise_x{x:g}",
        )
        for x in WAKE_STATIONS
    ),
    plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), SLICE_Z0, "slice_z0"),
    *(plane((x, 0.0, 0.0), (1.0, 0.0, 0.0), SLICE_CROSS, f"slice_x{x:g}") for x in WAKE_STATIONS),
)

CASE = fvm.FVMCase(
    name="reference_naca4412_flow",
    directory=CASE_DIR,
    numerics=fvm.Numerics(
        time_step_size=TIME_STEP,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="linear_upwind", gradient_scheme="gauss", time_scheme="backward"
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg", pressure_tolerance=1.0e-7, pressure_relative_tolerance=0.005
        ),
        pimple=fvm.PimpleConfig(
            n_outer_correctors=2,
            n_nonorthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        turbulence=fvm.TurbulenceConfig.smagorinsky(),
        execution=fvm.ComputeConfig(operator_backend="numba"),
    ),
    initial_conditions=fvm.InitialFields(velocity=FREESTREAM_VELOCITY),
    run=fvm.RunPlan(end_time=END_TIME),
    output=fvm.Output(
        schedule=fvm.EveryTime(FIELD_INTERVAL, start_time=STATISTICS_START),
        compression="lz4",
        ghost_layers=0,
    ),
    samplers=fvm.Samplers(SAMPLERS),
    backup=fvm.Backup(schedule=fvm.EveryTime(BACKUP_INTERVAL)),
    logging=fvm.LoggingConfig(detailed_timing=True),  # wall time per step
    mesh=MESH,
    boundaries=(
        fvm.BoundaryConfig.inlet("inlet", FREESTREAM_VELOCITY),
        fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
        # The stream runs along the lateral faces
        fvm.BoundaryConfig.freestream("ymin", FREESTREAM_VELOCITY),
        fvm.BoundaryConfig.freestream("ymax", FREESTREAM_VELOCITY),
        fvm.BoundaryConfig.slip("zmin"),
        fvm.BoundaryConfig.slip("zmax"),
        fvm.BoundaryConfig.wall("wing"),
    ),
    cores=CORES,
)


if __name__ == "__main__":
    with fvm.FVMSolver(CASE) as solver:
        solver.run(start_from="latest")
