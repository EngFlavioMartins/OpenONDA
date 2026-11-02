#!/usr/bin/env python3
"""Body-fitted FVM reference for the coupled elastically mounted cylinder.

The whole mesh rides on the springs of the cylinder. The distant inlet and
side boundaries hold the undisturbed stream, which the moving boundaries pass
through, as in the unbounded stream that the coupled case models. The
structure, Reynolds numbers and outputs are those of the coupled case.
"""

import argparse
import math
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

CASE_DIR = Path(__file__).resolve().parent

# Flow
DIAMETER = 1.0  # m
SPAN = 1.0  # m
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)  # m/s
DENSITY = 1.0  # kg/m^3
REYNOLDS_PER_REDUCED_VELOCITY = 250.0  # the low-Reynolds set of Martins and Avila (2019)
REDUCED_VELOCITIES = (4.0, 6.0, 8.0)  # the runs of allrun.sh

# Structure, as the coupled case
MASS_RATIO = 2.4  # 4 m / (rho pi D^2 L)
ADDED_MASS_COEFFICIENT = 1.0  # C_A
DAMPING_RATIO = 0.005  # b / (2 sqrt(k m))

# Startup vortex: breaks the wake symmetry and starts shedding
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, streamfunction peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = 0.65  # m, downstream of the cylinder axis

# Outer FVM domain
DOMAIN = (-40.0, 60.0, -60.0, 60.0)  # m, x and y bounds

# Mesh refinement from the cylinder into the wake
CELL_SIZE = 0.04  # m
WALL_CELL_SIZE = 0.02  # m, one octree level finer at the cylinder wall
NEAR_BODY = (-1.5, 1.5, -1.5, 1.5)  # m, cells of CELL_SIZE
NEAR_WAKE = (-2.0, 6.0, -2.0, 2.0)  # m, cells of 2 CELL_SIZE
WAKE = (-2.5, 12.0, -2.5, 2.5)  # m, cells of 4 CELL_SIZE
BACKGROUND = 16  # background cells in units of CELL_SIZE

# Time integration
TIME_STEP = 0.0025  # s
END_PERIODS = 25  # natural periods in water

# Output
FORCE_INTERVAL = 0.04  # s, forces and cylinder motion
FIELD_INTERVAL = 0.2  # s
FIELD_PERIODS = 2  # natural periods of fields before the end
BACKUP_INTERVAL = 1.0  # s


def structure(reduced_velocity, span):
    """Mass, stiffness and damping in kg, N/m and N s/m of the modelled span."""
    displaced_mass = DENSITY * math.pi * DIAMETER**2 / 4.0 * span
    mass = MASS_RATIO * displaced_mass
    natural_frequency = FREESTREAM_VELOCITY[0] / (reduced_velocity * DIAMETER)  # Hz, in water
    stiffness = (2.0 * math.pi * natural_frequency) ** 2 * (
        mass + ADDED_MASS_COEFFICIENT * displaced_mass
    )
    damping = 2.0 * DAMPING_RATIO * math.sqrt(stiffness * mass)
    return mass, stiffness, damping


def initial_velocity(position):
    """Freestream plus the curl of psi = A (1 - r^2/R^2)^4 behind the cylinder."""
    x = position[:, 0] - DISTURBANCE_CENTRE
    y = position[:, 1]
    support = np.maximum(1.0 - (x**2 + y**2) / DISTURBANCE_RADIUS**2, 0.0)
    swirl = 8.0 * DISTURBANCE_AMPLITUDE / DISTURBANCE_RADIUS**2 * support**3

    velocity = np.tile(FREESTREAM_VELOCITY, (len(position), 1))
    velocity[:, 0] -= swirl * y
    velocity[:, 1] += swirl * x
    return velocity


def build_case(reduced_velocity, end_time=None) -> fvm.FVMCase:
    """Return the FVM case of one reduced velocity."""
    reynolds_number = REYNOLDS_PER_REDUCED_VELOCITY * reduced_velocity
    kinematic_viscosity = FREESTREAM_VELOCITY[0] * DIAMETER / reynolds_number  # m^2/s
    natural_period = reduced_velocity * DIAMETER / FREESTREAM_VELOCITY[0]  # s, in water
    if end_time is None:
        end_time = END_PERIODS * natural_period
    field_start = max(0.0, end_time - FIELD_PERIODS * natural_period)
    half_span = 0.5 * SPAN
    mass, stiffness, damping = structure(reduced_velocity, SPAN)

    # Requested sizes select octree levels of one, two and four cell widths.
    request = 4.0 * CELL_SIZE / 3.0
    slab = 16 * request

    # Mesh a slab around the cylinder and retain one periodic spanwise cell.
    patches = msh.BoxPatches(
        xmin="inlet", xmax="outlet", ymin="ymin", ymax="ymax", zmin="zmin", zmax="zmax"
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=msh.CartesianMesher(
            domain=msh.BoxDomain(bounds=(*DOMAIN, -slab, slab), patches=patches),
            surfaces=(msh.STLSurface(CASE_DIR / "assets/cylinder_long.stl", patch="cylinder"),),
            max_cell_size=BACKGROUND * CELL_SIZE,
            refinements=(
                msh.BoxRefinement("nearBody", (*NEAR_BODY, -slab, slab), request),
                msh.BoxRefinement("nearWake", (*NEAR_WAKE, -slab, slab), 2 * request),
                msh.BoxRefinement("wake", (*WAKE, -slab, slab), 4 * request),
            ),
            patch_refinements=(
                msh.PatchRefinement("cylinder", request * WALL_CELL_SIZE / CELL_SIZE),
            ),
            surface_may_cross_domain_boundary=True,
        ),
        domain=msh.BoxDomain(bounds=(*DOMAIN, -half_span, half_span), patches=patches),
        levels=(-half_span, half_span),
    )

    samplers = (
        # Drag and lift coefficients per unit span, and the power of the fluid on the body.
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=FREESTREAM_VELOCITY[0],
            reference_area=DIAMETER * SPAN,
            reference_length=DIAMETER,
            schedule=fvm.EveryTime(FORCE_INTERVAL),
        ),
        # Position and velocity of the cylinder axis.
        fvm.MotionSampler(schedule=fvm.EveryTime(FORCE_INTERVAL)),
    )
    numerics = fvm.Numerics(
        time_step_size=TIME_STEP,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
        schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        linear=fvm.LinearSolverConfig(pressure_tolerance=1.0e-7, pressure_relative_tolerance=0.005),
        pimple=fvm.PimpleConfig(
            n_outer_correctors=3, velocity_relaxation=0.7, pressure_relaxation=0.3
        ),
        execution=fvm.ComputeConfig(operator_backend="numba"),
    )
    return fvm.FVMCase(
        name="reference_elastic_cylinder",
        directory=CASE_DIR,
        variant=f"vstar_{reduced_velocity:g}",
        numerics=numerics,
        initial_conditions=fvm.InitialFields(velocity=FREESTREAM_VELOCITY),
        run=fvm.RunPlan(end_time=end_time),
        output=fvm.Output(schedule=fvm.EveryTime(FIELD_INTERVAL, start_time=field_start)),
        samplers=fvm.Samplers(samplers),
        backup=fvm.Backup(schedule=fvm.EveryTime(BACKUP_INTERVAL)),
        logging=fvm.LoggingConfig(detailed_timing=True),  # wall time per step
        mesh=mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            # Slip walls would push the fluid along as they move; the sides
            # hold the inertial-frame stream instead.
            fvm.BoundaryConfig.inlet("ymin", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.inlet("ymax", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.cyclic("zmin", "zmax"),
            fvm.BoundaryConfig.cyclic("zmax", "zmin"),
            fvm.BoundaryConfig.wall("cylinder"),
        ),
        mesh_motion=fvm.MeshMotion(
            fvm.ElasticMount(
                mass=mass,
                stiffness=stiffness,
                damping=damping,
                directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                patch_names=("cylinder",),
            )
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduced-velocity",
        type=float,
        default=REDUCED_VELOCITIES[1],
        help="V* = U / (f_n D); the Reynolds number is 250 V*",
    )
    parser.add_argument(
        "--end-time", type=float, default=None, help="end time [s]; default 25 natural periods"
    )
    arguments = parser.parse_args()
    with fvm.FVMSolver(build_case(arguments.reduced_velocity, arguments.end_time)) as solver:
        # The startup vortex behind the cylinder; a restart replaces it with the backup
        cell_centres = solver.geo_data["cell_centre"][: solver.mesh_data["n_cells"]]
        solver.set_initial_velocity(initial_velocity(cell_centres))
        solver.run(start_from="latest")
