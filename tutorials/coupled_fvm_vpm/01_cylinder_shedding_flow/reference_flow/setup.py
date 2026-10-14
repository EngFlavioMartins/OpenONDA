#!/usr/bin/env python3
"""Body-fitted FVM reference for the coupled cylinder flow at Re = 150.

One periodic cell spans the cylinder. Distant slip walls and inlet approximate
the unbounded stream that the coupled case models.
"""

from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

CASE_DIR = Path(__file__).resolve().parent

# Flow
DIAMETER = 1.0  # m
SPAN = 1.0  # m
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)  # m/s
REYNOLDS_NUMBER = 150.0
DENSITY = 1.0  # kg/m^3
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY[0] * DIAMETER / REYNOLDS_NUMBER  # m^2/s

# Startup: the same near-wake vortex as the coupled case
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, streamfunction peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = 0.65  # m, downstream of the cylinder axis

# Domain: walls and inlet far enough away that blockage stays small
DOMAIN = (-40.0, 60.0, -60.0, 60.0)  # m, x and y bounds

# Mesh: octree cells of 0.04 m near the body, coarsening by two in each wake box
CELL_SIZE = 0.04  # m
NEAR_BODY = (-1.5, 1.5, -1.5, 1.5)  # m, cells of CELL_SIZE
NEAR_WAKE = (-2.0, 6.0, -2.0, 2.0)  # m, cells of 2 CELL_SIZE
WAKE = (-2.5, 12.0, -2.5, 2.5)  # m, cells of 4 CELL_SIZE
BACKGROUND = 16  # background cells in units of CELL_SIZE

# Time
TIME_STEP = 0.008  # s
END_TIME = 100.0  # s

# Observations, matching the coupled case
FORCE_INTERVAL = 0.04  # s
PROFILE_INTERVAL = 0.2  # s, transverse wake profiles at x = 2 and 4 m
VOLUME_INTERVAL = 4.0  # s
BACKUP_INTERVAL = 1.0  # s


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


def build_case():
    """Return the FVM setup and mesh."""
    half_span = 0.5 * SPAN

    # Requested sizes sit between octree levels, which then select 1, 2 and 4 CELL_SIZE
    request = 4.0 * CELL_SIZE / 3.0
    slab = 16 * request

    # Mesh a slab around the long cylinder, then keep one periodic cell across the span
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
            patch_refinements=(msh.PatchRefinement("cylinder", request),),
            surface_may_cross_domain_boundary=True,
        ),
        domain=msh.BoxDomain(bounds=(*DOMAIN, -half_span, half_span), patches=patches),
        levels=(-half_span, half_span),
    )

    flow = fvm.FVMSetup(
        case_name="reference_cylinder_flow",
        execution=fvm.ComputeConfig(operator_backend="numba"),
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP,
            end_time=END_TIME,
            output_schedule=fvm.RunSchedule(every_time=VOLUME_INTERVAL),
        ),
        backup=fvm.BackupConfig(
            schedule=fvm.RunSchedule(every_time=BACKUP_INTERVAL), write_at_end=True
        ),
        schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        linear=fvm.LinearSolverConfig(
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=2, velocity_relaxation=0.7, pressure_relaxation=0.3
        ),
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
        turbulence=fvm.TurbulenceConfig.none(),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", list(FREESTREAM_VELOCITY)),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.cyclic("zmin", "zmax"),
            fvm.BoundaryConfig.cyclic("zmax", "zmin"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        initial_velocity=list(FREESTREAM_VELOCITY),
        samplers=(
            # Drag and lift coefficients per unit span
            fvm.ForceSampler(
                patch_names=["cylinder"],
                reference_velocity=FREESTREAM_VELOCITY[0],
                reference_area=DIAMETER * SPAN,
                reference_length=DIAMETER,
                schedule=fvm.RunSchedule(every_time=FORCE_INTERVAL),
            ),
            # Recirculation region on the wake centreline
            fvm.LineSampler(
                start=[0.6, 0.0, 0.0],
                end=[1.4, 0.0, 0.0],
                n_points=11,
                k=12,
                reconstruction="affine",
                file_name="centreline",
                schedule=fvm.RunSchedule(every_time=PROFILE_INTERVAL),
            ),
            # Transverse wake profiles
            *(
                fvm.LineSampler(
                    start=[x, -2.0, 0.0],
                    end=[x, 2.0, 0.0],
                    n_points=41,
                    k=12,
                    reconstruction="affine",
                    file_name=f"transverse_x{x:g}",
                    schedule=fvm.RunSchedule(every_time=PROFILE_INTERVAL),
                )
                for x in (2.0, 4.0)
            ),
        ),
    )
    return flow, mesh


if __name__ == "__main__":
    flow, mesh = build_case()
    with fvm.create_fvm_solver(flow, case_dir=CASE_DIR, mesh=mesh) as solver:
        solver.run(start_from="latest", initial_velocity=initial_velocity)
