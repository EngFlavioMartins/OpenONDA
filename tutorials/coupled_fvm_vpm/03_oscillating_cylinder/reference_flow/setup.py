#!/usr/bin/env python3
"""Body-fitted FVM reference for the coupled heaving cylinder at Re = 150.

The whole mesh heaves with the cylinder. The distant inlet and side
boundaries hold the undisturbed stream, which the moving boundaries pass
through, as in the unbounded stream that the coupled case models.
"""

from pathlib import Path

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

# Prescribed transverse motion, as in the coupled case
AMPLITUDE = 0.2 * DIAMETER  # m
FREQUENCY = 0.18  # Hz

# Domain: inlet and sides far enough away that the body barely disturbs them
DOMAIN = (-40.0, 60.0, -60.0, 60.0)  # m, x and y bounds

# Mesh: octree cells of 0.04 m near the body, coarsening by two in each wake box
CELL_SIZE = 0.04  # m
NEAR_BODY = (-1.5, 1.5, -1.5, 1.5)  # m, cells of CELL_SIZE
NEAR_WAKE = (-2.0, 6.0, -2.0, 2.0)  # m, cells of 2 CELL_SIZE
WAKE = (-2.5, 12.0, -2.5, 2.5)  # m, cells of 4 CELL_SIZE
BACKGROUND = 16  # background cells in units of CELL_SIZE

# Time
TIME_STEP = 0.008  # s
END_TIME = 50.0  # s

# Observations, matching the coupled case
FORCE_INTERVAL = 0.04  # s
VOLUME_INTERVAL = 5.0  # s
BACKUP_INTERVAL = 1.0  # s


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
        case_name="reference_oscillating_cylinder",
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
            # Slip walls would push the fluid along as they heave; the sides
            # hold the inertial-frame stream instead.
            fvm.BoundaryConfig.inlet("ymin", list(FREESTREAM_VELOCITY)),
            fvm.BoundaryConfig.inlet("ymax", list(FREESTREAM_VELOCITY)),
            fvm.BoundaryConfig.cyclic("zmin", "zmax"),
            fvm.BoundaryConfig.cyclic("zmax", "zmin"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        initial_velocity=list(FREESTREAM_VELOCITY),
        mesh_motion=fvm.MeshMotion(
            fvm.HeavingMotion(AMPLITUDE, FREQUENCY, direction=(0.0, 1.0, 0.0))
        ),
        samplers=(
            # Drag and lift coefficients per unit span
            fvm.ForceSampler(
                patch_names=["cylinder"],
                reference_velocity=FREESTREAM_VELOCITY[0],
                reference_area=DIAMETER * SPAN,
                reference_length=DIAMETER,
                schedule=fvm.RunSchedule(every_time=FORCE_INTERVAL),
            ),
        ),
    )
    return flow, mesh


if __name__ == "__main__":
    flow, mesh = build_case()
    with fvm.create_fvm_solver(flow, case_dir=CASE_DIR, mesh=mesh) as solver:
        solver.run(start_from="latest")
