#!/usr/bin/env python3
"""Body-fitted FVM reference for the coupled cylinder flow at Re = 150.

One periodic cell spans the cylinder. Distant slip walls and inlet approximate
the unbounded stream that the coupled case models.
"""

import argparse
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

# Startup: the same near-wake vortex as the coupled case
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, streamfunction peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = 0.65  # m, downstream of the cylinder axis

# Domain sizing, Behr et al. (1995, CMAME 123)
DOMAIN = (-16.0, 30.0, -16.0, 16.0)  # m, x and y bounds

# Mesh: octree cells of CELL_SIZE over the coupled FVM domain, as in the coupled case, then
# cells of 2 CELL_SIZE along the wake all the way to the outlet
CELL_SIZE = 0.04  # m
FINE_BOX_MARGIN = 2.4  # D upstream and sideways, 2 D more downstream: the coupled FVM domain
WAKE = (-3.0, DOMAIN[1], -3.0, 3.0)  # m, cells of 2 CELL_SIZE
BACKGROUND = 16  # background cells in units of CELL_SIZE

# Time: a fixed step, as in the coupled case, refined with the cells
TIME_STEP = 0.008  # s, at CELL_SIZE
END_TIME = 100.0  # s

# Forces, wake profiles, flow fields and backups at the intervals of the coupled case
# (setup.py one level up), rounded to whole steps as there
FORCE_INTERVAL = 0.04  # s
PROFILE_INTERVAL = 0.2  # s, transverse wake profiles at x = 2 and 4 m
FIELD_INTERVAL = 0.2  # s
FIELD_START = 40.0  # s, developed shedding
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


def build_case(
    name,
    reynolds_number=REYNOLDS_NUMBER,
    cell_size=CELL_SIZE,
    fine_box_margin=FINE_BOX_MARGIN,
    end_time=END_TIME,
) -> fvm.FVMCase:
    """Return the FVM case of the cylinder flow, written under ``solution/<name>``.

    Cells of ``cell_size`` fill the coupled FVM domain of the box margin ``fine_box_margin``.
    """
    kinematic_viscosity = FREESTREAM_VELOCITY[0] * DIAMETER / reynolds_number  # m^2/s
    half_span = 0.5 * SPAN
    time_step = cell_size / CELL_SIZE * TIME_STEP
    margin = fine_box_margin * DIAMETER
    fine_box = (-margin, margin + 2.0 * DIAMETER, -margin, margin)

    # Output times are whole steps (a fixed step is shortened to land on any other output time),
    # written with at most 3 decimals: every step of allrun.sh has 3 decimals, so are its multiples
    def on_steps(interval):
        return round(max(1, round(interval / time_step)) * time_step, 3)

    # Requested sizes sit between octree levels, which then select 1 and 2 CELL_SIZE
    request = 4.0 * cell_size / 3.0
    # The meshed slab holds the whole cylinder STL (|z| <= 0.6 m): a slab that cuts through it
    # (half-depth 0.5 m) stalls the mesh optimisation
    slab = 16 * request

    # Mesh a slab around the long cylinder, then keep one periodic cell across the span
    patches = msh.BoxPatches(
        xmin="inlet", xmax="outlet", ymin="ymin", ymax="ymax", zmin="zmin", zmax="zmax"
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=msh.CartesianMesher(
            domain=msh.BoxDomain(bounds=(*DOMAIN, -slab, slab), patches=patches),
            surfaces=(msh.STLSurface(CASE_DIR / "assets/cylinder_long.stl", patch="cylinder"),),
            max_cell_size=BACKGROUND * cell_size,
            refinements=(
                msh.BoxRefinement("coupledDomain", (*fine_box, -slab, slab), request),
                msh.BoxRefinement("wake", (*WAKE, -slab, slab), 2 * request),
            ),
            patch_refinements=(msh.PatchRefinement("cylinder", request),),
            surface_may_cross_domain_boundary=True,
        ),
        domain=msh.BoxDomain(bounds=(*DOMAIN, -half_span, half_span), patches=patches),
        levels=(-half_span, half_span),
    )

    profiles = fvm.EveryTime(on_steps(PROFILE_INTERVAL))
    samplers = (
        # Drag and lift coefficients per unit span
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=FREESTREAM_VELOCITY[0],
            reference_area=DIAMETER * SPAN,
            reference_length=DIAMETER,
            schedule=fvm.EveryTime(on_steps(FORCE_INTERVAL)),
        ),
        # Recirculation region on the wake centreline
        fvm.LineSampler(
            (0.6, 0.0, 0.0),
            (1.4, 0.0, 0.0),
            n_points=11,
            n_neighbours=12,
            reconstruction="affine",
            file_name="centreline",
            schedule=profiles,
        ),
        # Transverse wake profiles
        *(
            fvm.LineSampler(
                (x, -2.0, 0.0),
                (x, 2.0, 0.0),
                n_points=41,
                n_neighbours=12,
                reconstruction="affine",
                file_name=f"transverse_x{x:g}",
                schedule=profiles,
            )
            for x in (2.0, 4.0)
        ),
    )
    numerics = fvm.Numerics(
        time_step_size=time_step,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
        schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        linear=fvm.LinearSolverConfig(pressure_tolerance=1.0e-7, pressure_relative_tolerance=0.005),
        pimple=fvm.PimpleConfig(
            n_outer_correctors=2, velocity_relaxation=0.7, pressure_relaxation=0.3
        ),
        execution=fvm.ComputeConfig(operator_backend="numba"),
    )
    return fvm.FVMCase(
        name=name,
        directory=CASE_DIR,
        variant=name,
        numerics=numerics,
        initial_conditions=fvm.InitialFields(velocity=FREESTREAM_VELOCITY),
        run=fvm.RunPlan(end_time=end_time),
        output=fvm.Output(
            schedule=fvm.EveryTime(on_steps(FIELD_INTERVAL), start_time=on_steps(FIELD_START)),
            compression="lz4",
            ghost_layers=0,
        ),
        samplers=fvm.Samplers(samplers),
        backup=fvm.Backup(schedule=fvm.EveryTime(on_steps(BACKUP_INTERVAL))),
        logging=fvm.LoggingConfig(detailed_timing=True),  # wall time per step
        mesh=mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.cyclic("zmin", "zmax"),
            fvm.BoundaryConfig.cyclic("zmax", "zmin"),
            fvm.BoundaryConfig.wall("cylinder"),
        ),
    )


if __name__ == "__main__":
    # One case per call; allrun.sh lists the cases
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="case name: solution/<name>, samples/<name>")
    parser.add_argument(
        "--reynolds-number", type=float, default=REYNOLDS_NUMBER, help="Reynolds number U D / nu"
    )
    parser.add_argument(
        "--cell-size", type=float, default=CELL_SIZE, help="cell size near the body [m]"
    )
    parser.add_argument(
        "--fine-box-margin",
        type=float,
        default=FINE_BOX_MARGIN,
        help="box margin of the coupled FVM domain that the cells of --cell-size fill [D]",
    )
    parser.add_argument("--end-time", type=float, default=END_TIME, help="end time [s]")
    arguments = parser.parse_args()

    case = build_case(
        arguments.name,
        arguments.reynolds_number,
        arguments.cell_size,
        arguments.fine_box_margin,
        arguments.end_time,
    )
    with fvm.FVMSolver(case) as solver:
        cell_centres = solver.geo_data["cell_centre"][: solver.mesh_data["n_cells"]]
        solver.set_initial_velocity(initial_velocity(cell_centres))
        solver.run(start_from="latest")
