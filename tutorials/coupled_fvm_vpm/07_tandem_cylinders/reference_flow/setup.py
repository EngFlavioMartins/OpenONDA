#!/usr/bin/env python3
"""Body-fitted FVM reference for the coupled flow past two cylinders in tandem at Re = 150.

One mesh holds both cylinders, 8 D apart along the stream, on the domain and
boundaries of the case 01 reference: one periodic cell spans the cylinders,
distant slip walls and an inlet approximate the unbounded stream. The cells of
the coupled case fill both coupled FVM domains and the gap between them.
"""

import argparse
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

from assets.cylinder_surface import CYLINDER_STL, write_cylinder_stl

CASE_DIR = Path(__file__).resolve().parent

# Flow
DIAMETER = 1.0  # m
SPAN = 1.0  # m
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)  # m/s
REYNOLDS_NUMBER = 150.0
DENSITY = 1.0  # kg/m^3

# Cylinders: centres along the stream, as the coupled case
CYLINDERS = (("upstream", (0.0, 0.0)), ("downstream", (8.0, 0.0)))  # name, centre (x, y) [m]

# Startup: the same near-wake vortex behind each cylinder as the coupled case
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, streamfunction peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = 0.65  # m, downstream of the cylinder axis

# Domain and boundaries of the case 01 reference, Behr et al. (1995, CMAME 123)
DOMAIN = (-16.0, 30.0, -16.0, 16.0)  # m, x and y bounds

# Mesh: octree cells of CELL_SIZE over both coupled FVM domains and the gap between them, then
# cells of 2 CELL_SIZE along the wake all the way to the outlet
CELL_SIZE = 0.04  # m
FINE_BOX_MARGIN = 2.4  # D upstream and sideways, 2 D more downstream: the coupled FVM domains
WAKE = (-3.0, DOMAIN[1], -3.0, 3.0)  # m, cells of 2 CELL_SIZE
BACKGROUND = 16  # background cells in units of CELL_SIZE

# Time: the fixed step of the coupled case
TIME_STEP = 0.008  # s
END_TIME = 200.0  # s

# Forces, wake profiles, flow fields and backups at the intervals of the coupled case
FORCE_INTERVAL = 0.04  # s
PROFILE_INTERVAL = 0.12  # s, transverse lines 2 and 4 D behind each cylinder
FIELD_INTERVAL = 0.2  # s
FIELD_START = 185.0  # s
BACKUP_INTERVAL = 1.0  # s


def initial_velocity(position):
    """Freestream plus the curl of psi = A (1 - r^2/R^2)^4 behind each cylinder."""
    velocity = np.tile(FREESTREAM_VELOCITY, (len(position), 1))
    for _, centre in CYLINDERS:
        x = position[:, 0] - centre[0] - DISTURBANCE_CENTRE
        y = position[:, 1] - centre[1]
        support = np.maximum(1.0 - (x**2 + y**2) / DISTURBANCE_RADIUS**2, 0.0)
        swirl = 8.0 * DISTURBANCE_AMPLITUDE / DISTURBANCE_RADIUS**2 * support**3
        velocity[:, 0] -= swirl * y
        velocity[:, 1] += swirl * x
    return velocity


def build_case(name, end_time=END_TIME) -> fvm.FVMCase:
    """Return the FVM case of the tandem cylinders, written under ``solution/<name>``."""
    kinematic_viscosity = FREESTREAM_VELOCITY[0] * DIAMETER / REYNOLDS_NUMBER  # m^2/s
    half_span = 0.5 * SPAN
    margin = FINE_BOX_MARGIN * DIAMETER
    downstream_x = CYLINDERS[-1][1][0]
    fine_box = (-margin, downstream_x + margin + 2.0 * DIAMETER, -margin, margin)

    # Output times are whole steps, written with at most 3 decimals
    def on_steps(interval):
        return round(max(1, round(interval / TIME_STEP)) * TIME_STEP, 3)

    # Requested sizes sit between octree levels, which then select 1 and 2 CELL_SIZE
    request = 4.0 * CELL_SIZE / 3.0
    # The meshed slab holds the whole cylinder STL (|z| <= 0.6 m), as the case 01 reference
    slab = 16 * request

    surfaces = tuple(
        msh.STLSurface(
            CYLINDER_STL
            if centre == (0.0, 0.0)
            else write_cylinder_stl(CASE_DIR / f"assets/cylinder_{name}.stl", centre),
            patch=f"cylinder_{name}",
        )
        for name, centre in CYLINDERS
    )
    patches = msh.BoxPatches(
        xmin="inlet", xmax="outlet", ymin="ymin", ymax="ymax", zmin="zmin", zmax="zmax"
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=msh.CartesianMesher(
            domain=msh.BoxDomain(bounds=(*DOMAIN, -slab, slab), patches=patches),
            surfaces=surfaces,
            max_cell_size=BACKGROUND * CELL_SIZE,
            refinements=(
                msh.BoxRefinement("coupledDomains", (*fine_box, -slab, slab), request),
                msh.BoxRefinement("wake", (*WAKE, -slab, slab), 2 * request),
            ),
            patch_refinements=tuple(
                msh.PatchRefinement(surface.patch, request) for surface in surfaces
            ),
            surface_may_cross_domain_boundary=True,
        ),
        domain=msh.BoxDomain(bounds=(*DOMAIN, -half_span, half_span), patches=patches),
        levels=(-half_span, half_span),
    )

    profiles = fvm.EveryTime(on_steps(PROFILE_INTERVAL))
    samplers = (
        # Pressure and viscous forces and moments on each cylinder, per unit span
        *(
            fvm.ForceSampler(
                patch_names=[f"cylinder_{name}"],
                reference_velocity=FREESTREAM_VELOCITY[0],
                reference_area=DIAMETER * SPAN,
                reference_length=DIAMETER,
                moment_centre=(*centre, 0.0),
                file_name=f"forces_{name}",
                schedule=fvm.EveryTime(on_steps(FORCE_INTERVAL)),
            )
            for name, centre in CYLINDERS
        ),
        # Transverse wake profiles 2 and 4 D behind each cylinder
        *(
            fvm.LineSampler(
                (centre[0] + distance, centre[1] - 2.0, 0.0),
                (centre[0] + distance, centre[1] + 2.0, 0.0),
                n_points=41,
                n_neighbours=12,
                reconstruction="affine",
                file_name=f"{name}_transverse_x{distance:g}D",
                schedule=profiles,
            )
            for name, centre in CYLINDERS
            for distance in (2.0, 4.0)
        ),
    )
    numerics = fvm.Numerics(
        time_step_size=TIME_STEP,
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
            *(fvm.BoundaryConfig.wall(surface.patch) for surface in surfaces),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default="fixed", help="case name: solution/<name>, samples/<name>"
    )
    parser.add_argument("--end-time", type=float, default=END_TIME, help="end time [s]")
    arguments = parser.parse_args()

    case = build_case(arguments.name, arguments.end_time)
    with fvm.FVMSolver(case) as solver:
        cell_centres = solver.geo_data["cell_centre"][: solver.mesh_data["n_cells"]]
        solver.set_initial_velocity(initial_velocity(cell_centres))
        solver.run(start_from="latest")
