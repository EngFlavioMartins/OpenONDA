#!/usr/bin/env python3
"""Coupled LES FVM–VPM flow past a cube at Re = 1000.

The FVM resolves the no-slip cube and its near wake on a body-fitted Cartesian
mesh; vortex particles carry the outer wake. Both solvers use the same
equilibrium Smagorinsky closure. A vortex behind the cube breaks the
symmetry of the impulsively started flow so that the wake sheds.
"""

from pathlib import Path

import numpy as np

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm

CASE_DIR = Path(__file__).resolve().parent
CASE_NAME = "coupled_cube_flow"

# Flow
CUBE_SIDE = 1.0  # m
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)  # m/s
REYNOLDS_NUMBER = 1000.0
DENSITY = 1.0  # kg/m^3
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY[0] * CUBE_SIDE / REYNOLDS_NUMBER  # m^2/s

# Startup vortex: breaks both wake symmetries.
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, vector-potential peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = (1.0, 0.0, 0.0)  # m, half a side behind the rear face
DISTURBANCE_AXIS = (0.0, -1.0, 2.0)

# FVM mesh and particle spacing
CELL_SIZE = 0.045  # m
BOX_MARGIN = 1.8  # m, clearance around the cube and separated shear layers
BOX_DOWNSTREAM = 5.05  # m: transfer surface at 1.5 formation lengths (wake reach R = 0.5)
FVM_TIME_STEP = 0.01  # s, Courant number 0.22
FVM_CORES = 8

# VPM: particles in the outer wake
VPM_DOMAIN = (-4.5, 24.0, -3.0, 3.0, -3.0, 3.0)  # m
OUTFLOW_FRINGE = 12.0  # m, smooth vorticity removal near the downstream boundary
VORTICITY_FLOOR = 0.003  # 1/s, remove weak diffused particles
CORE_RADIUS_RATIO = 1.05
PARTICLE_LIMIT = 2_000_000

# FVM–VPM transfer and run duration
FVM_SUBSTEPS = 1
END_TIME = 60.0  # s

# Output
FORCE_INTERVAL = 0.04  # s
PROFILE_INTERVAL = 0.24  # s, centreline and off-axis lines
SLICE_INTERVAL = 1.0  # s, mid-plane slices, at the times of the reference slices
SLICE_WINDOW = (-1.8, 12.0, -2.0, 2.0)  # m, x and y bounds of the particle slice
VOLUME_INTERVAL = 2.0  # s, FVM fields
BACKUP_INTERVAL = 1.0  # s, coupled restart states
TRANSFER_DIAGNOSTIC_INTERVAL = 2.4  # s
OFFAXIS_Y = 0.75 * CUBE_SIDE  # m
PROFILE_SPACING = 2.0 * CELL_SIZE  # m

# Geometry derived from the inputs above
FVM_DOMAIN_BOUNDS = (-BOX_MARGIN, BOX_DOWNSTREAM, -BOX_MARGIN, BOX_MARGIN, -BOX_MARGIN, BOX_MARGIN)
CUBE_BOUNDS = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
COUPLING_TIME_STEP = FVM_SUBSTEPS * FVM_TIME_STEP


def initial_velocity(position):
    """Freestream plus the curl of A (1 - r^2/R^2)^4 along the disturbance axis."""
    offset = position - np.asarray(DISTURBANCE_CENTRE)
    support = np.maximum(1.0 - np.sum(offset**2, axis=1) / DISTURBANCE_RADIUS**2, 0.0)
    swirl = 8.0 * DISTURBANCE_AMPLITUDE / DISTURBANCE_RADIUS**2 * support**3
    axis = np.asarray(DISTURBANCE_AXIS) / np.linalg.norm(DISTURBANCE_AXIS)
    return np.asarray(FREESTREAM_VELOCITY) + swirl[:, None] * np.cross(axis, offset)


FVM_MESH = msh.CartesianMesher(
    domain=msh.BoxDomain(
        bounds=FVM_DOMAIN_BOUNDS,
        patches=msh.BoxPatches(
            xmin="numericalBoundary",
            xmax="numericalBoundary",
            ymin="numericalBoundary",
            ymax="numericalBoundary",
            zmin="numericalBoundary",
            zmax="numericalBoundary",
        ),
    ),
    surfaces=(msh.STLSurface(CASE_DIR / "assets/cube.stl", patch="cube"),),
    max_cell_size=CELL_SIZE,
    cell_size_anchor=CELL_SIZE,
)

FVM_CASE = fvm.FVMCase(
    name=CASE_NAME,
    directory=CASE_DIR,
    numerics=fvm.Numerics(
        time_step_size=FVM_TIME_STEP,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="linear_upwind",
            gradient_scheme="gauss",
            time_scheme="backward",
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg",
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
        ),
        pimple=fvm.PimpleConfig(
            n_outer_correctors=2,
            n_nonorthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        turbulence=fvm.TurbulenceConfig.equilibrium_smagorinsky(),
        execution=fvm.ComputeConfig(operator_backend="numba"),
    ),
    initial_conditions=fvm.InitialFields(velocity=FREESTREAM_VELOCITY),
    run=fvm.RunPlan(end_time=END_TIME),
    output=fvm.Output(
        schedule=fvm.EveryTime(VOLUME_INTERVAL),
        compression="lz4",
        ghost_layers=0,
    ),
    samplers=fvm.Samplers(
        (
            # Force and moment coefficients on the frontal area
            fvm.ForceSampler(
                patch_names=["cube"],
                reference_velocity=FREESTREAM_VELOCITY[0],
                reference_area=CUBE_SIDE**2,
                reference_length=CUBE_SIDE,
                moment_centre=[0.0, 0.0, 0.0],
                schedule=fvm.EveryTime(FORCE_INTERVAL),
            ),
            *(
                fvm.LineSampler(
                    (FVM_DOMAIN_BOUNDS[0], y, 0.0),
                    (FVM_DOMAIN_BOUNDS[1], y, 0.0),
                    spacing=PROFILE_SPACING,
                    file_name=f"fvm_{name}",
                    schedule=fvm.EveryTime(PROFILE_INTERVAL),
                )
                for name, y in (("centreline", 0.0), ("offaxis_y075", OFFAXIS_Y))
            ),
            fvm.SurfaceSampler(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                FVM_DOMAIN_BOUNDS[:4],
                PROFILE_SPACING,
                body_bounds=CUBE_BOUNDS,
                file_name="fvm_slice_z0",
                schedule=fvm.EveryTime(SLICE_INTERVAL),
            ),
        )
    ),
    logging=fvm.LoggingConfig(console=False),
    mesh=FVM_MESH,
    boundaries=(
        # Particles supply the normal velocity and the normal derivative of tangential velocity
        fvm.BoundaryConfig(
            name="numericalBoundary",
            velocity_value=FREESTREAM_VELOCITY,
            pressure_type="fixed_flux_pressure",
        ),
        fvm.BoundaryConfig.wall("cube"),
    ),
    cores=FVM_CORES,
)

VPM_CASE = vpm.VPMCase(
    name=CASE_NAME,
    directory=CASE_DIR,
    numerics=vpm.Numerics(
        time_step_size=COUPLING_TIME_STEP,
        freestream_velocity=FREESTREAM_VELOCITY,
        integrator=vpm.RK2(),
        induction=vpm.FMMInduction(),
        viscous=vpm.ViscousConfig.gbd(
            particle_spacing=CELL_SIZE,
            padding=5.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            threshold_mode="absolute",
            threshold=VORTICITY_FLOOR * CELL_SIZE**3,
            core_radius_ratio=CORE_RADIUS_RATIO,
        ),
        turbulence=vpm.TurbulenceConfig.equilibrium_smagorinsky(),
        stabilization=vpm.StabilizationConfig.bounded_domain(
            VPM_DOMAIN,
            outflow_fringe_depth=(0.0, OUTFLOW_FRINGE, 0.0, 0.0, 0.0, 0.0),
        ),
        domain_bounds=VPM_DOMAIN,
        max_n_particles=PARTICLE_LIMIT,
    ),
    run=vpm.RunPlan(end_time=END_TIME),
    samplers=vpm.Samplers(
        (
            *(
                vpm.LineSampler(
                    (VPM_DOMAIN[0], y, 0.0),
                    (VPM_DOMAIN[1], y, 0.0),
                    spacing=PROFILE_SPACING,
                    include_derivatives=False,
                    file_name=f"vpm_{name}",
                    schedule=vpm.EveryTime(PROFILE_INTERVAL),
                )
                for name, y in (("centreline", 0.0), ("offaxis_y075", OFFAXIS_Y))
            ),
            # The particle flow over the whole compared wake, on the reference slice lattice
            vpm.SurfaceSampler(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                SLICE_WINDOW,
                PROFILE_SPACING,
                include_derivatives=False,
                file_name="vpm_slice_z0",
                schedule=vpm.EveryTime(SLICE_INTERVAL),
            ),
        )
    ),
    logging=vpm.LoggingConfig(console=False),
)

COUPLER_CONFIG = coupling.CouplerConfig(
    freestream_velocity=FREESTREAM_VELOCITY,
    transfer_diagnostic_interval_steps=round(TRANSFER_DIAGNOSTIC_INTERVAL / COUPLING_TIME_STEP),
    backup=coupling.Backup(schedule=coupling.EveryTime(BACKUP_INTERVAL)),
)


if __name__ == "__main__":
    with coupling.CoupledSolver(FVM_CASE, VPM_CASE, COUPLER_CONFIG) as solver:
        # The startup vortex behind the cube; a restart replaces it with the backup
        fvm_solver = solver.fvm_solver
        cell_centres = fvm_solver.geo_data["cell_centre"][: fvm_solver.mesh_data["n_cells"]]
        fvm_solver.set_initial_velocity(initial_velocity(cell_centres))
        solver.run(start_from="latest")
