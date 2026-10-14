#!/usr/bin/env python3
"""Coupled LES FVM–VPM flow past a cube at Re = 1000.

The FVM resolves the no-slip cube and its near wake on a body-fitted Cartesian
mesh; vortex particles carry the outer wake. Both solvers use the same
equilibrium Smagorinsky closure. A short transverse startup breaks the
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

# Startup: a transverse freestream that decays smoothly breaks the wake symmetry
STARTUP_VELOCITY = (1.0, 0.1, 0.05)  # m/s
TAPER_START = 1.0  # s, the transverse components then decay smoothly ...
TAPER_END = 2.0  # s, ... to the freestream

# FVM: uniform body-fitted cells, also the particle spacing
CELL_SIZE = 0.045  # m
BOX_MARGIN = 1.8  # m upstream and sideways of the cube centre: keeps the near field inside
BOX_DOWNSTREAM = 4.3  # m: the pure-FVM core extends past the vortex-formation region
FVM_TIME_STEP = 0.01  # s
FVM_CORES = 4

# VPM: particles in the outer wake
VPM_DOMAIN = (-4.5, 24.0, -3.0, 3.0, -3.0, 3.0)  # m
OUTFLOW_FRINGE = 12.0  # m, shed vortices fade before the downstream face
VORTICITY_FLOOR = 0.003  # 1/s, weaker particles are pruned after diffusion
CORE_RADIUS_RATIO = 1.05
PARTICLE_LIMIT = 2_000_000

# Coupling: FVM vorticity replaces the particles inside the transfer region
TRANSFER_INSET = 8.75  # cells from the FVM box faces in to the transfer-region faces
BLEND_WIDTH = 6  # cells over which the FVM weight rises from zero to one
VPM_ONLY_WIDTH = 2  # cells inside the transfer faces that only the particles own
FVM_SUBSTEPS = 5  # FVM steps per exchange; one particle step per exchange
INTERFACE_ITERATIONS = 6  # FVM solves per exchange until the boundary trace converges
INTERFACE_TOLERANCE = 1.0e-5  # m/s for normal velocity, 1/s for its tangential gradient
END_TIME = 100.0  # s

# Observations
FORCE_INTERVAL = 0.05  # s
PROFILE_INTERVAL = 0.25  # s, centreline and off-axis lines
SLICE_INTERVAL = 1.0  # s, mid-plane slices
VOLUME_INTERVAL = 2.0  # s, FVM fields
BACKUP_INTERVAL = 1.0  # s, coupled restart states
RENEWAL_CHECK_INTERVAL = 2.5  # s, costly checks of how well particles represent FVM vorticity
OFFAXIS_Y = 0.75 * CUBE_SIDE  # m
PROFILE_SPACING = 2.0 * CELL_SIZE  # m

# Geometry derived from the inputs above
FVM_BOX = (-BOX_MARGIN, BOX_DOWNSTREAM, -BOX_MARGIN, BOX_MARGIN, -BOX_MARGIN, BOX_MARGIN)
INSET = TRANSFER_INSET * CELL_SIZE
TRANSFER_REGION = tuple(bound + INSET * side for bound, side in zip(FVM_BOX, (1, -1) * 3))
CUBE_BOUNDS = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
EXCHANGE_TIME_STEP = FVM_SUBSTEPS * FVM_TIME_STEP


def on_exchanges(interval):
    """Observations fall on exchanges, where both solvers hold the same time."""
    return max(1, round(interval / EXCHANGE_TIME_STEP)) * EXCHANGE_TIME_STEP


FVM_MESH = msh.CartesianMesher(
    domain=msh.BoxDomain(
        bounds=FVM_BOX,
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
    initial_conditions=fvm.InitialFields(velocity=STARTUP_VELOCITY),
    run=fvm.RunPlan(end_time=END_TIME),
    output=fvm.Output(
        schedule=fvm.EveryTime(on_exchanges(VOLUME_INTERVAL)),
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
                schedule=fvm.EveryTime(on_exchanges(FORCE_INTERVAL)),
            ),
            *(
                fvm.LineSampler(
                    (FVM_BOX[0], y, 0.0),
                    (FVM_BOX[1], y, 0.0),
                    spacing=PROFILE_SPACING,
                    file_name=f"fvm_{name}",
                    schedule=fvm.EveryTime(on_exchanges(PROFILE_INTERVAL)),
                )
                for name, y in (("centreline", 0.0), ("offaxis_y075", OFFAXIS_Y))
            ),
            fvm.SurfaceSampler(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                FVM_BOX[:4],
                PROFILE_SPACING,
                body_bounds=CUBE_BOUNDS,
                file_name="fvm_slice_z0",
                schedule=fvm.EveryTime(on_exchanges(SLICE_INTERVAL)),
            ),
        )
    ),
    logging=fvm.LoggingConfig(console=False),
    mesh=FVM_MESH,
    boundaries=(
        # Particles supply the normal velocity and the normal derivative of tangential velocity
        fvm.BoundaryConfig(
            name="numericalBoundary",
            velocity_value=STARTUP_VELOCITY,
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
        time_step_size=EXCHANGE_TIME_STEP,
        freestream_velocity=STARTUP_VELOCITY,
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
                    schedule=vpm.EveryTime(on_exchanges(PROFILE_INTERVAL)),
                )
                for name, y in (("centreline", 0.0), ("offaxis_y075", OFFAXIS_Y))
            ),
            vpm.SurfaceSampler(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                FVM_BOX[:4],
                PROFILE_SPACING,
                include_derivatives=False,
                file_name="vpm_slice_z0",
                schedule=vpm.EveryTime(on_exchanges(SLICE_INTERVAL)),
            ),
            vpm.SurfaceSampler(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 12.0, -2.0, 2.0),
                PROFILE_SPACING,
                include_derivatives=False,
                file_name="vpm_wake_slice_z0",
                schedule=vpm.EveryTime(on_exchanges(SLICE_INTERVAL)),
            ),
        )
    ),
    logging=vpm.LoggingConfig(console=False),
)

COUPLER_CONFIG = coupling.CouplerConfig(
    freestream_velocity=STARTUP_VELOCITY,
    freestream=coupling.VelocityRamp(
        initial=STARTUP_VELOCITY,
        final=FREESTREAM_VELOCITY,
        start_time=TAPER_START,
        end_time=TAPER_END,
    ),
    transfer_region_bounds=TRANSFER_REGION,
    eta_blend_width=BLEND_WIDTH * CELL_SIZE,
    vpm_only_width=VPM_ONLY_WIDTH * CELL_SIZE,
    interface_iterations=INTERFACE_ITERATIONS,
    interface_normal_tolerance=INTERFACE_TOLERANCE,
    interface_gradient_tolerance=INTERFACE_TOLERANCE,
    transfer_diagnostic_interval_steps=round(RENEWAL_CHECK_INTERVAL / EXCHANGE_TIME_STEP),
    backup=coupling.Backup(schedule=coupling.EveryTime(on_exchanges(BACKUP_INTERVAL))),
)


if __name__ == "__main__":
    with coupling.CoupledSolver(FVM_CASE, VPM_CASE, COUPLER_CONFIG) as solver:
        solver.run(start_from="latest")
