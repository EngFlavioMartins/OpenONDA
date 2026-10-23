#!/usr/bin/env python3
"""Coupled LES FVM–VPM flow past a finite NACA 4412 wing at 10 degrees and Re = 1000.

The FVM resolves the no-slip wing, its tips and its near wake on a body-fitted
Cartesian mesh; vortex particles carry the trailing vortices and the outer
wake. Both solvers use the same Smagorinsky closure. The freestream direction
sets the incidence, with the chord along x.
"""

import math
from pathlib import Path

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm

from assets.naca_surface import write_wing_stl

CASE_DIR = Path(__file__).resolve().parent
CASE_NAME = "coupled_naca4412_flow"

# Wing: NACA 4412 section (4% camber at 40% chord, 12% thickness), chord centred on the origin
NACA_CODE = "4412"
CHORD = 1.0  # m
SPAN = 5.0  # m, flat end caps at z = +-2.5 m
N_SECTION_POINTS = 161  # points on each surface, leading to trailing edge
WING_STL = CASE_DIR / "assets/naca4412_wing.stl"

# Flow
ANGLE_OF_ATTACK = 10.0  # degrees
FREESTREAM_SPEED = 1.0  # m/s
REYNOLDS_NUMBER = 1000.0
DENSITY = 1.0  # kg/m^3
KINEMATIC_VISCOSITY = FREESTREAM_SPEED * CHORD / REYNOLDS_NUMBER  # m^2/s
ALPHA = math.radians(ANGLE_OF_ATTACK)
FREESTREAM_VELOCITY = (FREESTREAM_SPEED * math.cos(ALPHA), FREESTREAM_SPEED * math.sin(ALPHA), 0.0)

# FVM: uniform body-fitted cells, also the particle spacing
CELL_SIZE = 0.04  # m, 25 cells per chord
# Box in wake scales (cylinder error study, docs/reports/2026-10-23-coupling-error-sources.md):
# the core with full FVM weight starts the transfer inset plus the blend width (14.75 cells,
# 0.59 m) inside the box faces and keeps at least five shear-layer thicknesses (about 0.1 c at
# Re = 1000) from the wing below, from the rising wake above, and from the tips
FVM_BOX = (-1.6, 2.72, -1.15, 1.45, -3.6, 3.6)  # m
FVM_TIME_STEP = 0.008  # s, Courant number 0.2
FVM_CORES = 4

# VPM: particles in the outer wake, which rises along the 10-degree freestream
VPM_DOMAIN = (-3.2, 24.0, -2.4, 6.4, -4.8, 4.8)  # m
OUTFLOW_FRINGE = 12.0  # m, the trailing vortices fade before the downstream face
VORTICITY_FLOOR = 0.003  # 1/s, weaker particles are pruned after diffusion
CORE_RADIUS_RATIO = 1.05
PARTICLE_LIMIT = 2_000_000

# Coupling: FVM vorticity replaces the particles inside the transfer region
# Coupling: the coupler's recommended interface (transfer inset 8.75, blend 6 and particle-only
# band 2 particle spacings; sweeps until the boundary trace changes by less than 1e-5)
FVM_SUBSTEPS = 5  # FVM steps per exchange: particle step 0.04 s, Lagrangian step number about
# 0.75 for a particle-field strain near the cube's 18 1/s; fewer exchanges, smaller error

# Time: the starting vortex is 12 chords downstream before the statistics begin
STATISTICS_START = 12.0  # s
END_TIME = 24.0  # s

# Observations
FORCE_INTERVAL = 0.04  # s
PROFILE_INTERVAL = 0.2  # s, wake profiles over the whole run
SLICE_INTERVAL = 0.2  # s, planar slices over the statistics window
FIELD_INTERVAL = 2.0  # s, FVM fields and particles over the statistics window
BACKUP_INTERVAL = 1.0  # s, coupled restart states
RENEWAL_CHECK_INTERVAL = 2.4  # s, costly checks of how well particles represent FVM vorticity
WAKE_STATIONS = (2.0, 5.0)  # m, inside the pure-FVM core and in the particle wake
PROFILE_SPACING = 2.0 * CELL_SIZE  # m

# Geometry derived from the inputs above
EXCHANGE_TIME_STEP = FVM_SUBSTEPS * FVM_TIME_STEP
TRAILING_EDGE = 0.5 * CHORD  # m
VPM_SLICE_Z0 = (-1.6, 8.0, -1.6, 3.2)  # m, x and y bounds of the mid-span plane
VPM_SLICE_CROSS = (-1.6, 3.2, -4.0, 4.0)  # m, y and z bounds of the cross-flow planes


def on_exchanges(interval):
    """Observations fall on exchanges, where both solvers hold the same time."""
    return max(1, round(interval / EXCHANGE_TIME_STEP)) * EXCHANGE_TIME_STEP


def wake_height(x):
    """Height of the freestream line through the trailing edge, where the spanwise probes run."""
    return (x - TRAILING_EDGE) * math.tan(ALPHA)


# The analytical wing surface the mesher fits
write_wing_stl(WING_STL, NACA_CODE, CHORD, SPAN, N_SECTION_POINTS, n_span_segments=1)

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
    surfaces=(msh.STLSurface(WING_STL, patch="wing"),),
    max_cell_size=CELL_SIZE,
    cell_size_anchor=CELL_SIZE,
)

profiles = fvm.EveryTime(on_exchanges(PROFILE_INTERVAL))
slices = fvm.EveryTime(on_exchanges(SLICE_INTERVAL), start_time=STATISTICS_START)
fields = fvm.EveryTime(on_exchanges(FIELD_INTERVAL), start_time=STATISTICS_START)

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
        turbulence=fvm.TurbulenceConfig.smagorinsky(),
        execution=fvm.ComputeConfig(operator_backend="numba"),
    ),
    initial_conditions=fvm.InitialFields(velocity=FREESTREAM_VELOCITY),
    run=fvm.RunPlan(end_time=END_TIME),
    output=fvm.Output(schedule=fields, compression="lz4", ghost_layers=0),
    samplers=fvm.Samplers(
        (
            # Body-axis force coefficients on the planform area c b; the moment about the quarter chord
            fvm.ForceSampler(
                patch_names=["wing"],
                reference_velocity=FREESTREAM_SPEED,
                reference_area=CHORD * SPAN,
                reference_length=CHORD,
                moment_centre=(-0.25 * CHORD, 0.0, 0.0),
                schedule=fvm.EveryTime(on_exchanges(FORCE_INTERVAL)),
            ),
            # Wake profile across the mid-span wake and along the span through both tip vortices
            fvm.LineSampler(
                (WAKE_STATIONS[0], FVM_BOX[2], 0.0),
                (WAKE_STATIONS[0], FVM_BOX[3], 0.0),
                spacing=PROFILE_SPACING,
                n_neighbours=12,
                reconstruction="affine",
                file_name=f"fvm_transverse_x{WAKE_STATIONS[0]:g}",
                schedule=profiles,
            ),
            fvm.LineSampler(
                (WAKE_STATIONS[0], wake_height(WAKE_STATIONS[0]), FVM_BOX[4]),
                (WAKE_STATIONS[0], wake_height(WAKE_STATIONS[0]), FVM_BOX[5]),
                spacing=PROFILE_SPACING,
                n_neighbours=12,
                reconstruction="affine",
                file_name=f"fvm_spanwise_x{WAKE_STATIONS[0]:g}",
                schedule=profiles,
            ),
            # Mid-span plane and cross-flow plane, over the FVM box
            fvm.SurfaceSampler(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                FVM_BOX[:4],
                PROFILE_SPACING,
                n_neighbours=12,
                reconstruction="affine",
                file_name="fvm_slice_z0",
                schedule=slices,
                initial=False,
            ),
            fvm.SurfaceSampler(
                (WAKE_STATIONS[0], 0.0, 0.0),
                (1.0, 0.0, 0.0),
                FVM_BOX[2:],
                PROFILE_SPACING,
                n_neighbours=12,
                reconstruction="affine",
                file_name=f"fvm_slice_x{WAKE_STATIONS[0]:g}",
                schedule=slices,
                initial=False,
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
        fvm.BoundaryConfig.wall("wing"),
    ),
    cores=FVM_CORES,
)

particle_profiles = vpm.EveryTime(on_exchanges(PROFILE_INTERVAL))
particle_slices = vpm.EveryTime(on_exchanges(SLICE_INTERVAL), start_time=STATISTICS_START)

VPM_CASE = vpm.VPMCase(
    name=CASE_NAME,
    directory=CASE_DIR,
    numerics=vpm.Numerics(
        time_step_size=EXCHANGE_TIME_STEP,
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
        turbulence=vpm.TurbulenceConfig.smagorinsky(),
        stabilization=vpm.StabilizationConfig.bounded_domain(
            VPM_DOMAIN,
            outflow_fringe_depth=(0.0, OUTFLOW_FRINGE, 0.0, 0.0, 0.0, 0.0),
        ),
        domain_bounds=VPM_DOMAIN,
        max_n_particles=PARTICLE_LIMIT,
    ),
    run=vpm.RunPlan(end_time=END_TIME),
    output=vpm.Output(
        schedule=vpm.EveryTime(on_exchanges(FIELD_INTERVAL), start_time=STATISTICS_START)
    ),
    samplers=vpm.Samplers(
        (
            *(
                vpm.LineSampler(
                    (x, VPM_SLICE_CROSS[0], 0.0),
                    (x, VPM_SLICE_CROSS[1], 0.0),
                    spacing=PROFILE_SPACING,
                    include_derivatives=False,
                    file_name=f"vpm_transverse_x{x:g}",
                    schedule=particle_profiles,
                )
                for x in WAKE_STATIONS
            ),
            *(
                vpm.LineSampler(
                    (x, wake_height(x), VPM_SLICE_CROSS[2]),
                    (x, wake_height(x), VPM_SLICE_CROSS[3]),
                    spacing=PROFILE_SPACING,
                    include_derivatives=False,
                    file_name=f"vpm_spanwise_x{x:g}",
                    schedule=particle_profiles,
                )
                for x in WAKE_STATIONS
            ),
            vpm.SurfaceSampler(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                VPM_SLICE_Z0,
                PROFILE_SPACING,
                include_derivatives=False,
                file_name="vpm_slice_z0",
                schedule=particle_slices,
                initial=False,
            ),
            *(
                vpm.SurfaceSampler(
                    (x, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    VPM_SLICE_CROSS,
                    PROFILE_SPACING,
                    include_derivatives=False,
                    file_name=f"vpm_slice_x{x:g}",
                    schedule=particle_slices,
                    initial=False,
                )
                for x in WAKE_STATIONS
            ),
        )
    ),
    logging=vpm.LoggingConfig(console=False),
)

COUPLER_CONFIG = coupling.CouplerConfig(
    freestream_velocity=FREESTREAM_VELOCITY,
    transfer_diagnostic_interval_steps=round(RENEWAL_CHECK_INTERVAL / EXCHANGE_TIME_STEP),
    backup=coupling.Backup(schedule=coupling.EveryTime(on_exchanges(BACKUP_INTERVAL))),
)


if __name__ == "__main__":
    with coupling.CoupledSolver(FVM_CASE, VPM_CASE, COUPLER_CONFIG) as solver:
        solver.run(start_from="latest")
