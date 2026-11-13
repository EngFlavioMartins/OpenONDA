#!/usr/bin/env python3
"""Coupled LES FVM–VPM flow past a finite NACA 4412 wing at 10 degrees and Re = 1000.

The FVM resolves the no-slip wing, its tips and its near wake on a body-fitted
Cartesian mesh; vortex particles carry the trailing vortices and the outer
wake. Both solvers use the same Smagorinsky closure. The freestream runs along
x and the wing is pitched nose-up about its mid-chord, so the stream crosses the
outer FVM faces normally or not at all.
"""

import math
from pathlib import Path

import openonda.coupler as coupler
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm

from assets.naca_surface import write_wing_stl

CASE_DIR = Path(__file__).resolve().parent
CASE_NAME = "coupled_naca4412_flow"

# Wing geometry
NACA_CODE = "4412"
CHORD = 1.0  # m
SPAN = 5.0  # m
N_SECTION_POINTS = 161  # per surface
WING_STL = CASE_DIR / "assets/naca4412_wing.stl"

# Flow
ANGLE_OF_ATTACK = 10.0  # degrees
FREESTREAM_SPEED = 1.0  # m/s
REYNOLDS_NUMBER = 1000.0
DENSITY = 1.0  # kg/m^3
KINEMATIC_VISCOSITY = FREESTREAM_SPEED * CHORD / REYNOLDS_NUMBER  # m^2/s
ALPHA = math.radians(ANGLE_OF_ATTACK)
FREESTREAM_VELOCITY = (FREESTREAM_SPEED, 0.0, 0.0)

# FVM mesh and particle spacing
CELL_SIZE = 0.04  # m, 25 cells per chord
# Wing, tips and near wake; the box is centred on a plane of the 0.04 m lattice, so its cells
# lie on those of the reference mesh
FVM_DOMAIN_BOUNDS = (-1.6, 2.56, -1.32, 1.24, -3.6, 3.6)  # m
# The round trailing edge in cells of half the wing cell size, so its 20 mm base spans a cell
TRAILING_EDGE_BOX = (0.32, 0.52, -0.16, 0.04, -0.5 * SPAN - CELL_SIZE, 0.5 * SPAN + CELL_SIZE)  # m
# Its corners with the flat tip caps in cells of a quarter of the wing cell size: in half-size
# cells one wall face there spans the trailing edge and leaves its cell centre inside the wing
TIP_CORNER = (0.38, 0.47, -0.11, -0.02)  # m, x and y bounds around the round trailing edge
TIP_CORNER_SPAN = (0.5 * SPAN - 2 * CELL_SIZE, 0.5 * SPAN + CELL_SIZE)  # m, |z| bounds
FVM_INITIAL_TIME_STEP = 0.0035  # s, just below the CFL step of the 0.01 m trailing-edge cells
FVM_MAX_TIME_STEP = 0.04  # s, the FVM step adapts to a Courant number of 0.9
VPM_MAX_COUPLING_INTERVAL = 0.08  # s; at most 23 FVM substeps per particle step (0.08/0.0035),
# so the output times or the particle strain (CFL 0.9) set the interval, not the FVM CFL step
FVM_CORES = 8

# VPM: particles in the outer wake
VPM_DOMAIN = (-3.2, 24.0, -4.0, 3.2, -4.8, 4.8)  # m
OUTFLOW_FRINGE = 12.0  # m, smooth vorticity removal near the downstream boundary
VORTICITY_FLOOR = 0.01  # 1/s, remove weak diffused particles (cylinder study: 1e-3 moved the forces, 1e-2 did not)
CORE_RADIUS_RATIO = 1.05
WALL_FALLBACK_LIMIT = 0.01  # largest strength fraction per step deposited without first moments
PARTICLE_LIMIT = 2_000_000

# Time window
STATISTICS_START = 12.0  # s
END_TIME = 24.0  # s

# Output
FORCE_INTERVAL = 0.08  # s, the interval ceiling; with the 0.2 s profiles and slices, kept on the
# sample times of the reference flow, the intervals are 0.08 and 0.04 s, 0.067 s on average
PROFILE_INTERVAL = 0.2  # s, wake profiles over the whole run
SLICE_INTERVAL = 0.2  # s, planar slices over the statistics window
FIELD_INTERVAL = 1.0  # s, FVM fields and particles over the statistics window
BACKUP_INTERVAL = 0.48  # s, coupled restart states
WAKE_STATIONS = (2.0, 5.0)  # m, inside the replacement region and in the particle wake
PROFILE_SPACING = 2.0 * CELL_SIZE  # m
TRANSFER_DIAGNOSTIC_INTERVAL_STEPS = 36  # accepted shared steps, 2.4 s at 0.067 s

# Trailing-edge height and the force moment centre
WAKE_HEIGHT = -0.5 * CHORD * math.sin(ALPHA)  # m
QUARTER_CHORD = (-0.25 * CHORD * math.cos(ALPHA), 0.25 * CHORD * math.sin(ALPHA), 0.0)  # m
VPM_SLICE_Z0 = (-1.6, 8.0, -2.4, 2.4)  # m, x and y bounds of the mid-span plane
VPM_SLICE_CROSS = (-2.4, 2.4, -4.0, 4.0)  # m, y and z bounds of the cross-flow planes
STARTING_VORTEX_PLANES = (4.0, 8.0)  # m, cross-flow planes of the starting-vortex passage
STARTING_VORTEX_SPACING = 0.04  # m


# The wing surface for the mesher
write_wing_stl(WING_STL, NACA_CODE, CHORD, SPAN, N_SECTION_POINTS, 1, ANGLE_OF_ATTACK)

FVM_MESH = msh.CartesianMesher(
    domain=msh.BoxDomain(
        bounds=FVM_DOMAIN_BOUNDS,
        # The whole box is the coupling interface: exact Cartesian faces, edges and corners.
        patches=msh.BoxPatches.coupled("numericalBoundary"),
    ),
    surfaces=(msh.STLSurface(WING_STL, patch="wing"),),
    max_cell_size=CELL_SIZE,
    cell_size_anchor=0.25 * CELL_SIZE,
    refinements=(
        msh.BoxRefinement("trailingEdge", TRAILING_EDGE_BOX, 0.5 * CELL_SIZE),
        *(
            msh.BoxRefinement(f"tipCorner{side}", (*TIP_CORNER, *bounds), 0.25 * CELL_SIZE)
            for side, bounds in (
                ("Min", (-TIP_CORNER_SPAN[1], -TIP_CORNER_SPAN[0])),
                ("Max", TIP_CORNER_SPAN),
            )
        ),
    ),
)

profiles = fvm.EveryTime(PROFILE_INTERVAL)
slices = fvm.EveryTime(SLICE_INTERVAL, start_time=STATISTICS_START)
fields = fvm.EveryTime(FIELD_INTERVAL, start_time=STATISTICS_START)

FVM_CASE = fvm.FVMCase(
    name=CASE_NAME,
    directory=CASE_DIR,
    numerics=fvm.Numerics(
        time_step_size=FVM_INITIAL_TIME_STEP,
        time_step_control=fvm.MaximumCourantTimeStep(
            maximum=0.9,
            maximum_time_step_size=FVM_MAX_TIME_STEP,
        ),
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
            # Drag and lift coefficients on the planform area c b; the moment about the quarter chord
            fvm.ForceSampler(
                patch_names=["wing"],
                reference_velocity=FREESTREAM_SPEED,
                reference_area=CHORD * SPAN,
                reference_length=CHORD,
                moment_centre=QUARTER_CHORD,
                schedule=fvm.EveryTime(FORCE_INTERVAL),
            ),
            # Wake profile across the mid-span wake and along the span through both tip vortices
            fvm.LineSampler(
                (WAKE_STATIONS[0], FVM_DOMAIN_BOUNDS[2], 0.0),
                (WAKE_STATIONS[0], FVM_DOMAIN_BOUNDS[3], 0.0),
                spacing=PROFILE_SPACING,
                n_neighbours=12,
                reconstruction="affine",
                file_name=f"fvm_transverse_x{WAKE_STATIONS[0]:g}",
                schedule=profiles,
            ),
            fvm.LineSampler(
                (WAKE_STATIONS[0], WAKE_HEIGHT, FVM_DOMAIN_BOUNDS[4]),
                (WAKE_STATIONS[0], WAKE_HEIGHT, FVM_DOMAIN_BOUNDS[5]),
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
                FVM_DOMAIN_BOUNDS[:4],
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
                FVM_DOMAIN_BOUNDS[2:],
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

particle_profiles = vpm.EveryTime(PROFILE_INTERVAL)
particle_slices = vpm.EveryTime(SLICE_INTERVAL, start_time=STATISTICS_START)

VPM_CASE = vpm.VPMCase(
    name=CASE_NAME,
    directory=CASE_DIR,
    numerics=vpm.Numerics(
        time_step_size=VPM_MAX_COUPLING_INTERVAL,
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
            # The 20 mm trailing edge is thinner than the 0.04 m lattice: particles at it can
            # have no bounded moment-conserving wall stencil. They keep their strength; the run
            # stops if they carry more than 1 % of the total in one step (first run: t = 0.16 s).
            wall_fallback_limit=WALL_FALLBACK_LIMIT,
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
    output=vpm.Output(schedule=vpm.EveryTime(FIELD_INTERVAL, start_time=STATISTICS_START)),
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
                    (x, WAKE_HEIGHT, VPM_SLICE_CROSS[2]),
                    (x, WAKE_HEIGHT, VPM_SLICE_CROSS[3]),
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
            # The planes the starting vortex crosses, from t = 0
            *(
                vpm.SurfaceSampler(
                    (x, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    VPM_SLICE_CROSS,
                    STARTING_VORTEX_SPACING,
                    include_derivatives=False,
                    file_name=f"vpm_slice_x{x:g}",
                    schedule=vpm.EveryTime(SLICE_INTERVAL),
                )
                for x in STARTING_VORTEX_PLANES
            ),
        )
    ),
    logging=vpm.LoggingConfig(console=False),
)

COUPLER_CONFIG = coupler.CouplerConfig(
    freestream_velocity=FREESTREAM_VELOCITY,
    transfer_diagnostic_interval_steps=TRANSFER_DIAGNOSTIC_INTERVAL_STEPS,
    backup=coupler.Backup(schedule=coupler.EveryTime(BACKUP_INTERVAL)),
)

if __name__ == "__main__":
    with coupler.CoupledSolver(FVM_CASE, VPM_CASE, COUPLER_CONFIG) as solver:
        solver.run(start_from="latest")
