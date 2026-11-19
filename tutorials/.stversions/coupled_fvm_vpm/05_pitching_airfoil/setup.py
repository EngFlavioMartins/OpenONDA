#!/usr/bin/env python3
"""Coupled FVM–VPM flow past a NACA 0012 airfoil pitching about its quarter chord at Re = 1000.

The angle of attack follows alpha(t) = alpha0 sin(2 pi f t). The FVM box and
its body-fitted mesh turn with the airfoil about the quarter chord; the vortex
particles of the outer wake stay in the inertial frame, and the coupling
geometry is kept in the coordinates of the turning box.
"""

import argparse
import math
from pathlib import Path

import openonda.coupler as coupler
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm

from assets.naca_surface import write_wing_stl

CASE_DIR = Path(__file__).resolve().parent

# Airfoil, centred on the origin, the leading edge at x = -c/2
CHORD = 1.0  # m
NACA_CODE = "0012"
N_SECTION_POINTS = 161  # per surface
AIRFOIL_STL = CASE_DIR / "assets/naca0012_long.stl"

# Flow
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)  # m/s
REYNOLDS_NUMBER = 1000.0
DENSITY = 1.0  # kg/m^3
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY[0] * CHORD / REYNOLDS_NUMBER  # m^2/s

# Prescribed pitch, nose up for positive angles
PITCH_AMPLITUDE = 10.0  # degrees
FREQUENCY = 0.2  # Hz, reduced frequency pi f c / U = 0.63
PITCH_AXIS = (-0.25 * CHORD, 0.0, 0.0)  # m, the quarter chord

# FVM mesh and near wake
CELL_SIZE = 0.025  # m, also the particle spacing
FVM_DOMAIN = (-1.6, 2.6, -1.2, 1.2)  # m, x and y bounds in the frame of the airfoil
FVM_TIME_STEP = 0.005  # s

# VPM outer wake
VPM_DOMAIN = (-5.0, 25.0, -5.0, 5.0)  # m, x and y bounds
OUTFLOW_FRINGE = 12.0  # m, smooth vorticity removal near the downstream boundary
VORTICITY_FLOOR = 0.01  # 1/s, remove weak diffused particles
PARTICLE_LIMIT = 2_000_000

# FVM–VPM transfer
TRANSFER_INSET = 8.75  # cells between the outer FVM boundary and the transfer surface
FVM_SUBSTEPS = 5  # per particle step
END_TIME = 30.0  # s, six pitch periods

# Output
FORCE_INTERVAL = 0.025  # s
PROFILE_INTERVAL = 0.2  # s, transverse wake profiles at x = 2 and 4 m
FIELD_INTERVAL = 0.2  # s, FVM and particle fields
FIELD_START = 15.0  # s, periodic wake
BACKUP_INTERVAL = 1.0  # s
TRANSFER_DIAGNOSTIC_INTERVAL = 2.5  # s


def build_case(end_time=END_TIME, compute_device="auto"):
    """Return the FVM case, the particle case and the coupling settings."""
    # One FVM cell and one particle layer across the span make the flow planar.
    span = CELL_SIZE
    half_span = 0.5 * span

    # Transfer region and coupling interval.
    inset = TRANSFER_INSET * CELL_SIZE
    transfer_region = (
        FVM_DOMAIN[0] + inset,
        FVM_DOMAIN[1] - inset,
        FVM_DOMAIN[2] + inset,
        FVM_DOMAIN[3] - inset,
    )
    coupling_time_step_size = FVM_SUBSTEPS * FVM_TIME_STEP
    # Choose enough diffusion stages for the particle spacing and time step.
    fourier_number = KINEMATIC_VISCOSITY * coupling_time_step_size / CELL_SIZE**2
    diffusion_stages = math.floor(12.0 * fourier_number) + 1

    # Mesh a slab that the airfoil crosses, then retain one periodic spanwise cell.
    slab = 14 * CELL_SIZE
    write_wing_stl(AIRFOIL_STL, NACA_CODE, CHORD, 4.0 * slab, N_SECTION_POINTS, 1, 0.0)
    patches = msh.BoxPatches(
        xmin="numericalBoundary",
        xmax="numericalBoundary",
        ymin="numericalBoundary",
        ymax="numericalBoundary",
        zmin="zmin",
        zmax="zmax",
        # The x and y faces form the coupling interface, meshed as exact Cartesian planes.
        coupling=("numericalBoundary",),
    )
    source = msh.CartesianMesher(
        domain=msh.BoxDomain(bounds=(*FVM_DOMAIN, -slab, slab), patches=patches),
        surfaces=(msh.STLSurface(AIRFOIL_STL, patch="airfoil"),),
        max_cell_size=CELL_SIZE,
        cell_size_anchor=CELL_SIZE,
        patch_refinements=(msh.PatchRefinement("airfoil", CELL_SIZE),),
        surface_may_cross_domain_boundary=True,
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=source,
        domain=msh.BoxDomain(
            bounds=(*source.domain.bounds[:4], -half_span, half_span), patches=patches
        ),
        levels=(-half_span, half_span),
    )

    flow = fvm.FVMCase(
        name="coupled_pitching_airfoil",
        directory=CASE_DIR,
        numerics=fvm.Numerics(
            time_step_size=FVM_TIME_STEP,
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
            linear=fvm.LinearSolverConfig(
                pressure_tolerance=1.0e-7, pressure_relative_tolerance=0.005
            ),
            # Three outer passes converge the steps of the accelerating mesh
            pimple=fvm.PimpleConfig(
                n_outer_correctors=3, velocity_relaxation=0.7, pressure_relaxation=0.3
            ),
            execution=fvm.ComputeConfig(operator_backend="numba"),
        ),
        initial_conditions=fvm.InitialFields(velocity=FREESTREAM_VELOCITY),
        run=fvm.RunPlan(end_time=end_time),
        output=fvm.Output(
            schedule=fvm.EveryTime(FIELD_INTERVAL, start_time=FIELD_START),
            compression="lz4",
            ghost_layers=0,
        ),
        samplers=fvm.Samplers(
            (
                # Drag, lift and quarter-chord moment coefficients per unit span
                fvm.ForceSampler(
                    patch_names=["airfoil"],
                    reference_velocity=FREESTREAM_VELOCITY[0],
                    reference_area=CHORD * span,
                    reference_length=CHORD,
                    moment_centre=PITCH_AXIS,
                    schedule=fvm.EveryTime(FORCE_INTERVAL),
                ),
            )
        ),
        logging=fvm.LoggingConfig(console=False),
        mesh=mesh,
        boundaries=(
            # Particles supply the normal velocity and the normal derivative of tangential velocity
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_value=FREESTREAM_VELOCITY,
                pressure_type="fixed_flux_pressure",
            ),
            fvm.BoundaryConfig.cyclic("zmin", "zmax"),
            fvm.BoundaryConfig.cyclic("zmax", "zmin"),
            # A wall on the moving mesh moves with it
            fvm.BoundaryConfig.wall("airfoil"),
        ),
        # The whole FVM box turns with the airfoil; a positive angle about -z lifts the nose
        mesh_motion=fvm.MeshMotion(
            fvm.PitchingMotion(
                PITCH_AMPLITUDE,
                FREQUENCY,
                rotation_axis=(0.0, 0.0, -1.0),
                rotation_centre=PITCH_AXIS,
            )
        ),
    )

    particles = vpm.VPMCase(
        name="coupled_pitching_airfoil",
        directory=CASE_DIR,
        numerics=vpm.Numerics(
            time_step_size=coupling_time_step_size,
            compute_device=compute_device,
            freestream_velocity=FREESTREAM_VELOCITY,
            integrator=vpm.RK2(),
            # Full 3D induction of the particle layer and all its images in the span planes
            induction=vpm.SlipSlabInduction(vpm.FMMInduction(), z_min=-half_span, z_max=half_span),
            # Laminar grid-based diffusion on the particle lattice
            viscous=vpm.ViscousConfig.gbd(
                particle_spacing=CELL_SIZE,
                padding=max(5.0, diffusion_stages + 1.0),
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                threshold_mode="absolute",
                threshold=VORTICITY_FLOOR * CELL_SIZE**3,
                core_radius_ratio=1.0,
            ),
            turbulence=vpm.TurbulenceConfig.inviscid(),
            stabilization=vpm.StabilizationConfig.bounded_domain(
                (*VPM_DOMAIN, -half_span, half_span),
                outflow_fringe_depth=(0.0, OUTFLOW_FRINGE, 0.0, 0.0, 0.0, 0.0),
            ),
            domain_bounds=(*VPM_DOMAIN, -half_span, half_span),
            max_n_particles=PARTICLE_LIMIT,
        ),
        run=vpm.RunPlan(end_time=end_time),
        output=vpm.Output(schedule=vpm.EveryTime(FIELD_INTERVAL, start_time=FIELD_START)),
        # Transverse wake profiles at fixed inertial positions
        samplers=vpm.Samplers(
            tuple(
                vpm.LineSampler(
                    (x, -2.0, 0.0),
                    (x, 2.0, 0.0),
                    spacing=0.1,
                    include_derivatives=False,
                    file_name=f"vpm_transverse_x{x:g}",
                    schedule=vpm.EveryTime(PROFILE_INTERVAL),
                )
                for x in (2.0, 4.0)
            )
        ),
        logging=vpm.LoggingConfig(console=False),
    )

    coupling = coupler.CouplerConfig(
        freestream_velocity=FREESTREAM_VELOCITY,
        transfer_region_bounds=(*transfer_region, -half_span, half_span),
        transfer_diagnostic_interval_steps=round(
            TRANSFER_DIAGNOSTIC_INTERVAL / coupling_time_step_size
        ),
        backup=coupler.Backup(schedule=coupler.EveryTime(BACKUP_INTERVAL)),
    )
    return flow, particles, coupling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float, default=END_TIME, help="end time [s]")
    parser.add_argument(
        "--compute-device",
        default="auto",
        choices=("auto", "cpu", "cuda", "vulkan"),
        help="particle solver device; auto takes the GPU when there is one",
    )
    arguments = parser.parse_args()
    flow, particles, coupling = build_case(arguments.end_time, arguments.compute_device)
    with coupler.CoupledSolver(flow, particles, coupling) as solver:
        solver.run(start_from="latest")
