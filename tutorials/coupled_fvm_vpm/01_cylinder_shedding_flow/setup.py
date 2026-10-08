#!/usr/bin/env python3
"""Coupled FVM–VPM flow past a circular cylinder at Re = 150.

The flow is planar. The FVM resolves the no-slip wall and the near wake with
one periodic cell across the span. A single layer of three-dimensional vortex
particles carries the outer wake between free-slip planes one particle spacing
apart; with all its images in those planes it induces the planar flow.
"""

import argparse
import math
from pathlib import Path

import numpy as np

import openonda.coupler as coupler
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm

CASE_DIR = Path(__file__).resolve().parent

# Flow
DIAMETER = 1.0  # m
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)  # m/s
REYNOLDS_NUMBER = 150.0
DENSITY = 1.0  # kg/m^3
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY[0] * DIAMETER / REYNOLDS_NUMBER  # m^2/s

# Startup: a transverse freestream and a weak near-wake vortex trigger shedding
STARTUP_VELOCITY = (1.0, 0.1, 0.0)  # m/s
TAPER_START = 1.0  # s, the transverse component then decays smoothly ...
TAPER_END = 2.0  # s, ... to the freestream
DISTURBANCE_AMPLITUDE = 1.0e-3  # m^2/s, streamfunction peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = 0.65  # m, downstream of the cylinder axis

# FVM: uniform body-fitted cells near the wall
CELL_SIZE = 0.04  # m, also the particle spacing
BOX_MARGIN = 2.4  # D upstream and sideways, 2 D more downstream: holds the vortex-formation region
FVM_TIME_STEP = 0.008  # s

# VPM: particles in the outer wake
VPM_DOMAIN = (-5.0, 30.0, -5.0, 5.0)  # m, x and y bounds
OUTFLOW_FRINGE = 15.0  # m, shed vortices fade over about three wake wavelengths
VORTICITY_FLOOR = 0.01  # 1/s, weaker particles are pruned after diffusion
PARTICLE_LIMIT = 1_000_000

# Coupling: FVM vorticity replaces the particles inside the transfer region
TRANSFER_INSET = 8.75  # cells from the FVM box faces in to the transfer-region faces
BLEND_WIDTH = 6  # cells over which the FVM weight rises from zero to one
VPM_ONLY_WIDTH = 2  # cells inside the transfer faces that only the particles own
FVM_SUBSTEPS = 5  # FVM steps per exchange; one particle step per exchange
INTERFACE_ITERATIONS = 6  # FVM solves per exchange until the boundary trace converges
INTERFACE_TOLERANCE = 1.0e-5  # m/s for normal velocity, 1/s for its tangential gradient
END_TIME = 100.0  # s

# Observations
FORCE_INTERVAL = 0.04  # s
PROFILE_INTERVAL = 0.2  # s, transverse wake profiles at x = 2 and 4 m
VOLUME_INTERVAL = 4.0  # s, FVM fields
BACKUP_INTERVAL = 1.0  # s, coupled restart states and matching particle fields
RENEWAL_CHECK_INTERVAL = 2.4  # s, costly checks of how well particles represent FVM vorticity


def initial_velocity(position):
    """Startup freestream plus the curl of psi = A (1 - r^2/R^2)^4 behind the cylinder."""
    x = position[:, 0] - DISTURBANCE_CENTRE
    y = position[:, 1]
    support = np.maximum(1.0 - (x**2 + y**2) / DISTURBANCE_RADIUS**2, 0.0)
    swirl = 8.0 * DISTURBANCE_AMPLITUDE / DISTURBANCE_RADIUS**2 * support**3

    velocity = np.tile(STARTUP_VELOCITY, (len(position), 1))
    velocity[:, 0] -= swirl * y
    velocity[:, 1] += swirl * x
    return velocity


def build_case(
    box_margin=BOX_MARGIN,
    fvm_substeps=FVM_SUBSTEPS,
    transfer_inset=TRANSFER_INSET,
    end_time=END_TIME,
):
    """Return the FVM setup, particle case, coupling setup and mesh."""
    # One FVM cell and one particle layer across the span make the flow planar
    span = CELL_SIZE
    half_span = 0.5 * span

    # FVM box, transfer region inset from its faces, and the exchange step
    margin = box_margin * DIAMETER
    fvm_box = (-margin, margin + 2.0 * DIAMETER, -margin, margin)
    inner = margin - transfer_inset * CELL_SIZE
    transfer_region = (-inner, inner + 2.0 * DIAMETER, -inner, inner)
    exchange_time_step = fvm_substeps * FVM_TIME_STEP

    # Explicit GBD diffusion keeps 12 x Fourier number below one per stage; each
    # stage reaches one more grid cell beyond the span planes
    fourier_number = KINEMATIC_VISCOSITY * exchange_time_step / CELL_SIZE**2
    diffusion_stages = math.floor(12.0 * fourier_number) + 1

    # Mesh a slab around the long cylinder, then keep one periodic cell across the span
    patches = msh.BoxPatches(
        xmin="numericalBoundary",
        xmax="numericalBoundary",
        ymin="numericalBoundary",
        ymax="numericalBoundary",
        zmin="zmin",
        zmax="zmax",
    )
    slab = msh.CartesianMesher(
        domain=msh.BoxDomain(bounds=(*fvm_box, -16 * CELL_SIZE, 16 * CELL_SIZE), patches=patches),
        surfaces=(msh.STLSurface(CASE_DIR / "assets/cylinder_long.stl", patch="cylinder"),),
        max_cell_size=CELL_SIZE,
        cell_size_anchor=CELL_SIZE,
        patch_refinements=(msh.PatchRefinement("cylinder", CELL_SIZE),),
        surface_may_cross_domain_boundary=True,
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=slab,
        domain=msh.BoxDomain(
            bounds=(*slab.domain.bounds[:4], -half_span, half_span), patches=patches
        ),
        levels=(-half_span, half_span),
    )

    flow = fvm.FVMSetup(
        case_name="coupled_cylinder_flow",
        execution=fvm.ComputeConfig(operator_backend="numba"),
        output=fvm.OutputConfig(compression="lz4", ghost_layers=0),
        time=fvm.TimeConfig(
            time_step_size=FVM_TIME_STEP,
            end_time=end_time,
            output_schedule=fvm.RunSchedule(every_time=VOLUME_INTERVAL),
        ),
        schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        linear=fvm.LinearSolverConfig(
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
            momentum_tolerance=1.0e-6,
            momentum_relative_tolerance=0.05,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=2, velocity_relaxation=0.7, pressure_relaxation=0.3
        ),
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
        turbulence=fvm.TurbulenceConfig.none(),
        boundaries=[
            # Particles supply the normal velocity and the normal derivative of tangential velocity
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_value=list(STARTUP_VELOCITY),
                pressure_type="fixedFluxPressure",
            ),
            fvm.BoundaryConfig.cyclic("zmin", "zmax"),
            fvm.BoundaryConfig.cyclic("zmax", "zmin"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        initial_velocity=list(STARTUP_VELOCITY),
        samplers=(
            # Drag and lift coefficients per unit span
            fvm.ForceSampler(
                patch_names=["cylinder"],
                reference_velocity=FREESTREAM_VELOCITY[0],
                reference_area=DIAMETER * span,
                reference_length=DIAMETER,
                schedule=fvm.RunSchedule(every_time=FORCE_INTERVAL),
            ),
            # Near-wake profile inside the FVM box
            fvm.LineSampler(
                start=[2.0, -1.5, 0.0],
                end=[2.0, 1.5, 0.0],
                n_points=31,
                k=12,
                reconstruction="affine",
                file_name="fvm_transverse_x2",
                schedule=fvm.RunSchedule(every_time=PROFILE_INTERVAL),
            ),
        ),
    )

    particles = vpm.VPMCase(
        name="coupled_cylinder_flow",
        directory=CASE_DIR,
        numerics=vpm.Numerics(
            time_step_size=exchange_time_step,
            freestream_velocity=STARTUP_VELOCITY,
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
        samplers=vpm.Samplers(
            samples=tuple(
                vpm.LineSampler(
                    start=[x, -2.0, 0.0],
                    end=[x, 2.0, 0.0],
                    spacing=0.1,
                    include_derivatives=False,
                    file_name=f"vpm_transverse_x{x:g}",
                    schedule=vpm.EveryTime(PROFILE_INTERVAL),
                )
                for x in (2.0, 4.0)
            )
        ),
        run=vpm.RunPlan(steps=round(end_time / exchange_time_step)),
    )

    coupling = coupler.CouplerSetup(
        freestream_velocity=list(STARTUP_VELOCITY),
        freestream=coupler.VelocityRamp(
            initial=STARTUP_VELOCITY,
            final=FREESTREAM_VELOCITY,
            start_time=TAPER_START,
            end_time=TAPER_END,
        ),
        transfer_region_bounds=(*transfer_region, -half_span, half_span),
        eta_blend_width=BLEND_WIDTH * CELL_SIZE,
        vpm_only_width=VPM_ONLY_WIDTH * CELL_SIZE,
        interface_iterations=INTERFACE_ITERATIONS,
        interface_normal_tolerance=INTERFACE_TOLERANCE,
        interface_gradient_tolerance=INTERFACE_TOLERANCE,
        backup_interval_steps=round(BACKUP_INTERVAL / exchange_time_step),
        transfer_diagnostic_interval_steps=round(RENEWAL_CHECK_INTERVAL / exchange_time_step),
    )
    return flow, particles, coupling, mesh


if __name__ == "__main__":
    # Coupling parameters of the sensitivity study; the defaults are the baseline case
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-margin", type=float, default=BOX_MARGIN, help="FVM box margin [D]")
    parser.add_argument(
        "--fvm-substeps", type=int, default=FVM_SUBSTEPS, help="FVM steps per exchange"
    )
    parser.add_argument(
        "--transfer-inset", type=float, default=TRANSFER_INSET, help="transfer-region inset [cells]"
    )
    parser.add_argument("--end-time", type=float, default=END_TIME, help="end time [s]")
    arguments = parser.parse_args()

    flow, particles, coupling, mesh = build_case(
        arguments.box_margin, arguments.fvm_substeps, arguments.transfer_inset, arguments.end_time
    )
    with coupler.create_coupler(flow, particles, coupling, mesh=mesh) as solver:
        solver.run(start_from="latest", initial_velocity=initial_velocity)
