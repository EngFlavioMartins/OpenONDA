#!/usr/bin/env python3
"""Coupled FVM–VPM flow past a heaving circular cylinder at Re = 150.

The cylinder moves across the stream, y(t) = A sin(2 pi f t). The FVM box and
its body-fitted mesh move with the cylinder; the vortex particles of the outer
wake are solved in the frame translating with the box, where the stream is
U - V(t).
"""

import math
from pathlib import Path

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

# Prescribed transverse motion
AMPLITUDE = 0.2 * DIAMETER  # m
FREQUENCY = 0.18  # Hz

# FVM mesh and near wake
CELL_SIZE = 0.04  # m, also the particle spacing
BOX_MARGIN = 2.4  # D upstream and sideways; 2 D farther downstream
FVM_TIME_STEP = 0.008  # s

# VPM outer wake, relative to the moving box
VPM_DOMAIN = (-5.0, 30.0, -5.0, 5.0)  # m, x and y bounds
OUTFLOW_FRINGE = 15.0  # m, smooth vorticity removal near the downstream boundary
VORTICITY_FLOOR = 0.01  # 1/s, remove weak diffused particles
PARTICLE_LIMIT = 1_000_000

# FVM–VPM transfer
TRANSFER_INSET = 8.75  # cells between the outer FVM boundary and the transfer surface
BLEND_START = 2  # cells below the transfer surface
BLEND_END = 6  # cells below the transfer surface
FVM_SUBSTEPS = 5  # per particle step
COUPLING_ITERATIONS = 6  # maximum FVM solves per particle step
COUPLING_TOLERANCE = 1.0e-5  # boundary velocity and tangential-gradient tolerance
END_TIME = 50.0  # s

# Output
FORCE_INTERVAL = 0.04  # s
PROFILE_INTERVAL = 0.2  # s, transverse wake profiles at x = 2 and 4 m
FIELD_INTERVAL = 0.2  # s, FVM and particle fields
FIELD_START = 25.0  # s, locked-in wake
BACKUP_INTERVAL = 1.0  # s
TRANSFER_DIAGNOSTIC_INTERVAL = 2.4  # s


def build_case():
    """Return the FVM case, the particle case and the coupling settings."""
    # One FVM cell and one particle layer across the span make the flow planar.
    span = CELL_SIZE
    half_span = 0.5 * span

    # FVM domain, transfer region and coupling interval.
    margin = BOX_MARGIN * DIAMETER
    fvm_domain_bounds = (-margin, margin + 2.0 * DIAMETER, -margin, margin)
    inner = margin - TRANSFER_INSET * CELL_SIZE
    transfer_region = (-inner, inner + 2.0 * DIAMETER, -inner, inner)
    coupling_time_step_size = FVM_SUBSTEPS * FVM_TIME_STEP
    # Choose enough diffusion stages for the particle spacing and time step.
    fourier_number = KINEMATIC_VISCOSITY * coupling_time_step_size / CELL_SIZE**2
    diffusion_stages = math.floor(12.0 * fourier_number) + 1

    # Mesh a slab around the cylinder and retain one periodic spanwise cell.
    patches = msh.BoxPatches(
        xmin="numericalBoundary",
        xmax="numericalBoundary",
        ymin="numericalBoundary",
        ymax="numericalBoundary",
        zmin="zmin",
        zmax="zmax",
    )
    slab = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(*fvm_domain_bounds, -16 * CELL_SIZE, 16 * CELL_SIZE), patches=patches
        ),
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

    flow = fvm.FVMCase(
        name="coupled_oscillating_cylinder",
        directory=CASE_DIR,
        numerics=fvm.Numerics(
            time_step_size=FVM_TIME_STEP,
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
            linear=fvm.LinearSolverConfig(
                pressure_tolerance=1.0e-7, pressure_relative_tolerance=0.005
            ),
            pimple=fvm.PimpleConfig(
                n_outer_correctors=2, velocity_relaxation=0.7, pressure_relaxation=0.3
            ),
            execution=fvm.ComputeConfig(operator_backend="numba"),
        ),
        initial_conditions=fvm.InitialFields(velocity=FREESTREAM_VELOCITY),
        run=fvm.RunPlan(end_time=END_TIME),
        output=fvm.Output(
            schedule=fvm.EveryTime(FIELD_INTERVAL, start_time=FIELD_START),
            compression="lz4",
            ghost_layers=0,
        ),
        samplers=fvm.Samplers(
            (
                # Drag and lift coefficients per unit span
                fvm.ForceSampler(
                    patch_names=["cylinder"],
                    reference_velocity=FREESTREAM_VELOCITY[0],
                    reference_area=DIAMETER * span,
                    reference_length=DIAMETER,
                    schedule=fvm.EveryTime(FORCE_INTERVAL),
                ),
                # Near-wake profile inside the box, at a fixed inertial position
                fvm.LineSampler(
                    (2.0, -1.5, 0.0),
                    (2.0, 1.5, 0.0),
                    n_points=31,
                    n_neighbours=12,
                    reconstruction="affine",
                    file_name="fvm_transverse_x2",
                    schedule=fvm.EveryTime(PROFILE_INTERVAL),
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
            fvm.BoundaryConfig.wall("cylinder"),
        ),
        # The whole FVM box heaves with the cylinder
        mesh_motion=fvm.MeshMotion(
            fvm.HeavingMotion(AMPLITUDE, FREQUENCY, direction=(0.0, 1.0, 0.0))
        ),
    )

    particles = vpm.VPMCase(
        name="coupled_oscillating_cylinder",
        directory=CASE_DIR,
        numerics=vpm.Numerics(
            time_step_size=coupling_time_step_size,
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
        run=vpm.RunPlan(end_time=END_TIME),
        output=vpm.Output(
            schedule=vpm.EveryTime(FIELD_INTERVAL, start_time=FIELD_START)
        ),
        # Transverse wake profiles in the frame of the box; 0.4 m longer on each side,
        # so they cover y = -2 to 2 m about the mean cylinder position at any heave
        samplers=vpm.Samplers(
            tuple(
                vpm.LineSampler(
                    (x, -2.4, 0.0),
                    (x, 2.4, 0.0),
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
        blend_start_depth=BLEND_START * CELL_SIZE,
        blend_end_depth=BLEND_END * CELL_SIZE,
        max_coupling_iterations=COUPLING_ITERATIONS,
        normal_velocity_tolerance=COUPLING_TOLERANCE,
        normal_gradient_tolerance=COUPLING_TOLERANCE,
        transfer_diagnostic_interval_steps=round(
            TRANSFER_DIAGNOSTIC_INTERVAL / coupling_time_step_size
        ),
        backup=coupler.Backup(schedule=coupler.EveryTime(BACKUP_INTERVAL)),
    )
    return flow, particles, coupling


if __name__ == "__main__":
    flow, particles, coupling = build_case()
    with coupler.CoupledSolver(flow, particles, coupling) as solver:
        solver.run(start_from="latest")
