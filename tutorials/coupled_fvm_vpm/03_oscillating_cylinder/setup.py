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

# Prescribed transverse motion near the natural shedding frequency
AMPLITUDE = 0.2 * DIAMETER  # m
FREQUENCY = 0.18  # Hz

# FVM: uniform body-fitted cells near the wall
CELL_SIZE = 0.04  # m, also the particle spacing
BOX_MARGIN = 2.4  # D upstream and sideways, 2 D more downstream: holds the vortex-formation region
FVM_TIME_STEP = 0.008  # s

# VPM: particles in the outer wake, positions relative to the moving box
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
END_TIME = 50.0  # s

# Observations
FORCE_INTERVAL = 0.04  # s
VOLUME_INTERVAL = 5.0  # s, FVM fields
BACKUP_INTERVAL = 1.0  # s, coupled restart states and matching particle fields
RENEWAL_CHECK_INTERVAL = 2.4  # s, costly checks of how well particles represent FVM vorticity


def build_case():
    """Return the FVM case, the particle case and the coupling settings."""
    # One FVM cell and one particle layer across the span make the flow planar
    span = CELL_SIZE
    half_span = 0.5 * span

    # FVM box, transfer region inset from its faces, and the exchange step
    margin = BOX_MARGIN * DIAMETER
    fvm_box = (-margin, margin + 2.0 * DIAMETER, -margin, margin)
    inner = margin - TRANSFER_INSET * CELL_SIZE
    transfer_region = (-inner, inner + 2.0 * DIAMETER, -inner, inner)
    exchange_time_step = FVM_SUBSTEPS * FVM_TIME_STEP

    # Observations fall on exchanges, where both solvers hold the same time
    def on_exchanges(interval):
        return max(1, round(interval / exchange_time_step)) * exchange_time_step

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
            schedule=fvm.EveryTime(on_exchanges(VOLUME_INTERVAL)),
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
                    schedule=fvm.EveryTime(on_exchanges(FORCE_INTERVAL)),
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
            time_step_size=exchange_time_step,
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
        logging=vpm.LoggingConfig(console=False),
    )

    coupling = coupler.CouplerConfig(
        freestream_velocity=FREESTREAM_VELOCITY,
        transfer_region_bounds=(*transfer_region, -half_span, half_span),
        eta_blend_width=BLEND_WIDTH * CELL_SIZE,
        vpm_only_width=VPM_ONLY_WIDTH * CELL_SIZE,
        interface_iterations=INTERFACE_ITERATIONS,
        interface_normal_tolerance=INTERFACE_TOLERANCE,
        interface_gradient_tolerance=INTERFACE_TOLERANCE,
        transfer_diagnostic_interval_steps=round(RENEWAL_CHECK_INTERVAL / exchange_time_step),
        backup=coupler.Backup(schedule=coupler.EveryTime(on_exchanges(BACKUP_INTERVAL))),
    )
    return flow, particles, coupling


if __name__ == "__main__":
    flow, particles, coupling = build_case()
    with coupler.CoupledSolver(flow, particles, coupling) as solver:
        solver.run(start_from="latest")
