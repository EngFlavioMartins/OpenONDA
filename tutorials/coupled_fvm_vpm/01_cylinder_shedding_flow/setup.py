#!/usr/bin/env python3
"""Coupled FVM–VPM flow past a circular cylinder at Re = 150.

The flow is planar. The FVM resolves the no-slip wall and the near wake with
one periodic cell across the span. A single layer of three-dimensional vortex
particles carries the outer wake between free-slip planes one particle spacing
apart; with all its images in those planes it induces the planar flow. The span
follows the particle spacing, so the FVM cell and the particle layer share it.
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

# Startup vortex: breaks wake symmetry and starts shedding
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, streamfunction peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = 0.65  # m, downstream of the cylinder axis

# FVM mesh and near wake
CELL_SIZE = 0.04  # m, also the baseline particle spacing
BOX_MARGIN = 2.4  # D upstream and sideways; 2 D farther downstream
FVM_TIME_STEP = 0.008  # s

# VPM: particles in the outer wake
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
MAX_STRENGTH_RATIO = 1.8  # transfer strength-correction limit
END_TIME = 100.0  # s

# Output
FORCE_INTERVAL = 0.04  # s
PROFILE_INTERVAL = 0.2  # s, transverse wake profiles at x = 2 and 4 m
FIELD_INTERVAL = 0.2  # s, FVM and particle fields
FIELD_START = 40.0  # s, developed shedding
BACKUP_INTERVAL = 1.0  # s
TRANSFER_DIAGNOSTIC_INTERVAL = 2.4  # s


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
    box_margin=BOX_MARGIN,
    fvm_substeps=FVM_SUBSTEPS,
    transfer_inset=TRANSFER_INSET,
    end_time=END_TIME,
    fvm_time_step=FVM_TIME_STEP,
    particle_spacing=CELL_SIZE,
    coupling_iterations=COUPLING_ITERATIONS,
    coupling_tolerance=COUPLING_TOLERANCE,
    reynolds_number=REYNOLDS_NUMBER,
    cell_size=CELL_SIZE,
    box_downstream=BOX_MARGIN + 2.0,
    interface_cell_size=CELL_SIZE,
    max_strength_ratio=MAX_STRENGTH_RATIO,
    blend_start=BLEND_START,
    blend_end=BLEND_END,
):
    """Return the FVM case, the particle case and the coupling settings."""
    kinematic_viscosity = FREESTREAM_VELOCITY[0] * DIAMETER / reynolds_number  # m^2/s

    # One FVM cell and one particle layer across the span make the flow planar.
    span = particle_spacing
    half_span = 0.5 * span

    # FVM domain, transfer region and coupling interval.
    margin = box_margin * DIAMETER
    downstream = box_downstream * DIAMETER
    fvm_domain_bounds = (-margin, downstream, -margin, margin)
    inset = transfer_inset * interface_cell_size
    transfer_region = (-margin + inset, downstream - inset, -margin + inset, margin - inset)
    coupling_time_step_size = fvm_substeps * fvm_time_step

    # Output times coincide with accepted coupling states.
    def on_coupling_times(interval):
        return max(1, round(interval / coupling_time_step_size)) * coupling_time_step_size

    # Choose enough diffusion stages for the particle spacing and time step.
    fourier_number = kinematic_viscosity * coupling_time_step_size / particle_spacing**2
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
        domain=msh.BoxDomain(
            bounds=(*fvm_domain_bounds, -16 * cell_size, 16 * cell_size), patches=patches
        ),
        surfaces=(msh.STLSurface(CASE_DIR / "assets/cylinder_long.stl", patch="cylinder"),),
        max_cell_size=cell_size,
        cell_size_anchor=cell_size,
        patch_refinements=(msh.PatchRefinement("cylinder", cell_size),),
        surface_may_cross_domain_boundary=True,
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=slab,
        domain=msh.BoxDomain(
            bounds=(*slab.domain.bounds[:4], -half_span, half_span), patches=patches
        ),
        levels=(-half_span, half_span),
    )

    # Inner flow: drag and lift on the cylinder, a near-wake profile and volume fields
    samplers = (
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=FREESTREAM_VELOCITY[0],
            reference_area=DIAMETER * span,
            reference_length=DIAMETER,
            schedule=fvm.EveryTime(on_coupling_times(FORCE_INTERVAL)),
        ),
        fvm.LineSampler(
            (2.0, -1.5, 0.0),
            (2.0, 1.5, 0.0),
            n_points=31,
            n_neighbours=12,
            reconstruction="affine",
            file_name="fvm_transverse_x2",
            schedule=fvm.EveryTime(on_coupling_times(PROFILE_INTERVAL)),
        ),
    )
    flow = fvm.FVMCase(
        name="coupled_cylinder_flow",
        directory=CASE_DIR,
        numerics=fvm.Numerics(
            time_step_size=fvm_time_step,
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
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
        run=fvm.RunPlan(end_time=on_coupling_times(end_time)),
        output=fvm.Output(
            schedule=fvm.EveryTime(
                on_coupling_times(FIELD_INTERVAL), start_time=on_coupling_times(FIELD_START)
            ),
            compression="lz4",
            ghost_layers=0,
        ),
        samplers=fvm.Samplers(samplers),
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
            fvm.BoundaryConfig.wall("cylinder"),
        ),
    )

    # Outer wake: one particle layer between slip planes, with transverse wake profiles
    particles = vpm.VPMCase(
        name="coupled_cylinder_flow",
        directory=CASE_DIR,
        numerics=vpm.Numerics(
            time_step_size=coupling_time_step_size,
            freestream_velocity=FREESTREAM_VELOCITY,
            integrator=vpm.RK2(),
            # Full 3D induction of the particle layer and all its images in the span planes
            induction=vpm.SlipSlabInduction(vpm.FMMInduction(), z_min=-half_span, z_max=half_span),
            # Laminar grid-based diffusion on the particle lattice
            viscous=vpm.ViscousConfig.gbd(
                particle_spacing=particle_spacing,
                padding=max(5.0, diffusion_stages + 1.0),
                kinematic_viscosity=kinematic_viscosity,
                threshold_mode="absolute",
                threshold=VORTICITY_FLOOR * particle_spacing**3,
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
        run=vpm.RunPlan(end_time=on_coupling_times(end_time)),
        output=vpm.Output(
            schedule=vpm.EveryTime(
                on_coupling_times(FIELD_INTERVAL), start_time=on_coupling_times(FIELD_START)
            )
        ),
        samplers=vpm.Samplers(
            tuple(
                vpm.LineSampler(
                    (x, -2.0, 0.0),
                    (x, 2.0, 0.0),
                    spacing=0.1,
                    include_derivatives=False,
                    file_name=f"vpm_transverse_x{x:g}",
                    schedule=vpm.EveryTime(on_coupling_times(PROFILE_INTERVAL)),
                )
                for x in (2.0, 4.0)
            )
        ),
        logging=vpm.LoggingConfig(console=False),
    )

    # Coupling: transfer region, coupling iterations and coupled backups with particle fields
    coupling = coupler.CouplerConfig(
        freestream_velocity=FREESTREAM_VELOCITY,
        transfer_region_bounds=(*transfer_region, -half_span, half_span),
        blend_start_depth=blend_start * interface_cell_size,
        blend_end_depth=blend_end * interface_cell_size,
        max_coupling_iterations=coupling_iterations,
        normal_velocity_tolerance=coupling_tolerance,
        normal_gradient_tolerance=coupling_tolerance,
        max_strength_ratio=max_strength_ratio,
        transfer_diagnostic_interval_steps=round(
            TRANSFER_DIAGNOSTIC_INTERVAL / coupling_time_step_size
        ),
        backup=coupler.Backup(schedule=coupler.EveryTime(on_coupling_times(BACKUP_INTERVAL))),
    )
    return flow, particles, coupling


if __name__ == "__main__":
    # Coupling parameters of the sensitivity study; the defaults are the baseline case
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--box-margin",
        type=float,
        default=BOX_MARGIN,
        help="FVM domain margin upstream and sideways [D]",
    )
    parser.add_argument(
        "--box-downstream",
        type=float,
        help="downstream outer FVM boundary from the cylinder axis [D]; the margin + 2 by default",
    )
    parser.add_argument(
        "--fvm-substeps", type=int, default=FVM_SUBSTEPS, help="FVM steps per coupling interval"
    )
    parser.add_argument(
        "--transfer-inset", type=float, default=TRANSFER_INSET, help="transfer-region inset [cells]"
    )
    parser.add_argument("--end-time", type=float, default=END_TIME, help="end time [s]")
    parser.add_argument(
        "--cell-size", type=float, default=CELL_SIZE, help="FVM cell size near the wall [m]"
    )
    parser.add_argument(
        "--interface-cell-size",
        type=float,
        help="cell size counting the transfer inset and the blending depths [m]; the cell size by"
        " default, a fixed value keeps the interface in place while the cells are refined",
    )
    parser.add_argument(
        "--fvm-time-step",
        type=float,
        help="FVM time step [s]; scales with the cell size by default",
    )
    parser.add_argument(
        "--particle-spacing",
        type=float,
        help="particle spacing and span [m]; the cell size by default",
    )
    parser.add_argument(
        "--coupling-iterations",
        type=int,
        default=COUPLING_ITERATIONS,
        help="most coupling iterations (FVM solves) per coupling interval",
    )
    parser.add_argument(
        "--coupling-tolerance",
        type=float,
        default=COUPLING_TOLERANCE,
        help="boundary-data tolerance [m/s, 1/s]; 1e-30 forces every iteration",
    )
    parser.add_argument(
        "--reynolds-number", type=float, default=REYNOLDS_NUMBER, help="Reynolds number U D / nu"
    )
    parser.add_argument(
        "--max-strength-ratio",
        type=float,
        default=MAX_STRENGTH_RATIO,
        help="strength-ratio limit kappa_max of the transfer correction",
    )
    parser.add_argument(
        "--blend-start",
        type=float,
        default=BLEND_START,
        help="zero-weight layer depth delta_0 below the transfer surface [interface cells]",
    )
    parser.add_argument(
        "--blend-end",
        type=float,
        default=BLEND_END,
        help="depth delta_1 at which the blending weight reaches one [interface cells]",
    )
    arguments = parser.parse_args()

    # Refining the cells refines the steps at constant Courant number, and the particles with them
    refinement = arguments.cell_size / CELL_SIZE
    fvm_time_step = arguments.fvm_time_step or refinement * FVM_TIME_STEP
    particle_spacing = arguments.particle_spacing or arguments.cell_size
    box_downstream = arguments.box_downstream or arguments.box_margin + 2.0
    interface_cell_size = arguments.interface_cell_size or arguments.cell_size

    flow, particles, coupling = build_case(
        arguments.box_margin,
        arguments.fvm_substeps,
        arguments.transfer_inset,
        arguments.end_time,
        fvm_time_step,
        particle_spacing,
        arguments.coupling_iterations,
        arguments.coupling_tolerance,
        arguments.reynolds_number,
        arguments.cell_size,
        box_downstream,
        interface_cell_size,
        arguments.max_strength_ratio,
        arguments.blend_start,
        arguments.blend_end,
    )
    with coupler.CoupledSolver(flow, particles, coupling) as solver:
        # The startup vortex behind the cylinder; a restart replaces it with the backup
        fvm_solver = solver.fvm_solver
        cell_centres = fvm_solver.geo_data["cell_centre"][: fvm_solver.mesh_data["n_cells"]]
        fvm_solver.set_initial_velocity(initial_velocity(cell_centres))
        solver.run(start_from="latest")
