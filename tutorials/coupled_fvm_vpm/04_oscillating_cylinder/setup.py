#!/usr/bin/env python3
"""Coupled FVM–VPM flow past an elastically mounted circular cylinder with two degrees of freedom.

The cylinder of Martins and Avila (2019) rides on equal springs and dampers in
the streamwise and cross-flow directions, with mass ratio 2.4 and damping
ratio 0.5 %; the fluid force on its wall drives the motion. Each run takes one
reduced velocity V* = U/(f_n D) of the paper's low-Reynolds set, Re = 250 V*.
The FVM box and its body-fitted mesh translate with the cylinder; the vortex
particles of the outer wake are solved in the frame translating with the box,
where the stream is U - V(t).
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
DENSITY = 1.0  # kg/m^3
# Martins and Avila (2019), Int. J. Mech. Sci. 156, 235-249, low-Reynolds set (table 4):
# Re = 750-3300 over V* = 3-15 at a fixed natural frequency, and the upper-branch peak A1 at
# Re = 1500 is the V* = 6 peak of the text, so Re = 250 V*
REYNOLDS_PER_REDUCED_VELOCITY = 250.0
REDUCED_VELOCITIES = (4.0, 6.0, 8.0)  # the runs of allrun.sh

# Structure, as the paper: equal springs in x and y, natural frequency in water from the
# ideal added mass, damping ratio relative to the vacuum critical damping
MASS_RATIO = 2.4  # 4 m / (rho pi D^2 L)
ADDED_MASS_COEFFICIENT = 1.0  # C_A
DAMPING_RATIO = 0.005  # b / (2 sqrt(k m))

# Startup vortex: breaks the wake symmetry and starts shedding, as the fixed cylinder case
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, streamfunction peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = 0.65  # m, downstream of the cylinder axis

# FVM mesh and near wake
CELL_SIZE = 0.04  # m, at the coupling interface; also the particle spacing
WALL_CELL_SIZE = 0.02  # m, one octree level finer at the cylinder wall
BOX_MARGIN = 2.4  # D upstream and sideways; 2 D farther downstream
# Half depth of the meshing slab in cells: the long cylinder crosses its span faces, and a
# thin slab keeps the refined wall band small (the slab of 14 cells needed over 3 GB to mesh)
SLAB_HALF_CELLS = 6
FVM_TIME_STEP = 0.0025  # s

# VPM outer wake, relative to the moving box
VPM_DOMAIN = (-5.0, 30.0, -5.0, 5.0)  # m, x and y bounds
OUTFLOW_FRINGE = 15.0  # m, smooth vorticity removal near the downstream boundary
VORTICITY_FLOOR = 0.01  # 1/s, remove weak diffused particles
PARTICLE_LIMIT = 1_000_000

# FVM–VPM transfer
TRANSFER_INSET = 8.75  # cells between the outer FVM boundary and the transfer surface
BLEND_START = 2  # cells below the transfer surface
BLEND_END = 6  # cells below the transfer surface
FVM_SUBSTEPS = 16  # per particle step
COUPLING_ITERATIONS = 6  # maximum FVM solves per particle step
COUPLING_TOLERANCE = 1.0e-5  # boundary velocity and tangential-gradient tolerance
END_PERIODS = 25  # natural periods in water; the paper's 20 start from a developed flow

# Output
FORCE_INTERVAL = 0.04  # s, forces and cylinder motion
FIELD_INTERVAL = 0.2  # s, FVM and particle fields
FIELD_PERIODS = 2  # natural periods of fields before the end
BACKUP_INTERVAL = 1.0  # s
TRANSFER_DIAGNOSTIC_INTERVAL = 2.4  # s


def structure(reduced_velocity, span):
    """Mass, stiffness and damping in kg, N/m and N s/m of the modelled span."""
    displaced_mass = DENSITY * math.pi * DIAMETER**2 / 4.0 * span
    mass = MASS_RATIO * displaced_mass
    natural_frequency = FREESTREAM_VELOCITY[0] / (reduced_velocity * DIAMETER)  # Hz, in water
    stiffness = (2.0 * math.pi * natural_frequency) ** 2 * (
        mass + ADDED_MASS_COEFFICIENT * displaced_mass
    )
    damping = 2.0 * DAMPING_RATIO * math.sqrt(stiffness * mass)
    return mass, stiffness, damping


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


def build_case(reduced_velocity, end_time=None, compute_device="auto"):
    """Return the FVM case, the particle case and the coupling settings of one reduced velocity."""
    variant = f"vstar_{reduced_velocity:g}"
    reynolds_number = REYNOLDS_PER_REDUCED_VELOCITY * reduced_velocity
    kinematic_viscosity = FREESTREAM_VELOCITY[0] * DIAMETER / reynolds_number  # m^2/s
    natural_period = reduced_velocity * DIAMETER / FREESTREAM_VELOCITY[0]  # s, in water
    if end_time is None:
        end_time = END_PERIODS * natural_period
    field_start = max(0.0, end_time - FIELD_PERIODS * natural_period)

    # One FVM cell and one particle layer across the span make the flow planar.
    span = CELL_SIZE
    half_span = 0.5 * span
    mass, stiffness, damping = structure(reduced_velocity, span)

    # FVM domain, transfer region and coupling interval.
    margin = BOX_MARGIN * DIAMETER
    fvm_domain_bounds = (-margin, margin + 2.0 * DIAMETER, -margin, margin)
    inner = margin - TRANSFER_INSET * CELL_SIZE
    transfer_region = (-inner, inner + 2.0 * DIAMETER, -inner, inner)
    coupling_time_step_size = FVM_SUBSTEPS * FVM_TIME_STEP
    # Choose enough diffusion stages for the particle spacing and time step.
    fourier_number = kinematic_viscosity * coupling_time_step_size / CELL_SIZE**2
    diffusion_stages = math.floor(12.0 * fourier_number) + 1

    # Mesh a slab that the cylinder crosses, then retain one periodic spanwise cell.
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
    slab = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(*fvm_domain_bounds, -SLAB_HALF_CELLS * CELL_SIZE, SLAB_HALF_CELLS * CELL_SIZE),
            patches=patches,
        ),
        surfaces=(msh.STLSurface(CASE_DIR / "assets/cylinder_long.stl", patch="cylinder"),),
        max_cell_size=CELL_SIZE,
        # The exact lattice is anchored at the finest size; the interface keeps CELL_SIZE cells
        cell_size_anchor=WALL_CELL_SIZE,
        patch_refinements=(msh.PatchRefinement("cylinder", WALL_CELL_SIZE),),
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
        name="coupled_elastic_cylinder",
        directory=CASE_DIR,
        variant=variant,
        numerics=fvm.Numerics(
            time_step_size=FVM_TIME_STEP,
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
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
            schedule=fvm.EveryTime(FIELD_INTERVAL, start_time=field_start),
            compression="lz4",
            ghost_layers=0,
        ),
        samplers=fvm.Samplers(
            (
                # Drag and lift coefficients per unit span, and the power of the fluid on the body
                fvm.ForceSampler(
                    patch_names=["cylinder"],
                    reference_velocity=FREESTREAM_VELOCITY[0],
                    reference_area=DIAMETER * span,
                    reference_length=DIAMETER,
                    schedule=fvm.EveryTime(FORCE_INTERVAL),
                ),
                # Position and velocity of the cylinder axis
                fvm.MotionSampler(schedule=fvm.EveryTime(FORCE_INTERVAL)),
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
        # The whole FVM box rides on the springs of the cylinder
        mesh_motion=fvm.MeshMotion(
            fvm.ElasticMount(
                mass=mass,
                stiffness=stiffness,
                damping=damping,
                directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                patch_names=("cylinder",),
            )
        ),
    )

    particles = vpm.VPMCase(
        name="coupled_elastic_cylinder",
        directory=CASE_DIR,
        variant=variant,
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
                kinematic_viscosity=kinematic_viscosity,
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
        output=vpm.Output(schedule=vpm.EveryTime(FIELD_INTERVAL, start_time=field_start)),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduced-velocity",
        type=float,
        default=REDUCED_VELOCITIES[1],
        help="V* = U / (f_n D); the Reynolds number is 250 V*",
    )
    parser.add_argument(
        "--end-time", type=float, default=None, help="end time [s]; default 25 natural periods"
    )
    parser.add_argument(
        "--compute-device",
        default="auto",
        choices=("auto", "cpu", "cuda", "vulkan"),
        help="particle solver device; auto takes the GPU when there is one",
    )
    arguments = parser.parse_args()
    flow, particles, coupling = build_case(
        arguments.reduced_velocity, arguments.end_time, arguments.compute_device
    )
    with coupler.CoupledSolver(flow, particles, coupling) as solver:
        # The startup vortex behind the cylinder; a restart replaces it with the backup
        fvm_solver = solver.fvm_solver
        cell_centres = fvm_solver.geo_data["cell_centre"][: fvm_solver.mesh_data["n_cells"]]
        fvm_solver.set_initial_velocity(initial_velocity(cell_centres))
        solver.run(start_from="latest")
