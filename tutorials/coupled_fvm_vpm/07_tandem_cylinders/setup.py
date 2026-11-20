#!/usr/bin/env python3
"""Coupled FVM–VPM flow past two cylinders in tandem at Re = 150.

Two cylinders of diameter D have their centres 8 D apart along the stream. Each
cylinder carries the FVM domain, mesh, time steps and transfer settings of the
baseline run of case 01 (``01_cylinder_shedding_flow``). The two FVM domains do
not touch: the particles alone carry the wake of the upstream cylinder across
the gap to the downstream one, whose outer boundary takes its data from the
same particle field. With ``--heave-frequency`` the downstream cylinder heaves
across the stream, y = 0.2 D sin(2 pi f t), on its own moving mesh.
"""

import argparse
import math
from pathlib import Path

import numpy as np

import openonda.coupler as coupler
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm

from assets.cylinder_surface import CYLINDER_STL, write_cylinder_stl

CASE_DIR = Path(__file__).resolve().parent

# Flow, as case 01
DIAMETER = 1.0  # m
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)  # m/s
REYNOLDS_NUMBER = 150.0
DENSITY = 1.0  # kg/m^3

# Cylinders: centres along the stream
CENTRE_SPACING = 8.0  # D
CYLINDERS = (("upstream", (0.0, 0.0)), ("downstream", (CENTRE_SPACING * DIAMETER, 0.0)))  # m
HEAVE_AMPLITUDE = 0.2  # D, of the downstream cylinder with --heave-frequency

# Startup vortex behind each cylinder, as case 01: breaks wake symmetry and starts shedding
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, streamfunction peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = 0.65  # m, downstream of the cylinder axis

# FVM mesh and near wake of each cylinder: the baseline of case 01
CELL_SIZE = 0.04  # m, also the particle spacing
SLAB_HALF_DEPTH_CELLS = 12  # cells, the meshed slab of case 01 at this cell size
BOX_MARGIN = 2.4  # D upstream and sideways; 2 D farther downstream
FVM_TIME_STEP = 0.008  # s
FVM_SUBSTEPS = 5  # per particle step
FVM_OUTER_CORRECTORS = 2

# VPM: particles in the outer wake; the fringe starts 15 D behind the downstream cylinder
VPM_DOMAIN = (-5.0, 38.0, -5.0, 5.0)  # m, x and y bounds
OUTFLOW_FRINGE = 15.0  # m, smooth vorticity removal near the downstream boundary
VORTICITY_FLOOR = 0.01  # 1/s, remove weak diffused particles
PARTICLE_LIMIT = 2_000_000

# FVM–VPM transfer of each cylinder: the baseline of case 01
TRANSFER_INSET = 8.75  # cells between the outer FVM boundary and the transfer surface
BLEND_START = 2  # cells below the transfer surface
BLEND_END = 6  # cells below the transfer surface
COUPLING_ITERATIONS = 6  # maximum FVM solves per particle step
COUPLING_TOLERANCE = 1.0e-5  # boundary velocity and tangential-gradient tolerance
FIRST_ESTIMATE = "linear"  # boundary data at the end of an interval
TRANSFER_REMOVAL_THRESHOLD = 0.05  # 1/s, removal threshold of the replacement region
MOMENT_CORRECTION_FRACTION = 0.08  # largest moment correction, fraction of transferred strength
END_TIME = 200.0  # s, 200 D/U

# Output: forces every 0.04 s and wake lines every 0.12 s (over 100 and 40 samples per shedding
# period of about 6 s), fields over the last two periods and more, particle integrals every interval
FORCE_INTERVAL = 0.04  # s
PROFILE_INTERVAL = 0.12  # s, transverse lines 2 and 4 D behind each cylinder
# FVM and particle fields every five coupling intervals over the statistics window: 501 frames,
# for a video of 10 s at 50 frames per second
FIELD_INTERVAL = 0.2  # s
FIELD_START = 100.0  # s
BACKUP_INTERVAL = 1.0  # s
TRANSFER_DIAGNOSTIC_INTERVAL = 2.4  # s


def initial_velocity(position, centre):
    """Freestream plus the curl of psi = A (1 - r^2/R^2)^4 behind the cylinder at ``centre``."""
    x = position[:, 0] - centre[0] - DISTURBANCE_CENTRE
    y = position[:, 1] - centre[1]
    support = np.maximum(1.0 - (x**2 + y**2) / DISTURBANCE_RADIUS**2, 0.0)
    swirl = 8.0 * DISTURBANCE_AMPLITUDE / DISTURBANCE_RADIUS**2 * support**3

    velocity = np.tile(FREESTREAM_VELOCITY, (len(position), 1))
    velocity[:, 0] -= swirl * y
    velocity[:, 1] += swirl * x
    return velocity


def build_cylinder(
    name,
    centre,
    run_name,
    end_time=END_TIME,
    heave_frequency=None,
    box_margin=BOX_MARGIN,
    directory=CASE_DIR,
):
    """Return the FVM case and the coupling settings of the cylinder at ``centre`` (x, y) in m."""
    coupling_time_step = FVM_SUBSTEPS * FVM_TIME_STEP
    kinematic_viscosity = FREESTREAM_VELOCITY[0] * DIAMETER / REYNOLDS_NUMBER  # m^2/s
    half_span = 0.5 * CELL_SIZE  # one FVM cell and one particle layer across the span

    def on_coupling_times(interval):
        return max(1, round(interval / coupling_time_step)) * coupling_time_step

    # FVM domain and transfer region of case 01, centred on this cylinder
    centre_x, centre_y = centre
    margin = box_margin * DIAMETER
    bounds = (
        centre_x - margin,
        centre_x + margin + 2.0 * DIAMETER,
        centre_y - margin,
        centre_y + margin,
    )
    inset = TRANSFER_INSET * CELL_SIZE
    transfer_region = (bounds[0] + inset, bounds[1] - inset, bounds[2] + inset, bounds[3] - inset)

    # Mesh a slab through the long cylinder, then keep one periodic cell across the span
    surface = (
        CYLINDER_STL
        if centre == (0.0, 0.0)
        else write_cylinder_stl(Path(directory) / f"assets/cylinder_{name}.stl", centre)
    )
    patches = msh.BoxPatches(
        xmin="numericalBoundary",
        xmax="numericalBoundary",
        ymin="numericalBoundary",
        ymax="numericalBoundary",
        zmin="zmin",
        zmax="zmax",
        coupling=("numericalBoundary",),
    )
    slab_half_depth = SLAB_HALF_DEPTH_CELLS * CELL_SIZE
    slab = msh.CartesianMesher(
        domain=msh.BoxDomain(bounds=(*bounds, -slab_half_depth, slab_half_depth), patches=patches),
        surfaces=(msh.STLSurface(surface, patch="cylinder"),),
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

    samplers = [
        # Pressure and viscous forces and moments on the cylinder, per unit span
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=FREESTREAM_VELOCITY[0],
            reference_area=DIAMETER * 2.0 * half_span,
            reference_length=DIAMETER,
            moment_centre=(centre_x, centre_y, 0.0),
            schedule=fvm.EveryTime(on_coupling_times(FORCE_INTERVAL)),
        ),
        # The near-wake profile of case 01, 2 D behind the cylinder
        fvm.LineSampler(
            (centre_x + 2.0, centre_y - 1.5, 0.0),
            (centre_x + 2.0, centre_y + 1.5, 0.0),
            n_points=31,
            n_neighbours=12,
            reconstruction="affine",
            file_name="fvm_transverse_x2D",
            schedule=fvm.EveryTime(on_coupling_times(PROFILE_INTERVAL)),
        ),
    ]
    motion = None
    if heave_frequency is not None:
        motion = fvm.MeshMotion(
            fvm.HeavingMotion(
                amplitude=HEAVE_AMPLITUDE * DIAMETER,
                frequency=heave_frequency,
                direction=(0.0, 1.0, 0.0),
            )
        )
        samplers.append(
            fvm.MotionSampler(schedule=fvm.EveryTime(on_coupling_times(FORCE_INTERVAL)))
        )
    flow = fvm.FVMCase(
        name=name,
        directory=directory,
        variant=f"{run_name}/{name}",
        numerics=fvm.Numerics(
            time_step_size=FVM_TIME_STEP,
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
            linear=fvm.LinearSolverConfig(
                pressure_tolerance=1.0e-7, pressure_relative_tolerance=0.005
            ),
            pimple=fvm.PimpleConfig(
                n_outer_correctors=FVM_OUTER_CORRECTORS,
                velocity_relaxation=0.7,
                pressure_relaxation=0.3,
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
        samplers=fvm.Samplers(tuple(samplers)),
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
        mesh_motion=motion,
    )
    coupling = coupler.CouplerConfig(
        freestream_velocity=FREESTREAM_VELOCITY,
        transfer_region_bounds=(*transfer_region, -half_span, half_span),
        blend_start_depth=BLEND_START * CELL_SIZE,
        blend_end_depth=BLEND_END * CELL_SIZE,
        max_coupling_iterations=COUPLING_ITERATIONS,
        normal_velocity_tolerance=COUPLING_TOLERANCE,
        normal_gradient_tolerance=COUPLING_TOLERANCE,
        boundary_data_extrapolation=FIRST_ESTIMATE,
        transfer_removal_threshold=TRANSFER_REMOVAL_THRESHOLD,
        max_moment_correction_fraction=MOMENT_CORRECTION_FRACTION,
        transfer_diagnostic_interval_steps=round(TRANSFER_DIAGNOSTIC_INTERVAL / coupling_time_step),
        backup=coupler.Backup(schedule=coupler.EveryTime(on_coupling_times(BACKUP_INTERVAL))),
    )
    return flow, coupling


def build_particles(
    run_name,
    cylinders=CYLINDERS,
    end_time=END_TIME,
    vpm_domain=VPM_DOMAIN,
    compute_device="auto",
    directory=CASE_DIR,
):
    """Return the particle case of the outer wake of ``cylinders``, (name, centre) pairs."""
    kinematic_viscosity = FREESTREAM_VELOCITY[0] * DIAMETER / REYNOLDS_NUMBER  # m^2/s
    coupling_time_step = FVM_SUBSTEPS * FVM_TIME_STEP
    half_span = 0.5 * CELL_SIZE

    def on_coupling_times(interval):
        return max(1, round(interval / coupling_time_step)) * coupling_time_step

    # Diffusion stages for the particle spacing and time step, as case 01
    fourier_number = kinematic_viscosity * coupling_time_step / CELL_SIZE**2
    diffusion_stages = math.floor(12.0 * fourier_number) + 1
    profiles = vpm.EveryTime(on_coupling_times(PROFILE_INTERVAL))
    return vpm.VPMCase(
        name="coupled_tandem_cylinders",
        directory=directory,
        variant=run_name,
        numerics=vpm.Numerics(
            time_step_size=coupling_time_step,
            compute_device=compute_device,
            freestream_velocity=FREESTREAM_VELOCITY,
            integrator=vpm.RK2(),
            # Full 3D induction of the particle layer and all its images in the span planes
            induction=vpm.SlipSlabInduction(vpm.FMMInduction(), z_min=-half_span, z_max=half_span),
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
                (*vpm_domain, -half_span, half_span),
                outflow_fringe_depth=(0.0, OUTFLOW_FRINGE, 0.0, 0.0, 0.0, 0.0),
            ),
            domain_bounds=(*vpm_domain, -half_span, half_span),
            max_n_particles=PARTICLE_LIMIT,
        ),
        run=vpm.RunPlan(end_time=on_coupling_times(end_time)),
        output=vpm.Output(
            schedule=vpm.EveryTime(
                on_coupling_times(FIELD_INTERVAL), start_time=on_coupling_times(FIELD_START)
            )
        ),
        samplers=vpm.Samplers(
            (
                # Total strength, impulse, count and fringe removal every interval
                vpm.FlowIntegralsSampler(schedule=vpm.EveryTime(coupling_time_step)),
                # Transverse lines 2 and 4 D behind each cylinder
                *(
                    vpm.LineSampler(
                        (centre[0] + distance, centre[1] - 2.0, 0.0),
                        (centre[0] + distance, centre[1] + 2.0, 0.0),
                        spacing=0.1,
                        include_derivatives=True,
                        file_name=f"{name}_transverse_x{distance:g}D",
                        schedule=profiles,
                    )
                    for name, centre in cylinders
                    for distance in (2.0, 4.0)
                ),
            )
        ),
        logging=vpm.LoggingConfig(console=False),
    )


def build_case(end_time=END_TIME, heave_frequency=None, compute_device="auto"):
    """Return the FVM cases and coupling settings of both cylinders and the particle case."""
    run_name = "fixed" if heave_frequency is None else "heaving"
    domains = [
        build_cylinder(
            name, centre, run_name, end_time, heave_frequency if name == "downstream" else None
        )
        for name, centre in CYLINDERS
    ]
    flows, couplings = zip(*domains, strict=True)
    particles = build_particles(run_name, CYLINDERS, end_time, compute_device=compute_device)
    return flows, particles, couplings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float, default=END_TIME, help="end time [s]")
    parser.add_argument(
        "--heave-frequency",
        type=float,
        help="heave frequency of the downstream cylinder [Hz]; fixed cylinders without it",
    )
    parser.add_argument(
        "--compute-device",
        default="auto",
        choices=("auto", "cpu", "cuda", "vulkan"),
        help="particle solver device; auto takes the GPU when there is one",
    )
    arguments = parser.parse_args()
    flows, particles, couplings = build_case(
        arguments.end_time, arguments.heave_frequency, arguments.compute_device
    )
    with coupler.CoupledSolver(flows, particles, couplings) as solver:
        # The startup vortex behind each cylinder; a restart replaces it with the backup
        for domain, (_, centre) in zip(solver.domains, CYLINDERS, strict=True):
            fvm_solver = domain.fvm_solver
            cell_centres = fvm_solver.geo_data["cell_centre"][: fvm_solver.mesh_data["n_cells"]]
            fvm_solver.set_initial_velocity(initial_velocity(cell_centres, centre))
        solver.run(start_from="latest")
