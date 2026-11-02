#!/usr/bin/env python3
"""Coupled FVM–VPM flow through a three-bladed wind turbine with fully meshed blades.

The rotor is the one of the VLM tutorial ``tutorials/vpm/06_rotor_flow``: the
same blade planform, twist, radius, wind speed, tip-speed ratio, time step,
end time and output cadence, so that power and blade forces compare blade for
blade. The blades carry a NACA 0012 thickness and are meshed inside a flat box,
a disk normal to the rotor axis, which turns with the rotor; the vortex
particles carry the wake in the inertial frame.
"""

import argparse
import math
from pathlib import Path

import numpy as np

import openonda.coupler as coupler
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm

from assets.blade_surface import write_blade_stl

CASE_DIR = Path(__file__).resolve().parent
# The VLM blade, planform and twist: a copy of tutorials/vpm/06_rotor_flow/assets/blade.json
BLADE_FILE = CASE_DIR / "assets/blade.json"

# Operating point, as the VLM rotor
FREESTREAM_SPEED = 7.0  # m/s, along +x
TIP_SPEED_RATIO = 7.0
ROTOR_RADIUS = 6.0  # m
KINEMATIC_VISCOSITY = 1.5e-5  # m^2/s
AIR_DENSITY = 1.225  # kg/m^3
ANGULAR_VELOCITY = TIP_SPEED_RATIO * FREESTREAM_SPEED / ROTOR_RADIUS  # rad/s
BLADE_AZIMUTHS = (0.0, 120.0, 240.0)  # degrees about the rotor axis
# NACA 0018: the 12 % section's tip, 0.042 m thick, collapsed cells in the mesher's wall
# projection; the 18 % section keeps the tip more than two cells thick
BLADE_THICKNESS = 0.18  # chord fraction
N_SECTION_POINTS = 61  # per surface of a blade section

# Time, as the VLM rotor: the coupling interval is its particle step
COUPLING_TIME_STEP = 0.006  # s
END_TIME = 15.0  # s
FIELD_SAMPLE_INTERVAL = 0.06  # s, fields, wake planes and lines
BACKUP_INTERVAL = 0.024  # s
FVM_SUBSTEPS = 12  # per coupling interval; the tip cells see Courant numbers near one

# FVM disk and mesh
CELL_SIZE = 0.2  # m, at the coupling interface; also the particle spacing
BLADE_CELL_SIZE = 0.025  # m, three octree levels finer at the blade walls
DISK_HALF_THICKNESS = 2.4  # m, along the rotor axis; the blades stay eight background cells inside
DISK_HALF_WIDTH = 9.0  # m, normal to the axis; the blade tips stay eight background cells inside

# VPM wake
VPM_DOMAIN = (-6.0, 60.0, -14.0, 14.0, -14.0, 14.0)  # m, holds the turning disk
OUTFLOW_FRINGE = 15.0  # m, smooth vorticity removal near the downstream boundary
VORTICITY_FLOOR = 0.05  # 1/s, remove weak diffused particles
PARTICLE_LIMIT = 600_000
CORE_RADIUS_RATIO = 1.0

# Sampling, as the VLM rotor
STREAMWISE_STATIONS = (("025", 0.25), ("065", 0.65))
WAKE_PLANE_DIAMETERS = (0, 1, 2)
TRANSFER_DIAGNOSTIC_INTERVAL = 0.6  # s


def build_case(end_time=END_TIME, compute_device="auto"):
    """Return the FVM case, the particle case and the coupling settings."""
    fvm_time_step = COUPLING_TIME_STEP / FVM_SUBSTEPS
    rotor_motion = fvm.RotatingMotion(
        angular_speed=-ANGULAR_VELOCITY,
        rotation_axis=(1.0, 0.0, 0.0),
        acceleration_time=2.0 * math.pi / ANGULAR_VELOCITY,
    )

    # The blades as closed surfaces on the chord lines of the VLM blade
    surfaces = tuple(
        msh.STLSurface(
            write_blade_stl(
                CASE_DIR / f"assets/blade_{index}.stl",
                BLADE_FILE,
                BLADE_THICKNESS,
                N_SECTION_POINTS,
                azimuth,
            ),
            patch=f"blade_{index}",
        )
        for index, azimuth in enumerate(BLADE_AZIMUTHS)
    )
    patches = msh.BoxPatches(
        xmin="numericalBoundary",
        xmax="numericalBoundary",
        ymin="numericalBoundary",
        ymax="numericalBoundary",
        zmin="numericalBoundary",
        zmax="numericalBoundary",
        # All six faces form the coupling interface, meshed as exact Cartesian planes.
        coupling=("numericalBoundary",),
    )
    mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(
                -DISK_HALF_THICKNESS,
                DISK_HALF_THICKNESS,
                -DISK_HALF_WIDTH,
                DISK_HALF_WIDTH,
                -DISK_HALF_WIDTH,
                DISK_HALF_WIDTH,
            ),
            patches=patches,
        ),
        surfaces=surfaces,
        max_cell_size=CELL_SIZE,
        # The exact lattice is anchored at the finest size; the interface keeps CELL_SIZE cells
        cell_size_anchor=BLADE_CELL_SIZE,
        patch_refinements=tuple(
            msh.PatchRefinement(surface.patch, BLADE_CELL_SIZE) for surface in surfaces
        ),
    )

    flow = fvm.FVMCase(
        name="coupled_rotor",
        directory=CASE_DIR,
        numerics=fvm.Numerics(
            time_step_size=fvm_time_step,
            transport=fvm.TransportConfig(
                density=AIR_DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY
            ),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
            linear=fvm.LinearSolverConfig(
                pressure_tolerance=1.0e-7, pressure_relative_tolerance=0.005
            ),
            # Three outer passes converge the steps of the turning mesh
            pimple=fvm.PimpleConfig(
                n_outer_correctors=3, velocity_relaxation=0.7, pressure_relaxation=0.3
            ),
            turbulence=fvm.TurbulenceConfig.equilibrium_smagorinsky(),
            execution=fvm.ComputeConfig(operator_backend="numba"),
        ),
        initial_conditions=fvm.InitialFields(velocity=(FREESTREAM_SPEED, 0.0, 0.0)),
        run=fvm.RunPlan(end_time=end_time),
        output=fvm.Output(
            schedule=fvm.EveryTime(FIELD_SAMPLE_INTERVAL), compression="lz4", ghost_layers=0
        ),
        samplers=fvm.Samplers(
            (
                # Thrust, torque and power of each blade; coefficients on the disk area
                fvm.ForceSampler(
                    patch_names=[surface.patch for surface in surfaces],
                    reference_velocity=FREESTREAM_SPEED,
                    reference_area=math.pi * ROTOR_RADIUS**2,
                    reference_length=ROTOR_RADIUS,
                    moment_centre=(0.0, 0.0, 0.0),
                    file_name="blade_forces",
                    schedule=fvm.EveryTime(COUPLING_TIME_STEP),
                ),
            )
        ),
        logging=fvm.LoggingConfig(console=False),
        mesh=mesh,
        boundaries=(
            # Particles supply the normal velocity and the normal derivative of tangential velocity
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_value=(FREESTREAM_SPEED, 0.0, 0.0),
                pressure_type="fixed_flux_pressure",
            ),
            # Walls on the turning mesh turn with it
            *(fvm.BoundaryConfig.wall(surface.patch) for surface in surfaces),
        ),
        # The whole disk turns with the rotor, spinning up over one revolution as the VLM rotor
        mesh_motion=fvm.MeshMotion(rotor_motion),
    )

    plane_half_width = 0.65 * (2.0 * ROTOR_RADIUS)
    sample_spacing = ROTOR_RADIUS / 36
    plane_schedule = vpm.EveryTime(FIELD_SAMPLE_INTERVAL)
    particles = vpm.VPMCase(
        name="coupled_rotor",
        directory=CASE_DIR,
        numerics=vpm.Numerics(
            time_step_size=COUPLING_TIME_STEP,
            compute_device=compute_device,
            freestream_velocity=(FREESTREAM_SPEED, 0.0, 0.0),
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
                VPM_DOMAIN, outflow_fringe_depth=(0.0, OUTFLOW_FRINGE, 0.0, 0.0, 0.0, 0.0)
            ),
            domain_bounds=VPM_DOMAIN,
            max_n_particles=PARTICLE_LIMIT,
        ),
        run=vpm.RunPlan(end_time=end_time, state_limit_action="stop"),
        output=vpm.Output(schedule=vpm.EveryTime(FIELD_SAMPLE_INTERVAL)),
        samplers=vpm.Samplers(
            (
                vpm.FlowIntegralsSampler(schedule=plane_schedule),
                # Wake planes at the disk and one and two diameters downstream
                *(
                    vpm.SurfaceSampler(
                        point=[2.0 * ROTOR_RADIUS * distance, 0.0, 0.0],
                        normal=[1, 0, 0],
                        bounds=[
                            -plane_half_width,
                            plane_half_width,
                            -plane_half_width,
                            plane_half_width,
                        ],
                        spacing=sample_spacing,
                        file_name=f"wake_{distance}D",
                        include_derivatives=False,
                        schedule=plane_schedule,
                    )
                    for distance in WAKE_PLANE_DIAMETERS
                ),
                # Streamwise lines at two radial stations
                *(
                    vpm.LineSampler(
                        start=[-2.0 * ROTOR_RADIUS, radial_fraction * ROTOR_RADIUS, 0.0],
                        end=[6.0 * ROTOR_RADIUS, radial_fraction * ROTOR_RADIUS, 0.0],
                        spacing=sample_spacing,
                        file_name=f"streamwise_r{label}",
                        include_derivatives=False,
                        schedule=plane_schedule,
                    )
                    for label, radial_fraction in STREAMWISE_STATIONS
                ),
            )
        ),
        logging=vpm.LoggingConfig(console=False),
    )

    coupling = coupler.CouplerConfig(
        freestream_velocity=(FREESTREAM_SPEED, 0.0, 0.0),
        transfer_diagnostic_interval_steps=round(TRANSFER_DIAGNOSTIC_INTERVAL / COUPLING_TIME_STEP),
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
