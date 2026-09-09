#!/usr/bin/env python3
"""Wind turbine with a coupled VLM--VPM wake (LES).

A three-bladed turbine extracts energy at a tip-speed ratio of 7.0. The wake is resolved
with vortex particles whose position and strength use common Runge--Kutta stages;
the blade loading and the downstream wake planes are sampled for the
``allplot.sh`` figures.

The induction backend and stretching formulation are independent. Set
`stretching_scheme` to "direct", "mixed", or "transposed" in the case below.

Usage:
    python setup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

CASE_NAME = "rotor"

FREESTREAM_SPEED = 7.0  # [m/s]
TIP_SPEED_RATIO = 7.0
ROTOR_RADIUS = 6.0  # [m]
HUB_RADIUS = 1.0  # [m]
KINEMATIC_VISCOSITY = 1.5e-5  # [m^2/s]
AIR_DENSITY = 1.225  # [kg/m^3]
N_RADIAL_STATIONS = 23
MAX_N_PARTICLES = 400_000
ANGULAR_VELOCITY = TIP_SPEED_RATIO * FREESTREAM_SPEED / ROTOR_RADIUS  # [rad/s]

TIME_STEP_SIZE = 0.006  # [s]
END_TIME = 14.4  # [s]
N_STEPS = round(END_TIME / TIME_STEP_SIZE)
DEFAULT_SMAGORINSKY_COEFFICIENT = 0.17
RAMP_ROTATIONS = 1.0
SAMPLE_INTERVAL_TIME = 0.06  # write a snapshot every this many seconds
FORCE_INTERVAL_TIME = 0.012  # about 64 force samples per revolution
ROTATION_PERIOD = 2.0 * np.pi / ANGULAR_VELOCITY  # [s]
BACKUP_INTERVAL_TIME = ROTATION_PERIOD  # one checkpoint per revolution
PLANE_SAMPLING_ROTATIONS = 6.0


TUTORIAL_DIR = Path(__file__).resolve().parent


FIXED_WAKE_SPACING = (ROTOR_RADIUS - HUB_RADIUS) / (N_RADIAL_STATIONS - 1)


def run() -> None:
    blade_file = TUTORIAL_DIR / "assets/blade.json"

    rotation_period = 2.0 * np.pi / ANGULAR_VELOCITY
    ramp_time = RAMP_ROTATIONS * rotation_period

    rotation_kinematics = vpm.RotatingVLM(
        angular_speed=-ANGULAR_VELOCITY,
        axis=[1.0, 0.0, 0.0],
        acceleration_time=ramp_time,
    )

    vlm_setup = vpm.VLMSetup(
        surfaces=tuple(
            vpm.VLMSurfaceSetup(
                str(blade_file),
                name=f"blade_{blade_index}",
                kinematics=rotation_kinematics,
                rotation_degrees=(azimuth, 0.0, 0.0),
            )
            for blade_index, azimuth in enumerate((0.0, 120.0, 240.0))
        ),
        mesh=vpm.VLMMeshSetup.geometric(ratio=3.0),
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        wake_core_overlap=2.5,
        sample_surface_forces=True,
        logging_interval_steps=round(FORCE_INTERVAL_TIME / TIME_STEP_SIZE),
    )

    # Downstream planes at 1.5R, 3R, and 4.5R.
    off_wake = ROTOR_RADIUS * 1.2
    sample_spacing = ROTOR_RADIUS / 36
    plane_schedule = vpm.EverySteps(
        round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE),
        start_time=max(0.0, END_TIME - PLANE_SAMPLING_ROTATIONS * ROTATION_PERIOD),
    )
    plane_samplers = [
        vpm.SurfaceSampler(
            point=[x_loc, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-off_wake, off_wake, -off_wake, off_wake],
            spacing=sample_spacing,
            file_name=f"slice_x{int(round(x_loc))}m",
            include_derivatives=False,
            schedule=plane_schedule,
        )
        for x_loc in [1.5 * ROTOR_RADIUS, 3.0 * ROTOR_RADIUS, 4.5 * ROTOR_RADIUS]
    ]

    sample_steps = round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE)
    case = vpm.VPMCase(
        name=CASE_NAME,
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device="AUTO",
            integrator=vpm.SSPRK3(),
            vlm=vlm_setup,
            freestream_velocity=[FREESTREAM_SPEED, 0.0, 0.0],
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=DEFAULT_SMAGORINSKY_COEFFICIENT
            ),
            stabilization=vpm.StabilizationConfig(
                pedrizzetti_relaxation_factor=0.3,
                remove_particles_by_bounds=[
                    -2.0 * ROTOR_RADIUS,
                    20.0 * ROTOR_RADIUS,
                    -2.0 * ROTOR_RADIUS,
                    2.0 * ROTOR_RADIUS,
                    -2.0 * ROTOR_RADIUS,
                    2.0 * ROTOR_RADIUS,
                ],
            ),
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                particle_spacing=FIXED_WAKE_SPACING,
            ),
            induction=vpm.TreecodeInduction(
                stretching_scheme="transposed", theta=0.3, multipole_order=3
            ),
            particle_kernel="GAUSSIAN",
            max_n_particles=MAX_N_PARTICLES,
            write_precision="f32",
        ),
        backup=Backup(
            interval_steps=round(BACKUP_INTERVAL_TIME / TIME_STEP_SIZE),
            directory="solution",
            log_directory="solution",
        ),
        samplers=Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(sample_steps)),
                vpm.VLMSampler(schedule=vpm.EverySteps(sample_steps)),
                *plane_samplers,
            ),
            directory=CASE_NAME,
        ),
        run=vpm.RunPlan(steps=N_STEPS),
        directory=TUTORIAL_DIR,
    )
    vpm.VPMSolver(case).run()


if __name__ == "__main__":
    run()
