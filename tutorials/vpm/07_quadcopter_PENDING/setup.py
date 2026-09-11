#!/usr/bin/env python3
"""Simulate the wake of a quadcopter in climb (VLM--VPM).

Four two-bladed rotors counter-rotate on a small quadcopter frame. The vehicle
climbs at constant speed while the rotors shed their wakes into the flow. The
native blade forces, shaft power and downstream velocity are sampled for
the figures made by ``allplot.sh``.

The induction backend and stretching formulation are independent. Set
`stretching_scheme` to "direct", "mixed", or "transposed" in the case below.

Usage:
    python setup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.tutorial_runner import case_package
from openonda.vpm import Backup, Samplers

__package__ = case_package(Path(__file__).parent)
from .assets.generate_blade import create_rotor_blade, save_blade

CASE_NAME = "quadcopter"

# Rotor and flow
ROTATIONS_PER_MINUTE = 4000.0
ANGULAR_VELOCITY = ROTATIONS_PER_MINUTE * 2.0 * np.pi / 60.0  # [rad/s]
TIP_RADIUS = 0.15
HUB_RADIUS = 0.03  # [m]
AIR_DENSITY = 1.225  # [kg/m^3]
KINEMATIC_VISCOSITY = 1.5e-5  # [m^2/s]
NUMBER_OF_BLADES = 2
CLIMB_SPEED = 0.8
ARM_LENGTH = 0.16

# Time resolution
DEGREES_PER_STEP = 3.75
TIME_STEP_SIZE = np.deg2rad(DEGREES_PER_STEP) / ANGULAR_VELOCITY  # [s]
STEPS_PER_REVOLUTION = round(360.0 / DEGREES_PER_STEP)
NUMBER_OF_REVOLUTIONS = 24
N_STEPS = NUMBER_OF_REVOLUTIONS * STEPS_PER_REVOLUTION
SAMPLE_INTERVAL_TIME = 12 * TIME_STEP_SIZE  # eight field snapshots per revolution
BACKUP_INTERVAL_TIME = 3 * TIME_STEP_SIZE  # 32 coupled backups per revolution

# Process-resource guards for the long authored demo.  These are lifecycle
# limits, distinct from numerical health criteria, and preserve a restartable
# accepted state when a bound is reached.
MAX_PARTICLES = 450_000
MAX_RSS_BYTES = 12 * 2**30
MIN_AVAILABLE_MEMORY_BYTES = 2 * 2**30

WAKE_PLANES = (("sampled_zplane", -0.35), ("sampled_zplane_deep", -0.70))


TUTORIAL_DIR = Path(__file__).resolve().parent


def run() -> None:
    counterclockwise_file = TUTORIAL_DIR / "assets" / "blade_ccw.json"
    clockwise_file = TUTORIAL_DIR / "assets" / "blade_cw.json"
    blade_parameters = {
        "R_hub": HUB_RADIUS,
        "R_tip": TIP_RADIUS,
        "chord_root": 0.025,
        "chord_tip": 0.015,
        "pitch_root_deg": 12.0,
        "pitch_tip_deg": 6.0,
        "n_chord": 4,
        "n_span": 12,
    }
    save_blade(
        create_rotor_blade(**blade_parameters, clockwise=False),
        str(counterclockwise_file),
    )
    save_blade(
        create_rotor_blade(**blade_parameters, clockwise=True),
        str(clockwise_file),
    )

    rotors = (
        ("rotor_0", [ARM_LENGTH, ARM_LENGTH, 0.0], 1.0),
        ("rotor_1", [-ARM_LENGTH, ARM_LENGTH, 0.0], -1.0),
        ("rotor_2", [-ARM_LENGTH, -ARM_LENGTH, 0.0], 1.0),
        ("rotor_3", [ARM_LENGTH, -ARM_LENGTH, 0.0], -1.0),
    )
    vlm_setup = vpm.VLMSetup(
        surfaces=tuple(
            vpm.VLMSurfaceSetup(
                str(counterclockwise_file if direction > 0 else clockwise_file),
                name=f"{name}_blade_{blade_index}",
                kinematics=vpm.RotatingVLM(
                    angular_speed=ANGULAR_VELOCITY * direction,
                    axis=[0.0, 0.0, 1.0],
                    rotation_centre=position,
                ),
                translation=tuple(position),
                rotation_degrees=(0.0, 0.0, 360.0 / NUMBER_OF_BLADES * blade_index),
                group_id=rotor_index + 1,
            )
            for rotor_index, (name, coordinates, direction) in enumerate(rotors)
            for position in (np.array(coordinates),)
            for blade_index in range(NUMBER_OF_BLADES)
        ),
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        sigma_factor=2.5,
        sample_surface_forces=True,
        # Attached VLM loading is recorded on every accepted VPM step.
        logging_interval_steps=1,
    )

    sample_steps = round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE)
    case = vpm.VPMCase(
        name=CASE_NAME,
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device="AUTO",
            vlm=vlm_setup,
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
            ),
            induction=vpm.TreecodeInduction(
                stretching_scheme="transposed", theta=0.3, multipole_order=3
            ),
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(smagorinsky_coefficient=0.17),
            particle_kernel="WINCKELMANS",
            freestream_velocity=[0.0, 0.0, -CLIMB_SPEED],
            stabilization=vpm.StabilizationConfig(
                remove_particles_by_bounds=[-1.5, 1.5, -1.5, 1.5, -3.0, 1.0]
            ),
            max_n_particles=500_000,
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
                *(
                    vpm.SurfaceSampler(
                        point=[0.0, 0.0, height],
                        normal=[0.0, 0.0, 1.0],
                        bounds=[-0.4, 0.4, -0.4, 0.4],
                        spacing=0.0075,
                        file_name=plane_name,
                        include_derivatives=False,
                        schedule=vpm.EverySteps(sample_steps),
                    )
                    for plane_name, height in WAKE_PLANES
                ),
            ),
            directory=CASE_NAME,
        ),
        run=vpm.RunPlan(
            steps=N_STEPS,
            health_limit_action="STOP",
            resource_limits=vpm.ResourceLimits(
                max_particles=MAX_PARTICLES,
                max_rss_bytes=MAX_RSS_BYTES,
                min_available_memory_bytes=MIN_AVAILABLE_MEMORY_BYTES,
            ),
        ),
        directory=TUTORIAL_DIR,
    )
    vpm.VPMSolver(case).run()


if __name__ == "__main__":
    run()
