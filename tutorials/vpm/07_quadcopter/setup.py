#!/usr/bin/env python3
"""Simulate the wake of a quadcopter in climb (VLM--VPM).

Four two-bladed rotors counter-rotate on a small quadcopter frame. The vehicle
climbs at constant speed while the rotors shed their wakes into the flow. The
native blade forces, shaft power and downstream velocity are sampled for
the figures made by ``allplot.sh``.

To change the stretching formulation, set the induction's `stretching_scheme` to
"direct", "mixed" or "transposed".

Usage:
    python setup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openonda.vpm as vpm

from assets.generate_blade import create_rotor_blade, save_blade


CASE_NAME = "quadcopter"

# Rotor and flow
ROTATIONS_PER_MINUTE = 4000.0
ANGULAR_VELOCITY = ROTATIONS_PER_MINUTE * 2.0 * np.pi / 60.0  # [rad/s]
TIP_RADIUS = 0.15
HUB_RADIUS = 0.03  # [m]
AIR_DENSITY = 1.225  # [kg/m^3]
KINEMATIC_VISCOSITY = 1.5e-5  # [m^2/s]
NUMBER_OF_BLADES = 2
CLIMB_SPEED = 5.0
ARM_LENGTH = 0.18

# Time resolution
DEGREES_PER_STEP = 5.0
TIME_STEP_SIZE = np.deg2rad(DEGREES_PER_STEP) / ANGULAR_VELOCITY  # [s]
STEPS_PER_REVOLUTION = round(360.0 / DEGREES_PER_STEP)
NUMBER_OF_REVOLUTIONS = 20
END_TIME = NUMBER_OF_REVOLUTIONS * STEPS_PER_REVOLUTION * TIME_STEP_SIZE  # [s]
SAMPLE_INTERVAL_TIME = 12 * TIME_STEP_SIZE  # six field snapshots per revolution
INTEGRAL_INTERVAL_TIME = STEPS_PER_REVOLUTION * TIME_STEP_SIZE
BACKUP_INTERVAL_TIME = 2 * STEPS_PER_REVOLUTION * TIME_STEP_SIZE

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
        "pitch_root_deg": 28.0,
        "pitch_tip_deg": 12.0,
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
    vlm_setup = vpm.VLMConfig(
        surfaces=tuple(
            vpm.VLMSurface(
                str(counterclockwise_file if direction > 0 else clockwise_file),
                name=f"{name}_blade_{blade_index}",
                kinematics=vpm.RotatingMotion(
                    angular_speed=ANGULAR_VELOCITY * direction,
                    rotation_axis=(0.0, 0.0, 1.0),
                    rotation_centre=tuple(position),
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
        force=vpm.ForceConfig.kutta_joukowski(unsteady=False),
    )

    sample_steps = round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE)
    integral_steps = round(INTEGRAL_INTERVAL_TIME / TIME_STEP_SIZE)
    backup_schedule = vpm.EverySteps(round(BACKUP_INTERVAL_TIME / TIME_STEP_SIZE))
    case = vpm.VPMCase(
        name=CASE_NAME,
        directory=TUTORIAL_DIR,
        variant=CASE_NAME,
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            vlm=vlm_setup,
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
            ),
            induction=vpm.FMMInduction(),
            turbulence=vpm.TurbulenceConfig.smagorinsky(smagorinsky_coefficient=0.3),
            freestream_velocity=[0.0, 0.0, -CLIMB_SPEED],
            stabilization=vpm.StabilizationConfig(
                selective_eddy_viscosity_coefficient=2.0,
                pedrizzetti_relaxation_factor=0.1,
                pedrizzetti_relaxation_preserve_vortex_strength=False,
                pedrizzetti_relaxation_preserve_moments=True,
                remove_particles_by_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 0.3],
            ),
            max_n_particles=500_000,
        ),
        run=vpm.RunPlan(end_time=END_TIME, state_limit_action="stop"),
        output=vpm.Output(schedule=vpm.EverySteps(sample_steps)),
        samplers=vpm.Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(integral_steps)),
                *(
                    vpm.SurfaceSampler(
                        point=[0.0, 0.0, height],
                        normal=[0.0, 0.0, 1.0],
                        bounds=[-0.4, 0.4, -0.4, 0.4],
                        spacing=0.0075,
                        file_name=plane_name,
                        schedule=vpm.EverySteps(sample_steps),
                    )
                    for plane_name, height in WAKE_PLANES
                ),
            ),
        ),
        backup=vpm.Backup(schedule=backup_schedule),
    )
    vpm.VPMSolver(case).run()


if __name__ == "__main__":
    run()
