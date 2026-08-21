#!/usr/bin/env python3
"""Simulate the wake of a quadcopter in climb (VLM--VPM).

Four two-bladed rotors counter-rotate on a small quadcopter frame. The vehicle
climbs at constant speed while the rotors shed their wakes into the flow. The
particle count and the integrated vorticity history are sampled for the
diagnostic figures made by ``allplot.sh``.

Usage:
    python quad_setup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from assets.generate_blade import create_rotor_blade, save_blade
from openonda.vpm import (
    RotatingVLM,
    VPMSolver,
    StabilizationConfig,
    StretchingConfig,
    SurfaceSampler,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
    VLMSurfaceSetup,
    VLMSetup,
    VPMSetup,
)

TUTORIAL_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"
CASE_NAME = "quadcopter"

# ---- Rotor and flow ------------------------------------------------------
ROTATIONS_PER_MINUTE = 200.0
ANGULAR_VELOCITY = ROTATIONS_PER_MINUTE * 2.0 * np.pi / 60.0
TIP_RADIUS = 0.15
HUB_RADIUS = 0.03
AIR_DENSITY = 1.225
KINEMATIC_VISCOSITY = 1.5e-5
NUMBER_OF_BLADES = 2
CLIMB_SPEED = 0.8
ARM_LENGTH = 0.16

# ---- Time resolution ------------------------------------------------------
DEGREES_PER_STEP = 7.5
TIME_STEP = np.deg2rad(DEGREES_PER_STEP) / ANGULAR_VELOCITY
STEPS_PER_REVOLUTION = round(360.0 / DEGREES_PER_STEP)
NUMBER_OF_REVOLUTIONS = 6
NUMBER_OF_STEPS = NUMBER_OF_REVOLUTIONS * STEPS_PER_REVOLUTION
SAMPLE_INTERVAL_TIME = 0.0375  # write a snapshot every this many seconds
CHECKPOINT_INTERVAL_TIME = 0.0125  # 24 animation frames per rotor revolution


def cadence_steps(period: float) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / TIME_STEP))


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
    vlm_setup = VLMSetup(
        surfaces=tuple(
            VLMSurfaceSetup(
                str(counterclockwise_file if direction > 0 else clockwise_file),
                name=f"{name}_blade_{blade_index}",
                kinematics=RotatingVLM(
                    omega=ANGULAR_VELOCITY * direction,
                    axis=[0.0, 0.0, 1.0],
                    center=position,
                ),
                translation=tuple(position),
                rotation_deg=(0.0, 0.0, 360.0 / NUMBER_OF_BLADES * blade_index),
                group_id=rotor_index + 1,
            )
            for rotor_index, (name, coordinates, direction) in enumerate(rotors)
            for position in (np.array(coordinates),)
            for blade_index in range(NUMBER_OF_BLADES)
        ),
        viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        sigma_factor=2.5,
    )

    sample_steps = cadence_steps(SAMPLE_INTERVAL_TIME)
    solver = VPMSolver(
        setup=VPMSetup(
            time_step_size=TIME_STEP,
            compute_device="AUTO",
            vlm=vlm_setup,
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig.cs(),
            velocity=VelocityConfig.treecode(
                theta=0.35,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            turbulence=TurbulenceConfig.dns(),
            particle_kernel="WINCKELMANS",
            freestream_velocity=[0.0, 0.0, -CLIMB_SPEED],
            stabilization=StabilizationConfig(
                remove_particles_by_bounds=[-1.5, 1.5, -1.5, 1.5, -3.0, 1.0]
            ),
            logging_interval_steps=sample_steps,
            checkpoint_interval_steps=cadence_steps(CHECKPOINT_INTERVAL_TIME),
            checkpoint_name=CASE_NAME,
            checkpoint_directory=str(SOLUTION_DIR),
            sample_subdirectory=CASE_NAME,
            samplers=(
                SurfaceSampler(
                    point=[0.0, 0.0, -1.2],
                    normal=[0.0, 0.0, 1.0],
                    bounds=[-0.4, 0.4, -0.4, 0.4],
                    spacing=0.0075,
                    file_name="sampled_zplane",
                ),
            ),
        ),
        case_dir=TUTORIAL_DIR,
    )

    for _ in range(NUMBER_OF_STEPS):
        solver.advance()


def main() -> int:
    print("\n===== SIMULATION =====")
    print("---- Quadcopter climb: 4 rotors, 2 blades each, 6 revolutions ----")
    run()
    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
