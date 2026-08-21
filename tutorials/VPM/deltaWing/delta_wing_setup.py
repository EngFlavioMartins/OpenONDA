#!/usr/bin/env python3
"""Two delta wings crossing wakes (VLM--VPM, LES).

A leading wing and a following wing both heave and pitch through the flow,
shedding vortex wakes that the trailing wing crosses. The sampled forces and
circulation histories feed the ``allplot.sh`` figures.

Usage:
    python delta_wing_setup.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from assets.generate_surface import create_delta_wing, save_surface
import openonda.vpm as vpm


TUTORIAL_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"
CASE_NAME = "delta_wing"

# Wing and flow
FREESTREAM_VELOCITY = 5.0
KINEMATIC_VISCOSITY = 1.0e-3
ROOT_CHORD = 0.5
TIP_CHORD = 0.1
HALF_SPAN = 0.5
ANGLE_OF_ATTACK = 15.0
AIR_DENSITY = 1.225
WING_SEPARATION = 5.0 * HALF_SPAN

# Prescribed motion
HEAVE_AMPLITUDE = 0.2
HEAVE_FREQUENCY = 1.0
PITCH_PIVOT = ROOT_CHORD / 3.0
ANGULAR_FREQUENCY = 2.0 * np.pi * HEAVE_FREQUENCY

# Resolution
END_TIME = 8.8
TIME_STEP = 0.0025
NUMBER_OF_STEPS = round(END_TIME / TIME_STEP)
SAMPLE_INTERVAL_TIME = 0.08  # write a snapshot every this many seconds
CHECKPOINT_INTERVAL_TIME = 0.04  # 25 animation frames per heave cycle


def cadence_steps(period: float) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / TIME_STEP))


def heave_velocity(phase: float):
    def velocity(time: float) -> np.ndarray:
        vertical = HEAVE_AMPLITUDE * ANGULAR_FREQUENCY * np.sin(ANGULAR_FREQUENCY * time + phase)
        return np.array([0.0, 0.0, vertical])

    return velocity


def pitch_velocity(phase: float):
    def angular_velocity(time: float) -> np.ndarray:
        argument = ANGULAR_FREQUENCY * time + phase
        vertical = HEAVE_AMPLITUDE * ANGULAR_FREQUENCY * np.sin(argument)
        acceleration = HEAVE_AMPLITUDE * ANGULAR_FREQUENCY**2 * np.cos(argument)
        pitch_rate = (acceleration / FREESTREAM_VELOCITY) / (
            1.0 + (vertical / FREESTREAM_VELOCITY) ** 2
        )
        return np.array([0.0, -pitch_rate, 0.0])

    return angular_velocity


def run() -> None:
    sample_steps = cadence_steps(SAMPLE_INTERVAL_TIME)
    checkpoint_interval_steps = cadence_steps(CHECKPOINT_INTERVAL_TIME)
    surface_file = TUTORIAL_DIR / "delta_wing_surface.json"
    save_surface(
        create_delta_wing(
            root_chord=ROOT_CHORD,
            tip_chord=TIP_CHORD,
            half_span=HALF_SPAN,
            alpha=ANGLE_OF_ATTACK,
            n_chord=8,
            n_span=18,
        ),
        str(surface_file),
    )

    wings = (
        ("front_wing", WING_SEPARATION, 0.0),
        ("rear_wing", 0.0, np.pi),
    )
    vlm_setup = vpm.VLMSetup(
        surfaces=tuple(
            vpm.VLMSurfaceSetup(
                str(surface_file),
                name=name,
                kinematics=vpm.ManeuverVLM(
                    velocity_fn=heave_velocity(phase),
                    angular_velocity_fn=pitch_velocity(phase),
                    rotation_center=[x_position + PITCH_PIVOT, 0.0, 0.0],
                ),
                translation=(x_position, 0.0, 0.0),
                rotation_deg=(0.0, 0.0, 180.0),
                rotation_center=(x_position + PITCH_PIVOT, 0.0, 0.0),
            )
            for name, x_position, phase in wings
        ),
        mesh=vpm.VLMMeshSetup.geometric(ratio=3.0, region="end"),
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        sample_surface_forces=True,
        logging_interval_steps=sample_steps,
    )

    samplers = tuple(
        vpm.SurfaceSampler(
            point=[-distance * HALF_SPAN, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-1.5, 1.5, -1.0, 1.0],
            spacing=0.04,
            file_name=f"wake_{distance}span",
        )
        for distance in (1, 5, 10)
    )
    solver = vpm.VPMSolver(
        setup=vpm.VPMSetup(
            time_step_size=TIME_STEP,
            compute_device="AUTO",
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(c_s=0.3),
            vlm=vlm_setup,
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
            ),
            velocity=vpm.VelocityConfig.treecode(
                theta=0.35,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            freestream_velocity=[-FREESTREAM_VELOCITY, 0, 0],
            stabilization=vpm.StabilizationConfig(
                remove_particles_by_bounds=[
                    -8.0,
                    WING_SEPARATION + 1.0,
                    -2.0,
                    2.0,
                    -1.5,
                    1.5,
                ]
            ),
            logging_interval_steps=sample_steps,
            checkpoint_interval_steps=checkpoint_interval_steps,
            checkpoint_name=CASE_NAME,
            checkpoint_directory=str(SOLUTION_DIR),
            sample_subdirectory=CASE_NAME,
            samplers=samplers,
        ),
        case_dir=TUTORIAL_DIR,
    )

    samples_dir = TUTORIAL_DIR / "samples" / CASE_NAME
    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "A": HEAVE_AMPLITUDE,
        "omega": ANGULAR_FREQUENCY,
        "time_step_size": TIME_STEP,
        "n_steps": NUMBER_OF_STEPS,
        "separation": WING_SEPARATION,
        "half_span": HALF_SPAN,
        "wings": {"front_wing": 0.0, "rear_wing": np.pi},
    }
    (samples_dir / "motion_params.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    for _ in range(NUMBER_OF_STEPS):
        solver.advance()


def main() -> int:
    print("\n===== SIMULATION =====")
    print("---- Two heaving/pitching delta wings crossing wakes ----")
    run()
    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
