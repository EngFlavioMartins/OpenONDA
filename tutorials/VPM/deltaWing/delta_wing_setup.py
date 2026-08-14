#!/usr/bin/env python3
"""Run the two-wing wake-crossing tutorial.

Usage: ``python delta_wing_setup.py SAMPLE_PERIOD BACKUP_PERIOD``
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from assets.generate_surface import create_delta_wing, save_surface
from openonda.vpm import (
    ManeuverVLM,
    Solver,
    StabilizationConfig,
    SurfaceSampler,
    TurbulenceConfig,
    VelocityConfig,
    VLMMeshSetup,
    VLMSurfaceSetup,
    VLMSetup,
    VPMSetup,
)
from source.solvers.VPM.io.sampling import resolve_samples_dir


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
TIME_STEP = 0.004
NUMBER_OF_STEPS = 2200


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


def run(sample_period: float, backup_period: float) -> None:
    sample_steps = cadence_steps(sample_period)
    backup_steps = cadence_steps(backup_period)
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
    vlm_setup = VLMSetup(
        surfaces=tuple(
            VLMSurfaceSetup(
                str(surface_file),
                name=name,
                kinematics=ManeuverVLM(
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
        mesh=VLMMeshSetup.geometric(ratio=3.0, region="end"),
        viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        sample_surface_forces=True,
        logging_frequency=sample_steps,
    )

    samplers = tuple(
        SurfaceSampler(
            point=[-distance * HALF_SPAN, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-1.5, 1.5, -1.0, 1.0],
            spacing=0.04,
            file_name=f"wake_{distance}span",
        )
        for distance in (1, 5, 10)
    )
    solver = Solver(
        setup=VPMSetup(
            time_step_size=TIME_STEP,
            processing_unit="AUTO",
            turbulence=TurbulenceConfig.les_smagorinsky(cs=0.3),
            vlm=vlm_setup,
            velocity=VelocityConfig.treecode(
                theta=0.35,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            background_velocity=[-FREESTREAM_VELOCITY, 0, 0],
            stabilization=StabilizationConfig(
                remove_particles_by_bounds=[
                    -8.0,
                    WING_SEPARATION + 1.0,
                    -2.0,
                    2.0,
                    -1.5,
                    1.5,
                ]
            ),
            logging_frequency=sample_steps,
            backup_frequency=backup_steps,
            backup_file_name=CASE_NAME,
            backup_directory=str(SOLUTION_DIR),
            sample_subdirectory=CASE_NAME,
            samplers=samplers,
        )
    )

    samples_dir = resolve_samples_dir(SOLUTION_DIR, CASE_NAME)
    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "A": HEAVE_AMPLITUDE,
        "omega": ANGULAR_FREQUENCY,
        "dt": TIME_STEP,
        "num_steps": NUMBER_OF_STEPS,
        "separation": WING_SEPARATION,
        "half_span": HALF_SPAN,
        "wings": {"front_wing": 0.0, "rear_wing": np.pi},
    }
    (samples_dir / "motion_params.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    for _ in range(NUMBER_OF_STEPS):
        solver.update_state()


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    sample_period, backup_period = map(float, arguments)
    run(sample_period, backup_period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
