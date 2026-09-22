#!/usr/bin/env python3
"""Two delta wings crossing wakes (VLM--VPM).

A leading wing and a following wing both heave and pitch through the flow,
shedding vortex wakes that the trailing wing crosses. The sampled forces and
circulation histories feed the ``allplot.sh`` figures.

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
from .assets.generate_surface import create_delta_wing, save_surface

START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_NAME = "delta_wing"

# Wing and flow
FREESTREAM_VELOCITY = 5.0
KINEMATIC_VISCOSITY = 1.0e-3  # [m^2/s]
ROOT_CHORD = 0.5
TIP_CHORD = 0.1
HALF_SPAN = 0.5
ANGLE_OF_ATTACK = 15.0
AIR_DENSITY = 1.225  # [kg/m^3]
WING_SEPARATION = 5.0 * HALF_SPAN

# Prescribed motion
HEAVE_AMPLITUDE = 0.2
HEAVE_FREQUENCY = 1.0
PITCH_PIVOT = ROOT_CHORD / 3.0
ANGULAR_FREQUENCY = 2.0 * np.pi * HEAVE_FREQUENCY

# Resolution
END_TIME = 20.0  # [s]
TIME_STEP_SIZE = 0.0025  # [s]
N_STEPS = round(END_TIME / TIME_STEP_SIZE)
SAMPLE_INTERVAL_TIME = 0.025  # 40 coupled VPM/VLM backups per heave cycle; >=30 fps


TUTORIAL_DIR = Path(__file__).resolve().parent


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


def _write_surface() -> Path:
    surface_file = TUTORIAL_DIR / "assets" / "delta_wing_surface.json"
    save_surface(
        create_delta_wing(
            root_chord=ROOT_CHORD,
            tip_chord=TIP_CHORD,
            half_span=HALF_SPAN,
            angle_of_attack_degrees=ANGLE_OF_ATTACK,
            n_chordwise_panels=8,
            n_spanwise_panels=18,
        ),
        str(surface_file),
    )
    return surface_file


def build_case() -> vpm.VPMCase:
    """Build the owner-clocked Delta case."""
    sample_steps = round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE)
    surface_file = _write_surface()

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
                    velocity_function=heave_velocity(phase),
                    angular_velocity_function=pitch_velocity(phase),
                    rotation_centre=[x_position + PITCH_PIVOT, 0.0, 0.0],
                ),
                translation=(x_position, 0.0, 0.0),
                rotation_degrees=(0.0, 0.0, 180.0),
                rotation_centre=(PITCH_PIVOT, 0.0, 0.0),
                group_id=surface_index,
            )
            for surface_index, (name, x_position, phase) in enumerate(wings)
        ),
        mesh=vpm.VLMMeshSetup.geometric(ratio=3.0, region="end"),
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        wake_core_overlap=2.5,
        sample_surface_forces=True,
        # Attached VLM loading is an accepted-step diagnostic owned by VPM.
        logging_interval_steps=1,
    )

    samplers = tuple(
        vpm.SurfaceSampler(
            point=[-distance * HALF_SPAN, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-0.9, 0.9, -1.5, 0.9],  # include the descending far wake
            spacing=0.04,
            file_name=f"wake_{distance}span",
            include_derivatives=False,
            schedule=vpm.EverySteps(sample_steps),
        )
        for distance in (1, 5, 10)
    )
    case = vpm.VPMCase(
        name=CASE_NAME,
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device="CPU",
            turbulence=vpm.TurbulenceConfig.dns(),
            vlm=vlm_setup,
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
            ),
            induction=vpm.FMMInduction(stretching_scheme="transposed"),
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
            max_n_particles=250_000,
            write_precision="f32",
        ),
        backup=Backup(
            interval_steps=sample_steps,
            directory="solution",
            log_directory="solution",
        ),
        samplers=Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(sample_steps)),
                *samplers,
            ),
            directory=CASE_NAME,
        ),
        run=vpm.RunPlan(steps=N_STEPS),
        directory=TUTORIAL_DIR,
    )
    return case


def main() -> None:
    vpm.VPMSolver(build_case()).run(start_from=START_FROM)


if __name__ == "__main__":
    main()
