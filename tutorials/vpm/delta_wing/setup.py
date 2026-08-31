#!/usr/bin/env python3
"""Two delta wings crossing wakes (VLM--VPM, LES).

A leading wing and a following wing both heave and pitch through the flow,
shedding vortex wakes that the trailing wing crosses. The sampled forces and
circulation histories feed the ``allplot.sh`` figures.

Usage:
    python setup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from assets.generate_surface import create_delta_wing, save_surface
import openonda.vpm as vpm
from source.solvers.vpm.config.artifacts import Backup, Samplers


TUTORIAL_DIR = Path(__file__).resolve().parent
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
TIME_STEP_SIZE = 0.0025
N_STEPS = round(END_TIME / TIME_STEP_SIZE)
SAMPLE_INTERVAL_TIME = 0.08  # write a snapshot every this many seconds
BACKUP_INTERVAL_TIME = 0.04  # 25 animation frames per heave cycle


def cadence_steps(period: float) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / TIME_STEP_SIZE))


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
    treecode_theta = 0.30
    smagorinsky_coefficient = 0.0
    n_steps = N_STEPS
    sample_steps = cadence_steps(SAMPLE_INTERVAL_TIME)
    backup_steps = cadence_steps(BACKUP_INTERVAL_TIME)
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
                rotation_centre=(x_position + PITCH_PIVOT, 0.0, 0.0),
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
            bounds=[-0.9, 0.9, -0.9, 0.2],
            spacing=0.04,
            file_name=f"wake_{distance}span",
            include_derivatives=False,
            schedule=vpm.EverySteps(sample_steps),
        )
        for distance in (1, 5, 10)
    )
    case = vpm.VPMCase(
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device="AUTO",
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=smagorinsky_coefficient
            ),
            vlm=vlm_setup,
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
            ),
            velocity=vpm.VelocityConfig.treecode(
                theta=treecode_theta,
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
            write_precision="f32",
        ),
        backup=Backup(
            interval_steps=backup_steps,
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
        run=vpm.RunPlan(steps=n_steps),
        directory=TUTORIAL_DIR,
    )
    vpm.VPMSolver(case).run()


def main() -> int:
    print("\n===== SIMULATION =====")
    print("---- Two heaving/pitching delta wings crossing wakes ----")
    run()
    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
