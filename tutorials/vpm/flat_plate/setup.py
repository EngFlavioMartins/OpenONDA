#!/usr/bin/env python3
"""Flat-plate angle-of-attack sweep in a moving-body or wind frame (VLM--VPM).

A rectangular flat plate has chord 1 and span 10. Two families of cases are run:

  * ``moving``: the plate travels through still air, with a smooth
    ramp that avoids the impulsive-start transient.
  * ``static``: the plate is fixed and the wind hits it at the angle of attack
    (wind frame).

Each case is named ``exp_<mode>_aoa<NN>``; the sampled forces are used by
``allplot.sh`` to build the lift/drag polar and to compare moving and static
plates at matching angles. The bound/wake strength figure uses the native
force sampler, and velocity profiles use its sampled bound-point velocities.

The induction backend and stretching formulation are independent. Set
`stretching_scheme` to "direct", "mixed", or "transposed" in the case below.

Usage:
    python setup.py --mode moving --angle 8
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.tutorial_runner import case_package
from openonda.vpm import Backup, Samplers

__package__ = case_package(Path(__file__).parent)
from .assets.generate_surface import create_flat_plate, save_surface


# Plate and flow
CHORD = 1.0  # [m]
SPAN = 10.0  # [m]
CHORDWISE_PANELS = 8
SPANWISE_PANELS = 14
FREESTREAM_SPEED = 10.0  # [m/s]
DENSITY = 1.0  # [kg/m^3]
KINEMATIC_VISCOSITY = 1.0e-2  # [m^2/s]

# Time and wake resolution
TIME_STEP_SIZE = 0.0125  # [s]
RAMP_LENGTH = 0.6
FINAL_TRAVEL = 24.0
SMAGORINSKY_COEFFICIENT = 0.30
PARTICLE_CORE_FACTOR = 2.5
SAMPLE_INTERVAL_TIME = 0.0625  # write a snapshot every this many seconds
BACKUP_INTERVAL_TIME = 0.5  # checkpoints every five chord lengths


TUTORIAL_DIR = Path(__file__).resolve().parent


def run(mode: str, angle_of_attack: float) -> None:
    """Run one moving plate or the equivalent wind-frame case."""
    integer_angle = round(angle_of_attack)
    sign = "n" if integer_angle < 0 else ""
    name = f"exp_{mode}_aoa{sign}{abs(integer_angle):02d}"

    if mode == "moving":
        ramp_time = 2.0 * RAMP_LENGTH * CHORD / FREESTREAM_SPEED
        cruise_time = (FINAL_TRAVEL - RAMP_LENGTH) * CHORD / FREESTREAM_SPEED
        n_steps = round((ramp_time + cruise_time) / TIME_STEP_SIZE)
        motion = vpm.SmoothRampVLM(
            final_velocity=[-FREESTREAM_SPEED, 0.0, 0.0],
            acceleration_time=ramp_time,
        )
        freestream_velocity = [0.0, 0.0, 0.0]
        reference_velocity = (FREESTREAM_SPEED, 0.0, 0.0)
        geometry_angle = angle_of_attack
    else:
        n_steps = round(FINAL_TRAVEL * CHORD / FREESTREAM_SPEED / TIME_STEP_SIZE)
        angle = math.radians(angle_of_attack)
        reference_velocity = (
            FREESTREAM_SPEED * math.cos(angle),
            0.0,
            FREESTREAM_SPEED * math.sin(angle),
        )
        freestream_velocity = list(reference_velocity)
        motion = vpm.StaticVLM()
        geometry_angle = 0.0

    surface_dir = TUTORIAL_DIR / "assets" / "surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    surface_file = surface_dir / f"{name}.json"
    save_surface(
        create_flat_plate(
            chord=CHORD,
            span=SPAN,
            angle_of_attack_degrees=geometry_angle,
            n_chordwise_panels=CHORDWISE_PANELS,
            n_spanwise_panels=SPANWISE_PANELS,
        ),
        str(surface_file),
    )

    sample_steps = round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE)
    vlm_setup = vpm.VLMSetup(
        surfaces=(vpm.VLMSurfaceSetup(str(surface_file), kinematics=motion),),
        mesh=vpm.VLMMeshSetup.geometric(ratio=4.0, region="end"),
        density=DENSITY,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        freestream_velocity=reference_velocity,
        force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
        sigma_factor=PARTICLE_CORE_FACTOR,
        sample_surface_forces=True,
        logging_interval_steps=1,
    )
    case = vpm.VPMCase(
        name=name,
        numerics=vpm.Numerics(
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT
            ),
            time_step_size=TIME_STEP_SIZE,
            compute_device="CPU",
            integrator=vpm.SSPRK3(),
            vlm=vlm_setup,
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
            ),
            freestream_velocity=freestream_velocity,
            induction=vpm.DirectInduction(stretching_scheme="transposed"),
            write_precision="f32",
            max_n_particles=120_000,
        ),
        backup=Backup(
            interval_steps=round(BACKUP_INTERVAL_TIME / TIME_STEP_SIZE),
            directory=f"solution/{name}",
            log_directory=f"solution/{name}",
        ),
        samplers=Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(sample_steps)),
                vpm.VLMSampler(schedule=vpm.EverySteps(sample_steps)),
            ),
            directory=name,
        ),
        run=vpm.RunPlan(steps=n_steps),
        directory=TUTORIAL_DIR,
    )
    vpm.VPMSolver(case).run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("moving", "static"),
        required=True,
        help="plate motion: 'moving' travels through still air, 'static' is fixed",
    )
    parser.add_argument("--angle", type=float, required=True, help="angle of attack [deg]")
    args = parser.parse_args()

    run(args.mode, args.angle)


if __name__ == "__main__":
    main()
