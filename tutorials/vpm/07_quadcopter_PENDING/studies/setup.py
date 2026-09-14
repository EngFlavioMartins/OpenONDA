#!/usr/bin/env python3
"""Isolated-rotor resolution, relaxation and restart comparisons.

Run one named case with ``python setup.py --case coarse``. Each case writes
native checkpoints to solution/<case> and native samples to samples/<case>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.tutorial_runner import case_package
from openonda.vpm import Backup, Samplers

__package__ = case_package(Path(__file__).resolve().parents[1])
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

# Degrees/step, revolutions, chord panels, span panels, relaxation, capacity.
CASES = {
    "coarse": (3.75, 6, 4, 12, 0.0, 40_000),
    "time_refined": (1.875, 3, 4, 12, 0.0, 40_000),
    "mesh_refined": (3.75, 3, 8, 24, 0.0, 40_000),
    "relaxed": (3.75, 6, 4, 12, 0.3, 80_000),
    "relaxed_time_refined": (1.875, 3, 4, 12, 0.3, 40_000),
    "continue_8": (3.75, 2, 4, 12, 0.0, 40_000),
    "continue_12": (3.75, 4, 4, 12, 0.0, 60_000),
    "relaxed_moments": (3.75, 12, 4, 12, 0.3, 80_000),
}
RESTARTS = {"continue_8": ("coarse", 576), "continue_12": ("continue_8", 768)}
TUTORIAL_DIR = Path(__file__).resolve().parent


def run(name: str) -> None:
    degrees, revolutions, n_chord, n_span, relaxation, capacity = CASES[name]
    time_step = np.deg2rad(degrees) / ANGULAR_VELOCITY
    steps_per_revolution = round(360.0 / degrees)
    surface_dir = TUTORIAL_DIR / "assets" / name
    surface_dir.mkdir(parents=True, exist_ok=True)
    counterclockwise_file = surface_dir / "blade_ccw.json"
    clockwise_file = surface_dir / "blade_cw.json"
    blade_parameters = {
        "R_hub": HUB_RADIUS,
        "R_tip": TIP_RADIUS,
        "chord_root": 0.025,
        "chord_tip": 0.015,
        "pitch_root_deg": 12.0,
        "pitch_tip_deg": 6.0,
        "n_chord": n_chord,
        "n_span": n_span,
    }
    save_blade(
        create_rotor_blade(**blade_parameters, clockwise=False),
        str(counterclockwise_file),
    )
    save_blade(
        create_rotor_blade(**blade_parameters, clockwise=True),
        str(clockwise_file),
    )

    rotors = (("rotor_0", [ARM_LENGTH, ARM_LENGTH, 0.0], 1.0),)
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
        wake_core_overlap=2.5,
        sample_surface_forces=True,
        # Attached VLM loading is recorded on every accepted VPM step.
        logging_interval_steps=1,
    )

    case = vpm.VPMCase(
        name=CASE_NAME,
        numerics=vpm.Numerics(
            time_step_size=time_step,
            compute_device="CPU",
            vlm=vlm_setup,
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
            ),
            induction=vpm.FMMInduction(stretching_scheme="transposed"),
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(smagorinsky_coefficient=0.17),
            particle_kernel="GAUSSIAN",
            freestream_velocity=[0.0, 0.0, -CLIMB_SPEED],
            stabilization=vpm.StabilizationConfig(
                pedrizzetti_relaxation_factor=relaxation,
                pedrizzetti_relaxation_preserve_moments=(name == "relaxed_moments"),
                remove_particles_by_bounds=[-1.5, 1.5, -1.5, 1.5, -3.0, 1.0],
            ),
            max_n_particles=capacity,
            write_precision="f32",
        ),
        backup=Backup(
            interval_steps=2 * steps_per_revolution,
            directory=f"solution/{name}",
            log_directory=f"solution/{name}",
        ),
        samplers=Samplers(
            samples=(vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(12)),),
            directory=name,
        ),
        run=vpm.RunPlan(steps=revolutions * steps_per_revolution),
        directory=TUTORIAL_DIR,
    )
    solver = vpm.VPMSolver(case)
    if name in RESTARTS:
        previous, step = RESTARTS[name]
        solver.load_backup(TUTORIAL_DIR / "solution" / previous / f"vpm_{step:06d}.h5")
    solver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="coarse")
    run(parser.parse_args().case)
