#!/usr/bin/env python3
"""Simulate the wake of a quadcopter in climb (VLM--VPM).

Four two-bladed rotors counter-rotate on a small quadcopter frame. The vehicle
climbs at constant speed while the rotors shed their wakes into the flow. The
particle count and the integrated vorticity history are sampled for the
diagnostic figures made by ``allplot.sh``.

Usage:
    python quadcopter_setup.py
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from assets.generate_blade import create_rotor_blade, save_blade
import openonda.vpm as vpm

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
TIME_STEP_SIZE = np.deg2rad(DEGREES_PER_STEP) / ANGULAR_VELOCITY
STEPS_PER_REVOLUTION = round(360.0 / DEGREES_PER_STEP)
NUMBER_OF_REVOLUTIONS = 6
N_STEPS = NUMBER_OF_REVOLUTIONS * STEPS_PER_REVOLUTION
SAMPLE_INTERVAL_TIME = 0.0375  # write a snapshot every this many seconds
CHECKPOINT_INTERVAL_TIME = 0.0125  # 24 animation frames per rotor revolution

WAKE_PLANES = (("sampled_zplane", -0.35), ("sampled_zplane_deep", -0.70))


def cadence_steps(period: float) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / TIME_STEP_SIZE))


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
        logging_interval_steps=cadence_steps(SAMPLE_INTERVAL_TIME),
    )

    sample_steps = cadence_steps(SAMPLE_INTERVAL_TIME)
    solver = vpm.VPMSolver(
        setup=vpm.VPMSetup(
            time_step_size=TIME_STEP_SIZE,
            compute_device="AUTO",
            vlm=vlm_setup,
            stretching=vpm.StretchingConfig.disabled(),
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
            ),
            velocity=vpm.VelocityConfig.treecode(
                theta=0.35,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            turbulence=vpm.TurbulenceConfig.dns(),
            particle_kernel="WINCKELMANS",
            freestream_velocity=[0.0, 0.0, -CLIMB_SPEED],
            stabilization=vpm.StabilizationConfig(
                remove_particles_by_bounds=[-1.5, 1.5, -1.5, 1.5, -3.0, 1.0]
            ),
            logging_interval_steps=sample_steps,
            checkpoint_interval_steps=cadence_steps(CHECKPOINT_INTERVAL_TIME),
            checkpoint_name=CASE_NAME,
            checkpoint_directory=str(SOLUTION_DIR),
            sample_subdirectory=CASE_NAME,
            write_precision="f32",
            checkpoint_store_velocity_gradient=False,
            samplers=tuple(
                vpm.SurfaceSampler(
                    point=[0.0, 0.0, height],
                    normal=[0.0, 0.0, 1.0],
                    bounds=[-0.4, 0.4, -0.4, 0.4],
                    spacing=0.0075,
                    file_name=plane_name,
                    include_derivatives=False,
                )
                for plane_name, height in WAKE_PLANES
            ),
        ),
        case_dir=TUTORIAL_DIR,
    )

    samples_dir = TUTORIAL_DIR / "samples" / CASE_NAME
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "case": CASE_NAME,
                "n_steps": N_STEPS,
                "number_of_revolutions": NUMBER_OF_REVOLUTIONS,
                "time_step_size": TIME_STEP_SIZE,
                "compute_device": solver.compute_device,
                "precision": solver.precision,
                "write_precision": solver.write_precision,
                "checkpoint_store_velocity_gradient": solver.checkpoint_store_velocity_gradient,
                "turbulence_model": "DNS",
                "smagorinsky_coefficient": 0.0,
                "treecode_theta": 0.35,
                "number_of_rotors": len(rotors),
                "number_of_blades_per_rotor": NUMBER_OF_BLADES,
                "sample_interval_time": SAMPLE_INTERVAL_TIME,
                "checkpoint_interval_time": CHECKPOINT_INTERVAL_TIME,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = samples_dir / "run_manifest.json"
    try:
        for _ in range(N_STEPS):
            solver.advance()
    except BaseException:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "failed"
        manifest["completed"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status="complete", completed=True, final_time=solver.time)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    print("\n===== SIMULATION =====")
    print("---- Quadcopter climb: 4 rotors, 2 blades each, 6 revolutions ----")
    run()
    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
