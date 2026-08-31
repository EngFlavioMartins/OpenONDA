#!/usr/bin/env python3
"""Flat-plate angle-of-attack sweep in a moving-body or wind frame (VLM--VPM).

A rectangular flat plate of chord 1 and span 10 is accelerated impulsively to
Freestream speed, then allowed to travel. Two families of cases are run:

  * ``moving``: the plate travels through still air (body frame), a smooth
    ramp that avoids the impulsive-start transient.
  * ``static``: the plate is fixed and the wind hits it at the angle of attack
    (wind frame).

Each case is named ``exp_<mode>_aoa<NN>``; the sampled forces are used by
``allplot.sh`` to build the lift/drag polar and to compare moving and static
plates at matching angles. The ``static`` case at 8 degrees additionally
writes cross-flow wake planes for the Kelvin theorem figure.

Usage:
    python setup_plate.py --mode moving --angle 8
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from assets.generate_surface import create_flat_plate, save_surface
import openonda.vpm as vpm

TUTORIAL_DIR = Path(__file__).resolve().parent

# ---- Plate and flow ------------------------------------------------------
CHORD = 1.0
SPAN = 10.0
CHORDWISE_PANELS = 8
SPANWISE_PANELS = 14
FREESTREAM_SPEED = 10.0
DENSITY = 1.0
KINEMATIC_VISCOSITY = 1.0e-2

# ---- Time and wake resolution ---------------------------------------------
TIME_STEP_SIZE = 0.0125
RAMP_LENGTH = 0.6
FINAL_TRAVEL = 24.0
SMAGORINSKY_COEFFICIENT = 0.30
PARTICLE_CORE_FACTOR = 2.5
SAMPLE_INTERVAL_TIME = 0.0625  # write a snapshot every this many seconds
BACKUP_INTERVAL_TIME = 0.05  # one animation frame per half-chord of travel


def cadence_steps(period: float) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / TIME_STEP_SIZE))


def time_parameters(kinematics: str) -> tuple[int, float]:
    """Return the number of steps and ramp duration for one case."""
    if kinematics == "ramp":
        ramp_time = 2.0 * RAMP_LENGTH * CHORD / FREESTREAM_SPEED
        cruise_time = (FINAL_TRAVEL - RAMP_LENGTH) * CHORD / FREESTREAM_SPEED
        return round((ramp_time + cruise_time) / TIME_STEP_SIZE), ramp_time

    final_time = FINAL_TRAVEL * CHORD / FREESTREAM_SPEED
    return round(final_time / TIME_STEP_SIZE), 0.0


def plate_kinematics(
    kinematics: str,
    frame: str,
    angle_of_attack: float,
    ramp_time: float,
):
    """Build the motion and background flow for a body- or wind-frame case."""
    if kinematics == "ramp":
        return (
            vpm.SmoothRampVLM(
                final_velocity=[-FREESTREAM_SPEED, 0.0, 0.0],
                acceleration_time=ramp_time,
            ),
            [0.0, 0.0, 0.0],
        )
    if frame == "body":
        return (
            vpm.TranslatingVLM(velocity=np.array([-FREESTREAM_SPEED, 0.0, 0.0])),
            [0.0, 0.0, 0.0],
        )

    angle = math.radians(angle_of_attack)
    return (
        vpm.StaticVLM(),
        [FREESTREAM_SPEED * math.cos(angle), 0.0, FREESTREAM_SPEED * math.sin(angle)],
    )


def distance_travelled(times: np.ndarray, kinematics: str, ramp_time: float) -> np.ndarray:
    """Return plate travel in chord lengths at sampled physical times."""
    if kinematics != "ramp":
        return times * FREESTREAM_SPEED / CHORD

    ramp_time = max(ramp_time, 1.0e-12)
    ramp_distance = (
        0.5 * FREESTREAM_SPEED * (times - ramp_time / math.pi * np.sin(math.pi * times / ramp_time))
    )
    cruise_distance = 0.5 * FREESTREAM_SPEED * ramp_time + FREESTREAM_SPEED * (times - ramp_time)
    return np.where(times < ramp_time, ramp_distance, cruise_distance) / CHORD


def crossflow_samplers(name: str) -> tuple[vpm.SurfaceSampler, ...]:
    """Return the three final wake planes used by the Kelvin figure."""
    return tuple(
        vpm.SurfaceSampler(
            point=[position, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-6.5, 6.5, -0.5, 6.0],
            spacing=max(FREESTREAM_SPEED * TIME_STEP_SIZE, 0.05),
            file_name=f"{name}_crossflow_x{position:g}",
            include_derivatives=False,
            schedule=vpm.SamplingSchedule.final(),
        )
        for position in (5.0, 15.0, 25.0)
    )


def run_case(
    name: str, kinematics: str, frame: str, angle_of_attack: float, sample_planes: bool
) -> None:
    n_steps, ramp_time = time_parameters(kinematics)
    motion, freestream_velocity = plate_kinematics(
        kinematics,
        frame,
        angle_of_attack,
        ramp_time,
    )
    geometry_angle = 0.0 if frame == "wind" else angle_of_attack

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

    angle = math.radians(angle_of_attack)
    reference_velocity = (
        (
            FREESTREAM_SPEED * math.cos(angle),
            0.0,
            FREESTREAM_SPEED * math.sin(angle),
        )
        if frame == "wind"
        else (FREESTREAM_SPEED, 0.0, 0.0)
    )
    sample_steps = cadence_steps(SAMPLE_INTERVAL_TIME)
    vlm_setup = vpm.VLMSetup(
        surfaces=(vpm.VLMSurfaceSetup(str(surface_file), kinematics=motion),),
        mesh=vpm.VLMMeshSetup.geometric(ratio=4.0, region="end"),
        density=DENSITY,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        freestream_velocity=reference_velocity,
        force=vpm.ForceConfig.kutta_joukowski(),
        sigma_factor=PARTICLE_CORE_FACTOR,
        sample_surface_forces=True,
        logging_interval_steps=sample_steps,
    )
    final_samples = crossflow_samplers(name) if sample_planes else ()
    samples_dir = TUTORIAL_DIR / "samples" / name
    samples_dir.mkdir(parents=True, exist_ok=True)

    for stale_name in (
        "vlm_forces.csv",
        "vlm_spanwise_flat_plate.csv",
        "vlm_chordwise_flat_plate.csv",
    ):
        (samples_dir / stale_name).unlink(missing_ok=True)

    solver = vpm.VPMSolver(
        setup=vpm.VPMSetup.les_simulation(
            smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT,
            time_step_size=TIME_STEP_SIZE,
            compute_device="AUTO",
            advection=vpm.AdvectionConfig(scheme="RK3"),
            vlm=vlm_setup,
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
            ),
            freestream_velocity=freestream_velocity,
            velocity=vpm.VelocityConfig.treecode(
                theta=0.35,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            backup=vpm.Backup(
                interval_steps=cadence_steps(BACKUP_INTERVAL_TIME),
                directory="solution",
                log_directory="solution",
            ),
            samplers=vpm.Samplers(
                vpm.FlowIntegralsSampler(
                    schedule=vpm.SamplingSchedule(every_n_steps=sample_steps)
                ),
                *final_samples,
                directory=name,
            ),
            write_precision="f32",
            max_n_particles=120_000,
        ),
        case_dir=TUTORIAL_DIR,
    )

    (samples_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "case": name,
                "frame": frame,
                "kinematics": kinematics,
                "angle_of_attack_degrees": angle_of_attack,
                "time_step_size": TIME_STEP_SIZE,
                "n_steps": n_steps,
                "smagorinsky_coefficient": SMAGORINSKY_COEFFICIENT,
                "compute_device": solver.compute_device,
                "precision": solver.precision,
                "write_precision": solver.write_precision,
                "panel_resolution": {
                    "chordwise": CHORDWISE_PANELS,
                    "spanwise": SPANWISE_PANELS,
                },
                "treecode_theta": 0.35,
                "sample_interval_time": SAMPLE_INTERVAL_TIME,
                "backup_interval_time": BACKUP_INTERVAL_TIME,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for _ in range(n_steps):
        solver.advance()
    if final_samples:
        solver.execute_final_samples()

    forces_path = samples_dir / "vlm_forces.csv"
    if forces_path.exists():
        forces = pd.read_csv(forces_path)
        forces["nondimensional_distance_travelled"] = distance_travelled(
            forces["time"].to_numpy(),
            kinematics,
            ramp_time,
        )
        forces.to_csv(samples_dir / f"{name}.csv", index=False)
        forces_path.unlink()

    spanwise_path = samples_dir / "vlm_spanwise_flat_plate.csv"
    if spanwise_path.exists():
        spanwise = pd.read_csv(spanwise_path)
        final_step = spanwise["step"].max()
        spanwise[spanwise["step"] == final_step].to_csv(
            samples_dir / f"{name}_spanwise.csv",
            index=False,
        )


def run(mode: str, angle_of_attack: float) -> None:
    """Run one moving-body or static-wind case."""
    integer_angle = round(angle_of_attack)
    if not math.isclose(angle_of_attack, integer_angle):
        raise ValueError("The flat-plate sweep uses whole-degree angles of attack")
    if mode not in {"moving", "static"}:
        raise ValueError(f"Unknown flat-plate mode: {mode}")

    sign = "n" if integer_angle < 0 else ""
    name = f"exp_{mode}_aoa{sign}{abs(integer_angle):02d}"
    moving = mode == "moving"
    run_case(
        name=name,
        kinematics="ramp" if moving else "static",
        frame="body" if moving else "wind",
        angle_of_attack=angle_of_attack,
        sample_planes=not moving and integer_angle == 8,
    )


def main() -> int:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
