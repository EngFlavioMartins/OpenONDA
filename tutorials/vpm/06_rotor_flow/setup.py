#!/usr/bin/env python3
"""Three-bladed rotor with a VLM blade model and a VPM wake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

CASE_NAME = "rotor"
START_FROM = "latest"

FREESTREAM_SPEED = 7.0  # [m/s]
TIP_SPEED_RATIO = 7.0
ROTOR_RADIUS = 6.0  # [m]
HUB_RADIUS = 1.0  # [m]
KINEMATIC_VISCOSITY = 1.5e-5  # [m^2/s]
AIR_DENSITY = 1.225  # [kg/m^3]
MAX_N_PARTICLES = 600_000
ANGULAR_VELOCITY = TIP_SPEED_RATIO * FREESTREAM_SPEED / ROTOR_RADIUS  # [rad/s]

TIME_STEP_SIZE = 0.006  # [s]
END_TIME = 10  # [s]
N_STEPS = round(END_TIME / TIME_STEP_SIZE)
FIELD_SAMPLE_INTERVAL_TIME = 0.06  # [s]; compact wake-plane fields
BACKUP_INTERVAL_TIME = 0.024  # [s]; dense full-state animation/restart frames

TUTORIAL_DIR = Path(__file__).resolve().parent

def build_case(
    *,
    time_step_size: float = TIME_STEP_SIZE,
    steps: int = N_STEPS,
    solution_directory: Path = Path("solution"),
    sample_directory: Path = Path("samples") / CASE_NAME,
    run_plan: vpm.RunPlan | None = None,
) -> vpm.VPMCase:
    """Build the rotor case and its output schedules."""
    solution_directory = Path(solution_directory)
    sample_directory = Path(sample_directory)
    relative_sample_directory = sample_directory.relative_to("samples")

    blade_file = TUTORIAL_DIR / "assets/blade.json"
    segments = json.loads(blade_file.read_text())["wings"][0]["segments"]
    radial_panels = sum(segment["n_spanwise_panels"] for segment in segments)
    wake_spacing = (ROTOR_RADIUS - HUB_RADIUS) / radial_panels
    rotation_kinematics = vpm.RotatingVLM(
        angular_speed=-ANGULAR_VELOCITY,
        axis=[1.0, 0.0, 0.0],
        acceleration_time=2.0 * np.pi / ANGULAR_VELOCITY,
    )

    vlm_setup = vpm.VLMSetup(
        surfaces=tuple(
            vpm.VLMSurfaceSetup(
                str(blade_file),
                name=f"blade_{blade_index}",
                kinematics=rotation_kinematics,
                rotation_degrees=(azimuth, 0.0, 0.0),
                group_id=blade_index,
            )
            for blade_index, azimuth in enumerate((0.0, 120.0, 240.0))
        ),
        mesh=vpm.VLMMeshSetup.geometric(),
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        wake_core_overlap=2.5,
        force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
    )

    plane_half_width = 0.65 * (2.0 * ROTOR_RADIUS)
    sample_spacing = ROTOR_RADIUS / 36
    plane_schedule = vpm.EveryTime(FIELD_SAMPLE_INTERVAL_TIME)
    plane_samplers = [
        vpm.SurfaceSampler(
            point=[2.0 * ROTOR_RADIUS * distance, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-plane_half_width, plane_half_width, -plane_half_width, plane_half_width],
            spacing=sample_spacing,
            file_name=f"wake_{distance}D",
            include_derivatives=False,
            schedule=plane_schedule,
        )
        for distance in (1, 2)
    ]
    streamwise_samplers = [
        vpm.LineSampler(
            start=[-2.0 * ROTOR_RADIUS, radial_fraction * ROTOR_RADIUS, 0.0],
            end=[6.0 * ROTOR_RADIUS, radial_fraction * ROTOR_RADIUS, 0.0],
            spacing=sample_spacing,
            file_name=f"streamwise_r{label}",
            include_derivatives=False,
            schedule=plane_schedule,
        )
        for label, radial_fraction in (("000", 0.0), ("025", 0.25), ("065", 0.65), ("110", 1.1))
    ]

    if run_plan is None:
        run_plan = vpm.RunPlan(
            steps=steps,
            health_limit_action="STOP",
        )
    return vpm.VPMCase(
        name=CASE_NAME,
        numerics=vpm.Numerics(
            time_step_size=time_step_size,
            vlm=vlm_setup,
            freestream_velocity=[FREESTREAM_SPEED, 0.0, 0.0],
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(),
            stabilization=vpm.StabilizationConfig(
                selective_eddy_viscosity_coefficient=0.5,
                filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                    interval_steps=5,
                ),
            ),
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                particle_spacing=wake_spacing,
            ),
            induction=vpm.TreecodeInduction(
                theta=0.3,
                multipole_order=3,
            ),
            max_n_particles=MAX_N_PARTICLES,
        ),
        backup=Backup(
            interval_steps=max(1, round(BACKUP_INTERVAL_TIME / time_step_size)),
            directory=solution_directory,
            log_directory=solution_directory,
        ),
        samplers=Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EveryTime(FIELD_SAMPLE_INTERVAL_TIME)),
                *plane_samplers,
                *streamwise_samplers,
            ),
            directory=relative_sample_directory,
        ),
        run=run_plan,
        directory=TUTORIAL_DIR,
    )


def run(output_tag: str | None = None) -> None:
    """Run or continue one rotor case in an explicit output namespace."""
    solution_directory = Path("solution") if output_tag is None else Path("solution") / output_tag
    sample_directory = Path("samples") / CASE_NAME
    if output_tag is not None:
        sample_directory /= output_tag
    solver = vpm.VPMSolver(
        build_case(
            solution_directory=solution_directory,
            sample_directory=sample_directory,
        )
    )
    solver.run(start_from=START_FROM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-tag",
        help=(
            "case namespace under solution/ and samples/rotor/ (resumes its latest backup)"
        ),
    )
    run(parser.parse_args().output_tag)
