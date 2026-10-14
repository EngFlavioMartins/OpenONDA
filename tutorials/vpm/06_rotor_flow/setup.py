#!/usr/bin/env python3
"""Three-bladed rotor with a VLM blade model and a VPM wake."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from assets.output_files import read_json

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
# Allow the 2D wake to develop before the final five-revolution mean. With the
# matched BEM a ~= 0.219, ramp + 2D/[U*(1-2a)] + five revolutions is ~10.7 s.
# This planning margin does not replace the measured stationarity checks.
END_TIME = 15  # [s]
FIELD_SAMPLE_INTERVAL_TIME = 0.06  # [s]
BACKUP_INTERVAL_TIME = 0.024  # [s]
STREAMWISE_STATIONS = (("025", 0.25), ("065", 0.65))
# 1-a applies at the disk. 1D/2D test downstream development toward the
# 1-2a limit; no finite x/D guarantees a fully developed ideal wake.
WAKE_PLANE_DIAMETERS = (0, 1, 2)

TUTORIAL_DIR = Path(__file__).resolve().parent


def build_case(
    *,
    time_step_size: float = TIME_STEP_SIZE,
    end_time: float = END_TIME,
    variant: str = CASE_NAME,
    run_plan: vpm.RunPlan | None = None,
) -> vpm.VPMCase:
    """Build the rotor case and its output schedules."""

    blade_file = TUTORIAL_DIR / "assets/blade.json"
    segments = read_json(blade_file)["wings"][0]["segments"]
    radial_panels = sum(segment["n_spanwise_panels"] for segment in segments)
    wake_spacing = (ROTOR_RADIUS - HUB_RADIUS) / radial_panels
    rotation_kinematics = vpm.RotatingVLM(
        angular_speed=-ANGULAR_VELOCITY,
        axis=[1.0, 0.0, 0.0],
        acceleration_time=2.0 * np.pi / ANGULAR_VELOCITY,
    )

    vlm_setup = vpm.VLMConfig(
        surfaces=tuple(
            vpm.VLMSurface(
                str(blade_file),
                name=f"blade_{blade_index}",
                kinematics=rotation_kinematics,
                rotation_degrees=(azimuth, 0.0, 0.0),
                group_id=blade_index,
            )
            for blade_index, azimuth in enumerate((0.0, 120.0, 240.0))
        ),
        mesh=vpm.VLMMeshConfig.geometric(),
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
        for distance in WAKE_PLANE_DIAMETERS
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
        for label, radial_fraction in STREAMWISE_STATIONS
    ]

    if run_plan is None:
        run_plan = vpm.RunPlan(end_time=end_time, state_limit_action="stop")
    backup_schedule = vpm.EverySteps(max(1, round(BACKUP_INTERVAL_TIME / time_step_size)))
    return vpm.VPMCase(
        name=CASE_NAME,
        directory=TUTORIAL_DIR,
        variant=variant,
        numerics=vpm.Numerics(
            time_step_size=time_step_size,
            vlm=vlm_setup,
            freestream_velocity=[FREESTREAM_SPEED, 0.0, 0.0],
            turbulence=vpm.TurbulenceConfig.smagorinsky(),
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
            induction=vpm.FMMInduction(),
            max_n_particles=MAX_N_PARTICLES,
        ),
        run=run_plan,
        output=vpm.Output(schedule=backup_schedule),
        samplers=vpm.Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EveryTime(FIELD_SAMPLE_INTERVAL_TIME)),
                *plane_samplers,
                *streamwise_samplers,
            ),
        ),
        backup=vpm.Backup(schedule=backup_schedule),
    )


def run() -> None:
    """Run or continue the rotor case."""
    vpm.VPMSolver(build_case()).run(start_from=START_FROM)


if __name__ == "__main__":
    run()
