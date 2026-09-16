#!/usr/bin/env python3
"""Build and run the ordinary three-bladed rotor VLM--VPM case.

The bounded changed-step restart pilot lives in
``assets/run_restart_pilot.py`` so the ordinary tutorial remains a direct,
fresh-run case builder.

Usage:
    python setup.py --output-tag completion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

CASE_NAME = "rotor"

FREESTREAM_SPEED = 7.0  # [m/s]
TIP_SPEED_RATIO = 7.0
ROTOR_RADIUS = 6.0  # [m]
# Requested wake labels use the authored design diameter (D=2R=12 m).  Native
# blade geometry is read independently by post-processing for actual-radius
# load and BEM normalization.
STATION_REFERENCE_RADIUS = ROTOR_RADIUS
HUB_RADIUS = 1.0  # [m]
KINEMATIC_VISCOSITY = 1.5e-5  # [m^2/s]
AIR_DENSITY = 1.225  # [kg/m^3]
N_RADIAL_STATIONS = 23
MAX_N_PARTICLES = 600_000  # about 169000 emitted over 7.5 s, with capacity margin
# Lifecycle guardrails keep a fresh run restartable on the shared 16 GiB host;
# the hard numerical container capacity remains the separate MAX_N_PARTICLES.
RUN_PARTICLE_SOFT_LIMIT = 570_000
RUN_RSS_LIMIT_BYTES = 12 * 2**30
RUN_AVAILABLE_MEMORY_FLOOR_BYTES = 2 * 2**30
ANGULAR_VELOCITY = TIP_SPEED_RATIO * FREESTREAM_SPEED / ROTOR_RADIUS  # [rad/s]

TIME_STEP_SIZE = 0.006  # [s]
# This is a run horizon, not a validated stability limit. The retained native
# run failed at 7.68 s; stopping earlier does not establish production health.
END_TIME = 7.5  # [s]; selected before the known t=7.68 s health stop
N_STEPS = round(END_TIME / TIME_STEP_SIZE)
DEFAULT_SMAGORINSKY_COEFFICIENT = 0.17
RAMP_ROTATIONS = 1.0
FIELD_SAMPLE_INTERVAL_TIME = 0.06  # [s]; compact wake-plane fields
ROTATION_PERIOD = 2.0 * np.pi / ANGULAR_VELOCITY  # [s]
# Full VPM/VLM numerical backups follow blade motion at an owner cadence of
# 0.024 s (four authored steps, about 32 frames per revolution). Force/loading CSVs use the
# accepted-step owner clock and are not configured by a second VLM sampler.
BACKUP_INTERVAL_TIME = 0.024  # [s]; dense full-state animation/restart frames

TUTORIAL_DIR = Path(__file__).resolve().parent
FIXED_WAKE_SPACING = (ROTOR_RADIUS - HUB_RADIUS) / (N_RADIAL_STATIONS - 1)


def build_case(
    *,
    time_step_size: float = TIME_STEP_SIZE,
    steps: int = N_STEPS,
    solution_directory: Path = Path("solution"),
    sample_directory: Path = Path("samples") / CASE_NAME,
    run_plan: vpm.RunPlan | None = None,
) -> vpm.VPMCase:
    """Build the shared rotor physics and native output cadences.

    The optional ``run_plan`` is a lifecycle control used by the separate
    restart-pilot utility. It does not alter the ordinary VLM/VPM physics,
    plane geometry, or native sample schedules.
    """
    solution_directory = Path(solution_directory)
    sample_directory = Path(sample_directory)
    relative_sample_directory = sample_directory.relative_to("samples")

    blade_file = TUTORIAL_DIR / "assets/blade.json"
    rotation_period = 2.0 * np.pi / ANGULAR_VELOCITY
    ramp_time = RAMP_ROTATIONS * rotation_period
    rotation_kinematics = vpm.RotatingVLM(
        angular_speed=-ANGULAR_VELOCITY,
        axis=[1.0, 0.0, 0.0],
        acceleration_time=ramp_time,
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
        mesh=vpm.VLMMeshSetup.geometric(ratio=3.0),
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        wake_core_overlap=2.5,
        sample_surface_forces=True,
        force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
    )

    # Completion planes are compact 1D/2D sections: +/-0.65D in each
    # transverse coordinate. Their extent is checked against native frames in
    # the completion ledger; no 3D/5D plane is emitted by this setup.
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
    # Resolve the approach flow, rotor disk and downstream evolution at fixed
    # radial offsets. Native CSVs retain all three signed velocity components.
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
            resource_limits=vpm.ResourceLimits(
                max_particles=RUN_PARTICLE_SOFT_LIMIT,
                max_rss_bytes=RUN_RSS_LIMIT_BYTES,
                min_available_memory_bytes=RUN_AVAILABLE_MEMORY_FLOOR_BYTES,
            ),
        )
    return vpm.VPMCase(
        name=CASE_NAME,
        numerics=vpm.Numerics(
            time_step_size=time_step_size,
            compute_device="AUTO",
            integrator=vpm.SSPRK3(),
            vlm=vlm_setup,
            freestream_velocity=[FREESTREAM_SPEED, 0.0, 0.0],
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=DEFAULT_SMAGORINSKY_COEFFICIENT
            ),
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                particle_spacing=FIXED_WAKE_SPACING,
            ),
            induction=vpm.TreecodeInduction(
                stretching_scheme="transposed",
                theta=0.3,
                multipole_order=3,
                sort_particle_targets=True,
                traversal_block_dim=32,
            ),
            particle_kernel="GAUSSIAN",
            max_n_particles=MAX_N_PARTICLES,
            write_precision="f32",
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
    """Run one ordinary fresh rotor case in an explicit output namespace."""
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
    solver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-tag",
        help=(
            "fresh-run namespace under solution/ and samples/rotor/; required to avoid "
            "overwriting an existing native result"
        ),
    )
    run(parser.parse_args().output_tag)
