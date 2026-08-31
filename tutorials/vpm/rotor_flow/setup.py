#!/usr/bin/env python3
"""Rotor in forward flight with a fully-coupled VLM--VPM wake (LES).

A three-bladed rotor flies at a tip-speed ratio of 7.0. The wake is resolved
with vortex particles whose position and strength use common Runge--Kutta stages;
the blade loading and the downstream wake planes are sampled for the
``allplot.sh`` figures.

Usage:
    python setup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

TUTORIAL_DIR = Path(__file__).resolve().parent
CASE_NAME = "rotor"

FREESTREAM_SPEED = 7.0
TIP_SPEED_RATIO = 7.0
ROTOR_RADIUS = 6.0
HUB_RADIUS = 1.0
KINEMATIC_VISCOSITY = 1.5e-5
AIR_DENSITY = 1.225
N_RADIAL_STATIONS = 23
ANGULAR_VELOCITY = TIP_SPEED_RATIO * FREESTREAM_SPEED / ROTOR_RADIUS

TREECODE_THETA = 0.20
TIME_STEP_SIZE = 0.006
N_STEPS = 2400
DEFAULT_SMAGORINSKY_COEFFICIENT = 0.17
RAMP_ROTATIONS = 1.0
SAMPLE_INTERVAL_TIME = 0.12  # write a snapshot every this many seconds
BACKUP_INTERVAL_TIME = 0.03  # about 26 animation frames per rotor revolution
ROTATION_PERIOD = 2.0 * np.pi / ANGULAR_VELOCITY
PLANE_SAMPLING_ROTATIONS = 6.0
PLANE_SAMPLING_START_TIME = max(
    0.0, N_STEPS * TIME_STEP_SIZE - PLANE_SAMPLING_ROTATIONS * ROTATION_PERIOD
)


def nominal_wake_spacing(time_step_size: float) -> float:
    """Return the resolved wake length used by displacement subcycling.

    A VLM step creates one streamwise row of particles.  The limiting nominal
    spacing is therefore the smaller of the radial panel spacing and the
    fully-spun-up tip travel per macro step.
    """
    radial_spacing = (ROTOR_RADIUS - HUB_RADIUS) / (N_RADIAL_STATIONS - 1)
    tip_streamwise_spacing = ANGULAR_VELOCITY * ROTOR_RADIUS * time_step_size
    return min(radial_spacing, tip_streamwise_spacing)


def cadence_steps(period: float, time_step_size: float) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / time_step_size))


def build_case(
    sample_interval_time: float,
    backup_interval_time: float,
    *,
    vlm_setup: vpm.VLMSetup | None = None,
    samplers: tuple[vpm.SurfaceSampler, ...] | list[vpm.SurfaceSampler] = (),
    time_step_size: float = TIME_STEP_SIZE,
    smagorinsky_coefficient: float = DEFAULT_SMAGORINSKY_COEFFICIENT,
    steps: int = N_STEPS,
) -> vpm.VPMCase:
    """Build the complete declarative rotor case."""
    wake_spacing = nominal_wake_spacing(time_step_size)
    return vpm.VPMCase(
        numerics=vpm.Numerics(
            time_step_size=time_step_size,
            compute_device="AUTO",
            time_integration="COUPLED",
            advection=vpm.AdvectionConfig(scheme="RK2"),
            vlm=vlm_setup,
            freestream_velocity=[FREESTREAM_SPEED, 0.0, 0.0],
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=smagorinsky_coefficient
            ),
            stretching=vpm.StretchingConfig.transposed(
                scheme="RK2",
                use_treecode=True,
                treecode_theta=TREECODE_THETA,
            ),
            stabilization=vpm.StabilizationConfig.bounded_domain(
                bounds=[
                    -2.0 * ROTOR_RADIUS,
                    20.0 * ROTOR_RADIUS,
                    -2.0 * ROTOR_RADIUS,
                    2.0 * ROTOR_RADIUS,
                    -2.0 * ROTOR_RADIUS,
                    2.0 * ROTOR_RADIUS,
                ]
            ),
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                particle_spacing=wake_spacing,
            ),
            velocity=vpm.VelocityConfig.treecode(
                theta=TREECODE_THETA,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            particle_kernel="WINCKELMANS",
            write_precision="f32",
        ),
        backup=Backup(
            interval_steps=cadence_steps(backup_interval_time, time_step_size),
            directory="solution",
            log_directory="solution",
        ),
        samplers=Samplers(
            samples=(
                vpm.FlowIntegralsSampler(
                    schedule=vpm.EverySteps(cadence_steps(sample_interval_time, time_step_size))
                ),
                *samplers,
            ),
            directory=CASE_NAME,
        ),
        run=vpm.RunPlan(steps=steps),
        directory=TUTORIAL_DIR,
    )


def main() -> int:
    from assets.generate_openvsp_blade import RotorBladeDesign, generate_rotorflow_openvsp_blade

    sample_interval_time = SAMPLE_INTERVAL_TIME
    backup_interval_time = BACKUP_INTERVAL_TIME
    n_steps = N_STEPS
    time_step_size = TIME_STEP_SIZE
    smagorinsky_coefficient = DEFAULT_SMAGORINSKY_COEFFICIENT

    blade_file = TUTORIAL_DIR / "assets/blade.json"

    blade_design = RotorBladeDesign(
        rotor_radius=ROTOR_RADIUS,
        hub_radius=HUB_RADIUS,
        root_chord=0.6,
        tip_chord=0.35,
        freestream_speed=FREESTREAM_SPEED,
        tip_speed_ratio=TIP_SPEED_RATIO,
        design_axial_induction_factor=1.0 / 3.0,
        design_angle_of_attack_degrees=5.0,
        n_radial_stations=N_RADIAL_STATIONS,
        n_chordwise_stations=7,
    )

    if Path(blade_file).exists():
        print(f"Using cached VLM blade surface: {blade_file} (skipping OpenVSP regeneration)")
    else:
        generate_rotorflow_openvsp_blade(
            output_dir=str(TUTORIAL_DIR / "assets/openvsp"),
            json_path=str(blade_file),
            design=blade_design,
        )

    rotation_period = 2.0 * np.pi / ANGULAR_VELOCITY
    ramp_time = RAMP_ROTATIONS * rotation_period

    def rotor_angular_velocity(t: float) -> np.ndarray:
        if ramp_time > 0.0 and t < ramp_time:
            factor = np.sin(0.5 * np.pi * max(t, 0.0) / ramp_time) ** 2
        else:
            factor = 1.0
        return np.array([-ANGULAR_VELOCITY * factor, 0.0, 0.0])

    rotation_kinematics = vpm.ManeuverVLM(
        angular_velocity_function=rotor_angular_velocity,
        rotation_centre=np.zeros(3),
    )

    vlm_setup = vpm.VLMSetup(
        surfaces=tuple(
            vpm.VLMSurfaceSetup(
                str(blade_file),
                name=f"blade_{blade_index}",
                kinematics=rotation_kinematics,
                rotation_degrees=(azimuth, 0.0, 0.0),
            )
            for blade_index, azimuth in enumerate((0.0, 120.0, 240.0))
        ),
        mesh=vpm.VLMMeshSetup.geometric(ratio=3.0),
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        sample_surface_forces=True,
        logging_interval_steps=cadence_steps(sample_interval_time, TIME_STEP_SIZE),
    )

    # Downstream planes at 1.5R, 3R, and 4.5R.
    off_wake = ROTOR_RADIUS * 1.2
    sample_spacing = ROTOR_RADIUS / 36
    plane_schedule = vpm.EverySteps(
        cadence_steps(sample_interval_time, TIME_STEP_SIZE),
        start_time=PLANE_SAMPLING_START_TIME,
    )
    plane_samplers = [
        vpm.SurfaceSampler(
            point=[x_loc, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-off_wake, off_wake, -off_wake, off_wake],
            spacing=sample_spacing,
            file_name=f"slice_x{int(round(x_loc))}m",
            include_derivatives=False,
            schedule=plane_schedule,
        )
        for x_loc in [1.5 * ROTOR_RADIUS, 3.0 * ROTOR_RADIUS, 4.5 * ROTOR_RADIUS]
    ]

    case = build_case(
        sample_interval_time,
        backup_interval_time,
        vlm_setup=vlm_setup,
        samplers=plane_samplers,
        time_step_size=time_step_size,
        smagorinsky_coefficient=smagorinsky_coefficient,
        steps=n_steps,
    )

    print("\n===== SIMULATION =====")
    vpm.VPMSolver(case).run()
    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
