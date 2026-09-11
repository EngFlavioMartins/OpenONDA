#!/usr/bin/env python3
"""Two delta wings crossing wakes (VLM--VPM).

A leading wing and a following wing both heave and pitch through the flow,
shedding vortex wakes that the trailing wing crosses. The sampled forces and
circulation histories feed the ``allplot.sh`` figures.

The induction backend and stretching formulation are independent. Set
`stretching_scheme` to "direct", "mixed", or "transposed" in the case below.

Usage:
    python setup.py
    python setup.py --resume solution/vpm_001600.h5 --output-tag continuation_from_001600
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

import openonda.vpm as vpm
from openonda.tutorial_runner import case_package
from openonda.vpm import Backup, Samplers

__package__ = case_package(Path(__file__).parent)
from .assets.generate_surface import create_delta_wing, save_surface

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
END_TIME = 10.0  # [s]
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


def build_case(
    *,
    n_steps: int = N_STEPS,
    backup_directory: str = "solution",
    sample_directory: str = CASE_NAME,
) -> vpm.VPMCase:
    """Build the owner-clocked Delta case for a fresh or resumed run."""
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")

    # The VPM-owned backup clock is the sole surface-output clock.  With
    # dt=.0025 this is exactly 10 accepted steps, giving a dense 0.025 s
    # coupled VPM+VLM frame suitable for animation and restart.
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
            )
            for name, x_position, phase in wings
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
            directory=backup_directory,
            log_directory=backup_directory,
        ),
        samplers=Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(sample_steps)),
                *samplers,
            ),
            directory=sample_directory,
        ),
        run=vpm.RunPlan(steps=n_steps),
        directory=TUTORIAL_DIR,
    )
    return case


def _checkpoint_state(checkpoint: str | Path) -> tuple[Path, int, float]:
    """Read only the source clock needed to size a continuation run.

    The target solver remains responsible for validating the complete native
    restart, including numerical and coupled VLM identity.
    """
    path = Path(checkpoint)
    if not path.is_absolute():
        path = TUTORIAL_DIR / path
    if path.suffix != ".h5":
        path = Path(f"{path}.h5")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"restart checkpoint does not exist: {path}")
    with h5py.File(path, "r") as archive:
        solver = archive["solver"]
        return (
            path,
            int(solver.attrs["step"]),
            float(solver.attrs["time"]),
        )


def _fresh_continuation_namespace(output_tag: str) -> tuple[str, str]:
    if Path(output_tag).name != output_tag or output_tag in {"", ".", ".."}:
        raise ValueError("output_tag must be one simple directory name")
    backup_directory = TUTORIAL_DIR / "solution" / output_tag
    sample_directory = TUTORIAL_DIR / "samples" / f"{CASE_NAME}_{output_tag}"
    for directory in (backup_directory, sample_directory):
        if directory.exists():
            raise FileExistsError(f"continuation output namespace already exists: {directory}")
    return f"solution/{output_tag}", f"{CASE_NAME}_{output_tag}"


def run_from_backup(
    checkpoint: str | Path,
    *,
    output_tag: str | None = None,
    endpoint_time: float = END_TIME,
) -> None:
    """Continue the unchanged Delta model from a complete native checkpoint.

    The source checkpoint and its sibling manifest remain untouched. New dense
    owner-clocked HDF5 and sampled outputs go to a fresh namespace.
    """
    source, source_step, source_time = _checkpoint_state(checkpoint)
    if not math.isfinite(endpoint_time) or endpoint_time < source_time:
        raise ValueError("endpoint_time must be finite and no earlier than the checkpoint time")
    remaining_steps = round((endpoint_time - source_time) / TIME_STEP_SIZE)
    if not math.isclose(
        source_time + remaining_steps * TIME_STEP_SIZE,
        endpoint_time,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ValueError("endpoint_time must lie on the configured time-step grid")
    if output_tag is None:
        output_tag = f"continuation_from_{source_step:06d}"
    backup_directory, sample_directory = _fresh_continuation_namespace(output_tag)
    case = build_case(
        n_steps=remaining_steps,
        backup_directory=backup_directory,
        sample_directory=sample_directory,
    )
    solver = vpm.VPMSolver(case)
    try:
        solver.load_backup(source)
        if solver.step != source_step or not math.isclose(
            solver.time, source_time, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise RuntimeError("loaded checkpoint clock does not match its native HDF5 clock")
        solver.run()
    finally:
        solver.close()


def run() -> None:
    vpm.VPMSolver(build_case()).run()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        help="complete native HDF5 checkpoint for an unchanged-model continuation",
    )
    parser.add_argument(
        "--output-tag",
        help="fresh continuation namespace below solution/ and samples/",
    )
    parser.add_argument(
        "--endpoint-time",
        type=float,
        default=END_TIME,
        help=f"final physical time in seconds (default: {END_TIME:g})",
    )
    args = parser.parse_args()
    if args.resume is None:
        if args.output_tag is not None or not math.isclose(
            args.endpoint_time, END_TIME, rel_tol=0.0, abs_tol=1.0e-12
        ):
            parser.error("--output-tag and --endpoint-time require --resume")
        run()
    else:
        run_from_backup(
            args.resume,
            output_tag=args.output_tag,
            endpoint_time=args.endpoint_time,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
