#!/usr/bin/env python3
"""Continue the fixed-core coverage pilot to qualify a CPU thread count.

The source checkpoint and every numerical setting remain fixed.  Only the
supported ``TI_CPU_MAX_NUM_THREADS`` runtime setting and fresh output namespace
may differ between sequential qualifications.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

import h5py

import openonda.vpm as vpm
from openonda.tutorial_runner import case_package


TUTORIAL = Path(__file__).resolve().parents[1]
__package__ = case_package(TUTORIAL) + ".assets"
from .. import setup_les

PARTICLE_COUNT = 27_200
PARTICLE_SPACING = 0.05
PARTICLE_CORE_RADIUS = 0.06
SMAGORINSKY_COEFFICIENT = 0.24080542149013215


def _checkpoint_state(path: Path) -> tuple[int, float, int]:
    with h5py.File(path, "r") as handle:
        return (
            int(handle["solver"].attrs["step"]),
            float(handle["solver"].attrs["time"]),
            int(handle["solver"].attrs["n_particles_total"]),
        )


def build_continuation(case_name: str, *, steps: int, wall_minutes: float):
    case = setup_les.build_case(
        "baseline",
        scenario="seeded_breakdown",
        compute_device="CPU",
        steps=steps,
        wall_minutes=wall_minutes,
        particle_spacing=PARTICLE_SPACING,
        particle_core_radius=PARTICLE_CORE_RADIUS,
        smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT,
        case_name=case_name,
    )
    return replace(case, run=replace(case.run, initial_samples=False))


def run(
    resume: Path,
    case_name: str,
    *,
    source_step: int,
    threads: int,
    steps: int,
    wall_minutes: float,
) -> None:
    configured_threads = os.environ.get("TI_CPU_MAX_NUM_THREADS")
    if configured_threads != str(threads):
        raise RuntimeError(
            "set the supported TI_CPU_MAX_NUM_THREADS environment variable to "
            f"{threads} before starting this process"
        )
    resume = resume.resolve()
    if not resume.is_file():
        raise FileNotFoundError(f"native checkpoint does not exist: {resume}")
    expected_state = (source_step, source_step * 0.0075, PARTICLE_COUNT)
    if _checkpoint_state(resume) != expected_state:
        raise ValueError(
            "coverage continuation checkpoint does not match the declared "
            f"step/time/count {expected_state}"
        )
    for path in (TUTORIAL / "solution" / case_name, TUTORIAL / "samples" / case_name):
        if path.exists():
            raise FileExistsError(f"fresh qualification namespace already exists: {path}")

    solver = vpm.VPMSolver(build_continuation(case_name, steps=steps, wall_minutes=wall_minutes))
    try:
        solver.load_backup(resume)
        solver.run()
    finally:
        solver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--source-step", type=int, default=40)
    parser.add_argument("--threads", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--wall-minutes", type=float, default=5.0)
    args = parser.parse_args()
    run(
        args.resume,
        args.case_name,
        source_step=args.source_step,
        threads=args.threads,
        steps=args.steps,
        wall_minutes=args.wall_minutes,
    )


if __name__ == "__main__":
    main()
