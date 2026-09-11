#!/usr/bin/env python3
"""Run a bounded, changed-step restart pilot from a native rotor checkpoint.

This utility is deliberately separate from the ordinary fresh-run setup.  It
performs a short sampler-free preflight, then a bounded continuation in a
fresh, identity-derived namespace.  The solver owns changed-step restart
provenance and terminal metadata; this driver does not edit private solver
state.

Usage:
    python assets/run_restart_pilot.py \
        --resume solution/vpm_001152.h5 \
        --resume-dt 0.001 \
        --attempt pilot
"""

from __future__ import annotations

if not __package__:
    from pathlib import Path

    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import argparse
from dataclasses import replace
import h5py
import math
from pathlib import Path
import re

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

from ..setup import CASE_NAME, END_TIME, TUTORIAL_DIR, build_case

PREFLIGHT_STEPS = 8
PILOT_STEPS = 120
PREFLIGHT_WALL_LIMIT_SECONDS = 600.0
PILOT_WALL_LIMIT_SECONDS = 1800.0
PILOT_PARTICLE_SOFT_LIMIT = 570_000
PILOT_RSS_LIMIT_BYTES = 12 * 2**30
PILOT_AVAILABLE_MEMORY_FLOOR_BYTES = 2 * 2**30


def _validate_attempt(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("--attempt must be a simple filesystem name")
    return value


def _checkpoint_state(path: Path) -> tuple[int, float, float]:
    """Read accepted clock metadata without constructing a live solver."""
    if not path.is_file():
        raise FileNotFoundError(f"restart checkpoint does not exist: {path}")
    with h5py.File(path, "r") as archive:
        solver = archive["solver"]
        return (
            int(solver.attrs["step"]),
            float(solver.attrs["time"]),
            float(solver.attrs["time_step_size"]),
        )


def _remaining_steps(start_time: float, time_step_size: float) -> int:
    """Return accepted steps needed to reach the authored bounded horizon."""
    remaining = END_TIME - start_time
    if remaining < -1.0e-10:
        raise ValueError(
            f"checkpoint time {start_time:.17g} is beyond the pilot endpoint {END_TIME:.17g}"
        )
    return max(0, int(math.ceil(max(0.0, remaining) / time_step_size - 1.0e-12)))


def _namespace_paths(
    *, source_step: int, source_time: float, time_step_size: float, attempt: str
) -> tuple[Path, Path]:
    """Build safe, deterministic output paths from the restart identity."""
    dt_token = f"{time_step_size:.12g}".replace(".", "p")
    time_token = f"{source_time:.12g}".replace(".", "p")
    tag = f"{attempt}_from_step_{source_step}_t_{time_token}_dt_{dt_token}"
    return Path("solution") / tag, Path("samples") / CASE_NAME / tag


def _require_fresh_namespace(solution_directory: Path, sample_directory: Path) -> None:
    """Fail closed so a pilot can never overwrite native output."""
    solution_root = TUTORIAL_DIR / solution_directory
    sample_root = TUTORIAL_DIR / sample_directory
    if solution_root.exists() or sample_root.exists():
        raise FileExistsError(
            "restart pilot namespace already exists; choose a new --attempt so native "
            "backups, metadata, and samples remain immutable"
        )


def _resource_limits() -> vpm.ResourceLimits:
    return vpm.ResourceLimits(
        max_particles=PILOT_PARTICLE_SOFT_LIMIT,
        max_rss_bytes=PILOT_RSS_LIMIT_BYTES,
        min_available_memory_bytes=PILOT_AVAILABLE_MEMORY_FLOOR_BYTES,
    )


def _pilot_run_plan(steps: int) -> vpm.RunPlan:
    """Return the bounded continuation lifecycle, including CPU override."""
    return vpm.RunPlan(
        steps=steps,
        initial_samples=False,
        final_backup=True,
        health_limit_action="STOP",
        wall_time_limit_seconds=PILOT_WALL_LIMIT_SECONDS,
        resource_limits=_resource_limits(),
        runtime_compute_device="CPU",
    )


def _preflight_case(
    pilot_case: vpm.VPMCase,
    solution_directory: Path,
    sample_directory: Path,
    continuation_steps: int,
) -> vpm.VPMCase:
    """Clone pilot numerics with no scientific samplers for the short preflight."""
    if continuation_steps < 1:
        raise ValueError("continuation_steps must be positive")
    preflight_solution = solution_directory / "preflight"
    preflight_samples = sample_directory / "preflight"
    preflight_plan = replace(
        _pilot_run_plan(min(PREFLIGHT_STEPS, continuation_steps)),
        wall_time_limit_seconds=PREFLIGHT_WALL_LIMIT_SECONDS,
    )
    return replace(
        pilot_case,
        name="rotor_restart_preflight",
        backup=Backup(
            interval_steps=0,
            directory=preflight_solution,
            log_directory=preflight_solution,
        ),
        samplers=Samplers(
            samples=(),
            directory=preflight_samples.relative_to("samples"),
        ),
        run=preflight_plan,
    )


def run(resume: Path, resume_dt: float, attempt: str = "pilot") -> None:
    """Execute one bounded changed-step continuation in a fresh namespace."""
    if not np.isfinite(resume_dt) or resume_dt <= 0.0:
        raise ValueError("--resume-dt must be finite and positive")
    attempt = _validate_attempt(attempt)
    resume = Path(resume)
    if not resume.is_absolute():
        resume = TUTORIAL_DIR / resume
    source_step, source_time, source_dt = _checkpoint_state(resume)
    available_steps = _remaining_steps(source_time, float(resume_dt))
    if available_steps == 0:
        raise ValueError("checkpoint is already at the authored pilot endpoint")
    pilot_steps = min(PILOT_STEPS, available_steps)
    solution_directory, sample_directory = _namespace_paths(
        source_step=source_step,
        source_time=source_time,
        time_step_size=float(resume_dt),
        attempt=attempt,
    )
    _require_fresh_namespace(solution_directory, sample_directory)

    pilot_case = build_case(
        time_step_size=float(resume_dt),
        steps=pilot_steps,
        solution_directory=solution_directory,
        sample_directory=sample_directory,
        run_plan=_pilot_run_plan(pilot_steps),
    )
    print(
        f"restart pilot: source step={source_step}, time={source_time:.9g}, "
        f"source dt={source_dt:.9g}, requested dt={resume_dt:.9g}, "
        f"target time={source_time + pilot_steps * resume_dt:.9g}, steps={pilot_steps}"
    )

    preflight_solver = vpm.VPMSolver(
        _preflight_case(pilot_case, solution_directory, sample_directory, pilot_steps)
    )
    try:
        preflight_solver.load_backup(resume, time_step_size=float(resume_dt))
        preflight_solver.run()
    finally:
        preflight_solver.close()
    if preflight_solver.run_status != "completed":
        raise RuntimeError(f"restart preflight did not complete: {preflight_solver.run_status}")

    solver = vpm.VPMSolver(pilot_case)
    try:
        solver.load_backup(resume, time_step_size=float(resume_dt))
        # Preserve an immutable, native restart point before the continuation.
        solver.save_backup()
        solver.run()
    finally:
        solver.close()
    if solver.run_status not in {
        "completed",
        "wall_time_limit",
        "resource_limit",
        "resolution_lost",
    }:
        raise RuntimeError(f"restart pilot failed: {solver.run_status}")
    print(f"restart pilot status: {solver.run_status}; metadata: {solution_directory}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, type=Path, help="native .h5 checkpoint")
    parser.add_argument("--resume-dt", required=True, type=float, help="new accepted step size")
    parser.add_argument("--attempt", default="pilot", help="fresh namespace token")
    args = parser.parse_args()
    run(args.resume, args.resume_dt, args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
