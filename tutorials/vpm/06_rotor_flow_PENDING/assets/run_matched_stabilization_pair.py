#!/usr/bin/env python3
"""Run one auditable member of the rotor baseline/stabilized pair.

A changed physical model is not a valid changed-step restart.  This utility
therefore creates a fresh case for each variant using the public
``StabilizationConfig`` API.  Run the baseline and stretching-viscosity
variants with the same ``--dt`` and ``--endpoint`` before comparing them.
After a matched pair reaches the diagnostic endpoint, ``--resume`` can extend
that same variant from its own checkpoint without changing its physics.

Examples::

    python assets/run_matched_stabilization_pair.py \
        --variant baseline --output-tag matched_baseline_7p5 --endpoint 7.5
    python assets/run_matched_stabilization_pair.py \
        --variant selective_eddy_viscosity --output-tag matched_stabilized_7p5 \
        --endpoint 7.5
    python assets/run_matched_stabilization_pair.py \
        --variant baseline --resume solution/matched_baseline_7p5/vpm_001250.h5 \
        --output-tag matched_baseline_9p0 --endpoint 9.0
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import re

import h5py
import numpy as np

import openonda.vpm as vpm

if not __package__:
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

from ..setup import (  # noqa: E402
    CASE_NAME,
    END_TIME,
    TIME_STEP_SIZE,
    TUTORIAL_DIR,
    build_case,
)

MAX_ENDPOINT = 10.0
STRETCHING_VISCOSITY_COEFFICIENT = 0.5
VARIANTS = ("baseline", "selective_eddy_viscosity")


def _validate_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("--output-tag must be a simple filesystem name")
    return value


def _validate_endpoint(value: float) -> float:
    if not np.isfinite(value) or not 0.0 < value <= MAX_ENDPOINT:
        raise ValueError(f"endpoint must be finite and lie in (0, {MAX_ENDPOINT}]")
    return float(value)


def _steps_to_endpoint(start_time: float, endpoint: float, time_step_size: float) -> int:
    """Return exact accepted steps for a bounded endpoint on a fixed clock."""
    if not np.isfinite(start_time) or not np.isfinite(endpoint):
        raise ValueError("start_time and endpoint must be finite")
    if not np.isfinite(time_step_size) or time_step_size <= 0.0:
        raise ValueError("time_step_size must be finite and positive")
    remaining = float(endpoint) - float(start_time)
    if remaining <= 0.0:
        raise ValueError("endpoint must be later than the checkpoint time")
    steps = int(round(remaining / time_step_size))
    if steps < 1 or not np.isclose(
        steps * time_step_size,
        remaining,
        rtol=0.0,
        atol=max(1.0e-12, 1.0e-9 * time_step_size),
    ):
        raise ValueError("endpoint must be an integer number of accepted steps from the start")
    return steps


def _stabilization(variant: str) -> vpm.StabilizationConfig:
    if variant == "baseline":
        return vpm.StabilizationConfig.disabled()
    if variant == "selective_eddy_viscosity":
        return vpm.StabilizationConfig.selective_eddy_viscosity(
            coefficient=STRETCHING_VISCOSITY_COEFFICIENT
        )
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


def _output_directories(output_tag: str) -> tuple[Path, Path]:
    tag = _validate_tag(output_tag)
    return Path("solution") / tag, Path("samples") / CASE_NAME / tag


def build_trial_case(
    variant: str,
    *,
    output_tag: str,
    steps: int,
    time_step_size: float = TIME_STEP_SIZE,
    resumed: bool = False,
) -> vpm.VPMCase:
    """Build a fresh or same-model continuation case without private mutation."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    solution_directory, sample_directory = _output_directories(output_tag)
    case = build_case(
        time_step_size=time_step_size,
        steps=steps,
        solution_directory=solution_directory,
        sample_directory=sample_directory,
    )
    run = replace(
        case.run,
        # The source identity remains Numerics.compute_device=AUTO; CPU is an
        # explicit runtime-only choice for this auditable diagnostic pair.
        runtime_compute_device="CPU",
        initial_samples=False if resumed else case.run.initial_samples,
    )
    return replace(
        case,
        numerics=replace(case.numerics, stabilization=_stabilization(variant)),
        run=run,
    )


def _checkpoint_state(path: Path) -> tuple[int, float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"restart checkpoint does not exist: {path}")
    with h5py.File(path, "r") as archive:
        solver = archive["solver"]
        return (
            int(solver.attrs["step"]),
            float(solver.attrs["time"]),
            float(solver.attrs["time_step_size"]),
        )


def _require_fresh_namespace(output_tag: str) -> tuple[Path, Path]:
    solution_directory, sample_directory = _output_directories(output_tag)
    for relative in (solution_directory, sample_directory):
        if (TUTORIAL_DIR / relative).exists():
            raise FileExistsError(
                f"matched-trial namespace already exists: {relative}; choose a fresh --output-tag"
            )
    return solution_directory, sample_directory


def run(
    *,
    variant: str,
    output_tag: str,
    endpoint: float = END_TIME,
    time_step_size: float = TIME_STEP_SIZE,
    resume: Path | None = None,
) -> None:
    """Run one fresh matched member or extend its own unchanged-model checkpoint."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    endpoint = _validate_endpoint(endpoint)
    solution_directory, _ = _require_fresh_namespace(output_tag)

    source_step = 0
    source_time = 0.0
    if resume is not None:
        resume = Path(resume)
        if not resume.is_absolute():
            resume = TUTORIAL_DIR / resume
        source_step, source_time, source_dt = _checkpoint_state(resume)
        if not np.isclose(time_step_size, source_dt, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "same-model extension must retain the checkpoint time step; "
                f"requested {time_step_size:.17g}, stored {source_dt:.17g}"
            )
    steps = _steps_to_endpoint(source_time, endpoint, time_step_size)
    case = build_trial_case(
        variant,
        output_tag=output_tag,
        steps=steps,
        time_step_size=time_step_size,
        resumed=resume is not None,
    )
    solver = vpm.VPMSolver(case)
    try:
        if resume is not None:
            solver.load_backup(resume)
        solver.run()
    finally:
        solver.close()
    if solver.run_status not in {
        "completed",
        "wall_time_limit",
        "resource_limit",
        "resolution_lost",
    }:
        raise RuntimeError(f"matched trial failed: {solver.run_status}")
    print(
        f"matched trial status: {solver.run_status}; variant={variant}; "
        f"source_step={source_step}; source_time={source_time:.9g}; "
        f"target_time={endpoint:.9g}; steps={steps}; solution={solution_directory}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--endpoint", type=float, default=END_TIME)
    parser.add_argument("--dt", type=float, default=TIME_STEP_SIZE)
    parser.add_argument(
        "--resume", type=Path, help="same-variant checkpoint for an unchanged-model extension"
    )
    args = parser.parse_args()
    run(
        variant=args.variant,
        output_tag=args.output_tag,
        endpoint=args.endpoint,
        time_step_size=args.dt,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
