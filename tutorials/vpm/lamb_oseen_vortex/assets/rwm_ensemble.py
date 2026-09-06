#!/usr/bin/env python3
"""Run an ensemble of independent random walks for one Lamb--Oseen case."""

from __future__ import annotations

import argparse
import math

import pandas as pd

from tutorials.vpm.lamb_oseen_vortex.setup import (
    MERGING_SAMPLE_INTERVAL_STEPS,
    RWM_ENSEMBLE_SIZE,
    SAMPLE_INTERVAL_TIME,
    TIME_STEP_SIZE,
    TUTORIAL_DIR,
    run_case,
)


def field_interval_steps(case: str) -> int:
    return (
        MERGING_SAMPLE_INTERVAL_STEPS
        if case == "merging"
        else round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE)
    )


def run_ensemble(
    case: str,
    number_of_realizations: int = RWM_ENSEMBLE_SIZE,
    first_random_seed: int = 42000,
    *,
    first_realization: int = 0,
    resume: bool = False,
) -> None:
    """Advance independent random walks of the same initial vortex field."""
    if number_of_realizations < 4:
        raise ValueError("an RWM ensemble requires at least four realizations")

    for realization in range(first_realization, number_of_realizations):
        random_seed = first_random_seed + realization
        name = f"{case}_rwm_{realization:03d}"
        print(
            f"[RWM] {case} | realization "
            f"{realization + 1}/{number_of_realizations} | seed={random_seed}",
            flush=True,
        )
        run_case(
            case,
            "RWM",
            name=name,
            random_seed=random_seed,
            surfaces=False,
            backup_steps=field_interval_steps(case),
            resume=resume,
        )


def required_ensemble_size(current: int, relative_error: float, limit: float) -> int:
    """Plan the next independent batch using MCSE's inverse-square-root scaling."""
    if not math.isfinite(relative_error) or relative_error < 0.0:
        raise ValueError("RWM relative standard error must be finite and non-negative")
    if relative_error <= limit:
        return current
    return max(current + 4, math.ceil(1.1 * current * (relative_error / limit) ** 2))


def run_converged_ensemble(case, pilot, first_seed, maximum, resume=False):
    from .postprocess import RWM_RELATIVE_STANDARD_ERROR_LIMIT, aggregate_case

    if maximum < pilot:
        raise ValueError("maximum realizations must be at least the pilot size")
    count = pilot
    if resume:
        # Preserve previously extended ensembles instead of dropping their seeds.
        existing = list((TUTORIAL_DIR / "solution").glob(f"{case}_rwm_[0-9][0-9][0-9]"))
        count = max(count, len(existing))
    first = 0
    while True:
        run_ensemble(case, count, first_seed, first_realization=first, resume=resume)
        aggregate_case(TUTORIAL_DIR / "solution", TUTORIAL_DIR / "samples", case, count)
        convergence = pd.read_csv(TUTORIAL_DIR / "samples" / f"{case}_rwm/rwm_convergence.csv")
        values = convergence[
            ["relative_standard_error_l2_velocity", "relative_standard_error_l2_vorticity"]
        ]
        if values.isna().any().any():
            raise ValueError("RWM convergence contains missing standard errors")
        error = float(values.to_numpy().max())
        target = required_ensemble_size(count, error, RWM_RELATIVE_STANDARD_ERROR_LIMIT)
        print(f"[RWM] {case} | {count} seeds | maximum relative MCSE={error:.3%}", flush=True)
        if target == count:
            return
        if count >= maximum:
            raise RuntimeError(
                f"{case}: MCSE={error:.3%} still exceeds 7.5% with {count} seeds; increase --maximum-realizations"
            )
        first, count = count, min(target, maximum)
        print(f"[RWM] {case} | extending ensemble to {count} independent seeds", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("vortex", "dipole", "merging"))
    parser.add_argument(
        "--number-of-realizations",
        type=int,
        default=RWM_ENSEMBLE_SIZE,
        help=f"number of independent random-seed realizations (default: {RWM_ENSEMBLE_SIZE})",
    )
    parser.add_argument("--first-random-seed", type=int, default=42000)
    parser.add_argument(
        "--converge",
        action="store_true",
        help="extend independent seeds until the 7.5% precision gate passes",
    )
    parser.add_argument("--maximum-realizations", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.converge:
        run_converged_ensemble(
            args.case,
            args.number_of_realizations,
            args.first_random_seed,
            args.maximum_realizations,
            args.resume,
        )
    else:
        run_ensemble(
            args.case, args.number_of_realizations, args.first_random_seed, resume=args.resume
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
