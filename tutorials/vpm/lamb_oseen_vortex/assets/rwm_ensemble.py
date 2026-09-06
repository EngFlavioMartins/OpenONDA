#!/usr/bin/env python3
"""Run an ensemble of independent random walks for one Lamb--Oseen case."""

from __future__ import annotations

import argparse

from tutorials.vpm.lamb_oseen_vortex.setup import (
    MERGING_SAMPLE_INTERVAL_STEPS,
    RWM_ENSEMBLE_SIZE,
    SAMPLE_INTERVAL_TIME,
    TIME_STEP_SIZE,
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
) -> None:
    """Advance independent random walks of the same initial vortex field."""
    if number_of_realizations < 4:
        raise ValueError("an RWM ensemble requires at least four realizations")

    for realization in range(number_of_realizations):
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
        )


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_ensemble(
        args.case,
        args.number_of_realizations,
        args.first_random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
