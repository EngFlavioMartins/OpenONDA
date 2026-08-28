#!/usr/bin/env python3
"""Run an ensemble for one stochastic Lamb--Oseen comparison case."""

from __future__ import annotations

import argparse

from lamb_oseen_setup import run_case


CIRCULATIONS = {
    "vortex": (+1.0,),
    "dipole": (+1.0, -1.0),
    "merging": (+1.0, +1.0),
}


def run_ensemble(
    case: str,
    number_of_realizations: int,
    first_random_seed: int,
    compute_device: str,
) -> None:
    """Advance independent random walks of the same initial vortex field."""
    if number_of_realizations < 4:
        raise ValueError("an RWM ensemble requires at least four realizations")

    circulations = CIRCULATIONS[case]
    case_name = f"{case}_rwm"

    for realization in range(number_of_realizations):
        random_seed = first_random_seed + realization
        print(
            f"===== {case_name}: realization "
            f"{realization + 1}/{number_of_realizations}, seed {random_seed} ====="
        )
        run_case(
            case,
            "rwm",
            circulations,
            case_name,
            compute_device,
            random_seed,
            realization,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CIRCULATIONS), required=True)
    parser.add_argument("--number-of-realizations", type=int, required=True)
    parser.add_argument("--first-random-seed", type=int, default=42000)
    parser.add_argument("--compute-device", default="CPU")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_ensemble(
        args.case,
        args.number_of_realizations,
        args.first_random_seed,
        args.compute_device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
