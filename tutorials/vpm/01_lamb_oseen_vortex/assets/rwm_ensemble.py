"""Run independent random walks of one Lamb--Oseen initial field."""

import argparse

from ..setup import (
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
    *,
    first_realization: int = 0,
) -> None:
    """Advance independent random walks of the same initial vortex field."""
    for realization in range(first_realization, number_of_realizations):
        random_seed = first_random_seed + realization
        name = f"{case}_rwm_{realization:03d}"
        print(
            f"[RWM] {case} | realization {realization + 1}/{number_of_realizations} | seed={random_seed}",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("vortex", "dipole", "merging"))
    parser.add_argument("--number-of-realizations", type=int, default=RWM_ENSEMBLE_SIZE)
    parser.add_argument("--first-random-seed", type=int, default=42000)
    args = parser.parse_args()
    run_ensemble(args.case, args.number_of_realizations, args.first_random_seed)


if __name__ == "__main__":
    main()
