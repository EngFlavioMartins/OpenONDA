"""Run a time-limited cube-flow trial without changing the production setup."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import cube_flow_setup as case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--case-directory", type=Path)
    parser.add_argument("--eta-blend-width", type=float)
    arguments = parser.parse_args()

    if arguments.case_directory is not None:
        case.CASE_DIR = arguments.case_directory.resolve()
        case.CASE_DIR.mkdir(parents=True, exist_ok=True)
        case.VPM_SETUP = replace(
            case.VPM_SETUP,
            checkpoint_directory=str(case.CASE_DIR / "solution"),
        )

    time = replace(case.FVM_SETUP.time, end_time=arguments.end_time)
    case.FVM_SETUP = replace(case.FVM_SETUP, time=time)
    case.COUPLER_SETUP = replace(
        case.COUPLER_SETUP,
        eta_blend_width=(
            case.COUPLER_SETUP.eta_blend_width
            if arguments.eta_blend_width is None
            else arguments.eta_blend_width
        ),
    )
    case.main()


if __name__ == "__main__":
    main()
