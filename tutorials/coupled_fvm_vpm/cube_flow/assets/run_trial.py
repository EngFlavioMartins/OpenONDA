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
    parser.add_argument("--case-directory", type=Path, required=True)
    parser.add_argument("--eta-blend-width", type=float)
    parser.add_argument("--gbd-threshold-scale", type=float, default=1.0)
    parser.add_argument(
        "--panel-coupling-scope",
        choices=("vpm_boundary_condition", "full"),
        default="vpm_boundary_condition",
    )
    parser.add_argument("--restart-from", type=Path)
    arguments = parser.parse_args()

    if arguments.end_time <= 0.0:
        raise ValueError("end time must be positive")
    if arguments.gbd_threshold_scale <= 0.0:
        raise ValueError("GBD threshold scale must be positive")

    case.CASE_DIR = arguments.case_directory.resolve()
    if arguments.restart_from is None and any(
        (case.CASE_DIR / name).exists() for name in ("solution", "samples")
    ):
        raise FileExistsError(
            f"Trial directory is not empty: {case.CASE_DIR}. "
            "Use a new isolated directory; trial output is never appended."
        )
    case.CASE_DIR.mkdir(parents=True, exist_ok=True)

    viscous = replace(
        case.VPM_SETUP.viscous,
        gbd_threshold=(
            case.GBD_VORTICITY_FLOOR * case.VPM_PARTICLE_SPACING**3 * arguments.gbd_threshold_scale
        ),
    )
    case.VPM_PANEL_SOLVER.coupling_scope = arguments.panel_coupling_scope
    case.VPM_SETUP = replace(
        case.VPM_SETUP,
        viscous=viscous,
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
    case.main(
        restart_from=(None if arguments.restart_from is None else arguments.restart_from.resolve())
    )


if __name__ == "__main__":
    main()
