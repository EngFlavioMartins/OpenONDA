"""Run an isolated cube-flow trial with strict, resumable step limits."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import cube_flow_setup as case  # noqa: E402

TRANSFER_RESTART_ALLOWLIST = frozenset(
    {
        "coupler.transfer_method",
        "coupler.eta_blend_width",
        "coupler.vpm_only_width",
        "coupler.transfer_vorticity_cutoff",
        "coupler.transfer_boundary_prune_multiplier",
        "coupler.transfer_amplification_cap",
        "coupler.transfer_discretization_error_limit",
        "coupler.fvm_consistency_width",
        # Accepted only to resume checkpoints written by the brief two-file
        # checkpoint-history implementation. It no longer exists in CouplerSetup.
        "coupler.vpm_checkpoint_retention",
        "vpm.viscous.gbd_threshold",
        "panel.coupling_scope",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end-time",
        type=float,
        help=(
            "configure the physical horizon; on restart it must match the checkpoint configuration"
        ),
    )
    parser.add_argument(
        "--coupling-steps",
        type=int,
        help="stop after this many coupling steps without changing the solver setup",
    )
    parser.add_argument("--case-directory", type=Path, required=True)
    parser.add_argument("--eta-blend-width", type=float)
    parser.add_argument("--transfer-vorticity-cutoff", type=float)
    parser.add_argument("--transfer-boundary-prune-multiplier", type=float)
    parser.add_argument("--gbd-threshold-scale", type=float, default=1.0)
    parser.add_argument(
        "--panel-coupling-scope",
        choices=("vpm_boundary_condition", "full"),
        default="vpm_boundary_condition",
    )
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument(
        "--allow-transfer-config-differences",
        action="store_true",
        help="allow only the enumerated transfer fields to differ from a restart seed",
    )
    arguments = parser.parse_args()

    if arguments.end_time is None and arguments.coupling_steps is None:
        raise ValueError("specify --end-time, --coupling-steps, or both")
    if arguments.end_time is not None and arguments.end_time <= 0.0:
        raise ValueError("end time must be positive")
    if arguments.coupling_steps is not None and arguments.coupling_steps <= 0:
        raise ValueError("coupling steps must be positive")
    if arguments.gbd_threshold_scale <= 0.0:
        raise ValueError("GBD threshold scale must be positive")
    if (
        arguments.transfer_vorticity_cutoff is not None
        and arguments.transfer_vorticity_cutoff < 0.0
    ):
        raise ValueError("transfer vorticity cutoff must be non-negative")
    if (
        arguments.transfer_boundary_prune_multiplier is not None
        and arguments.transfer_boundary_prune_multiplier <= 0.0
    ):
        raise ValueError("transfer boundary prune multiplier must be positive")
    if arguments.allow_transfer_config_differences and arguments.restart_from is None:
        raise ValueError("--allow-transfer-config-differences requires --restart-from")
    case.CASE_DIR = arguments.case_directory.resolve()
    if any((case.CASE_DIR / name).exists() for name in ("solution", "samples")):
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

    if arguments.end_time is not None:
        time = replace(case.FVM_SETUP.time, end_time=arguments.end_time)
        case.FVM_SETUP = replace(case.FVM_SETUP, time=time)
    case.COUPLER_SETUP = replace(
        case.COUPLER_SETUP,
        eta_blend_width=(
            case.COUPLER_SETUP.eta_blend_width
            if arguments.eta_blend_width is None
            else arguments.eta_blend_width
        ),
        transfer_vorticity_cutoff=(
            case.COUPLER_SETUP.transfer_vorticity_cutoff
            if arguments.transfer_vorticity_cutoff is None
            else arguments.transfer_vorticity_cutoff
        ),
        transfer_boundary_prune_multiplier=(
            case.COUPLER_SETUP.transfer_boundary_prune_multiplier
            if arguments.transfer_boundary_prune_multiplier is None
            else arguments.transfer_boundary_prune_multiplier
        ),
    )
    case.main(
        restart_from=(None if arguments.restart_from is None else arguments.restart_from.resolve()),
        restart_allowed_config_differences=(
            TRANSFER_RESTART_ALLOWLIST if arguments.allow_transfer_config_differences else ()
        ),
        max_coupling_steps=arguments.coupling_steps,
        checkpoint_at_stop=True,
    )


if __name__ == "__main__":
    main()
