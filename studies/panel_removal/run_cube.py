"""Isolated, restartable cube A/B experiment using the historical native mesh.

The equations and numerical settings come from the production cube tutorial.
The study controls panel participation, output volume, process count, run endpoint
and the maximum interface sweep count; stopping tolerances remain unchanged.
Existing tutorial outputs are never overwritten.
"""

import argparse
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.vpm as vpm

ROOT = Path(__file__).resolve().parents[2]


def configuration(
    output, panel, cores=4, end_time=30.0, interface_iterations=6, snapshot_interval=0.25
):
    """Build the isolated cube comparison from the tutorial's physical case.

    output is the experiment directory; panel selects body-panel participation.
    cores is the positive MPI process count, end_time is in s, and
    interface_iterations is the maximum sweeps per coupled step (default six).
    snapshot_interval is the coupled checkpoint and retained field interval in s.
    It must be a positive multiple of the coupling step. The stopping tolerances
    and historical native mesh remain those of the case.
    Returns FVM setup, VPM case, coupling settings and the native-mesh Path;
    a missing mesh raises FileNotFoundError. Configuration does not start a run.
    """
    case = importlib.import_module("tutorials.coupled_fvm_vpm.02_cube_flow.setup")
    output_steps = snapshot_interval / case.VPM_TIME_STEP_SIZE
    if (
        not math.isfinite(output_steps)
        or output_steps < 1
        or not math.isclose(output_steps, round(output_steps), rel_tol=0, abs_tol=1e-10)
    ):
        raise ValueError("snapshot_interval must be a positive multiple of the coupling step")
    fvm_setup = replace(
        case.FVM_SETUP,
        cores=cores,
        time=replace(
            case.FVM_SETUP.time,
            end_time=end_time,
            output_schedule=fvm.RunSchedule(every_time=snapshot_interval),
        ),
        samplers=tuple(
            s for s in case.FVM_SAMPLERS if isinstance(s, fvm.ForceSampler | fvm.LineSampler)
        ),
    )
    numerics = case.VPM_CASE.numerics
    if not panel:
        numerics = replace(numerics, panel_solver=None, bodies=())
    vpm_case = replace(
        case.VPM_CASE,
        directory=output,
        numerics=numerics,
        run=replace(case.VPM_CASE.run, steps=round(end_time / case.VPM_TIME_STEP_SIZE)),
        samplers=vpm.Samplers(
            samples=tuple(s for s in case.VPM_SAMPLERS if isinstance(s, vpm.LineSampler))
        ),
    )
    settings = replace(
        case.COUPLER_SETUP,
        backup_interval_steps=round(output_steps),
        interface_iterations=interface_iterations,
    )
    mesh = case.CASE_DIR / "constant/mesh.npz"
    if not mesh.is_file():
        raise FileNotFoundError(f"The baseline native mesh is required: {mesh}")
    return fvm_setup, vpm_case, settings, mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--end-time", type=float, default=30.0)
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=0.25,
        help="Coupled checkpoint and retained FVM/VPM frame interval in seconds",
    )
    parser.add_argument("--interface-iterations", type=int, default=6)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "tutorials/coupled_fvm_vpm/02_cube_flow":
        raise ValueError("Use an isolated experiment directory")
    fvm_setup, vpm_case, settings, mesh = configuration(
        output,
        args.panel,
        args.cores,
        args.end_time,
        args.interface_iterations,
        args.snapshot_interval,
    )
    with coupling.create_coupler(
        fvm_setup,
        vpm_case,
        settings,
        mesh=mesh,
        case_dir=output,
        require_empty_output=not args.restart,
    ) as solver:
        solver.initialize()
        if solver._is_master:
            output.joinpath("experiment.json").write_text(
                json.dumps(
                    {
                        "panel": args.panel,
                        "mesh": str(mesh),
                        "cores": args.cores,
                        "end_time": args.end_time,
                        "snapshot_interval": args.snapshot_interval,
                        "interface_iterations": args.interface_iterations,
                        "start": "fresh" if not args.restart else "coupled_checkpoint",
                        "purpose": "Clean cube panel-removal trajectory; unchanged equations and tolerances, optional higher interface sweep cap",
                        "body_mask": "cube mask inferred by the coupler from native wall geometry",
                    },
                    indent=2,
                )
                + "\n"
            )
            if not args.panel and solver.vorticity_transfer._body_bounds is None:
                raise RuntimeError("Panel-free cube requires the native body solid mask")
        solver.run(
            restart_from=output / "solution/backups" if args.restart else None, backup_at_stop=True
        )


if __name__ == "__main__":
    main()
