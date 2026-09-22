#!/usr/bin/env python3
"""Safe, resumable launcher for the Re=150 cylinder study."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from openonda.cylinder_case import (
    DEFAULT_CYLINDER_CASE,
    as_config,
    file_hash,
    new_run_directory,
    write_complete_marker,
    write_manifest,
    software_fingerprint,
)
from openonda.tutorial_runner import load_case_module
from openonda.cylinder_campaign import initialize_cylinder_perturbation


CASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = CASE_DIR / "reference_flow"
INPUTS = (CASE_DIR / "setup.py", CASE_DIR / "assets" / "cylinder_long.stl")
REFERENCE_INPUTS = (
    REFERENCE_DIR / "setup.py",
    REFERENCE_DIR / "assets" / "cylinder_long.stl",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("coupled", "reference"), required=True)
    parser.add_argument("--root", type=Path, default=CASE_DIR / "study_results" / "cylinder")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-coupling-steps", type=int)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--grid", action="append", metavar="NAME=H")
    parser.add_argument("--override", action="append", metavar="NAME=VALUE")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--reference-cores", type=int)
    return parser.parse_args()


def select_run_directory(options: argparse.Namespace) -> Path:
    if _explicit_mpi():
        comm = _mpi_comm()
        if comm.Get_rank() == 0:
            selected = (
                options.run_dir.resolve()
                if options.run_dir is not None
                else new_run_directory(
                    options.root.resolve(),
                    f"{options.kind}-pilot" if options.pilot else options.kind,
                )
            )
            selected_text = str(selected)
        else:
            selected_text = None
        return Path(comm.bcast(selected_text, root=0))
    if options.run_dir is not None:
        return options.run_dir.resolve()
    label = f"{options.kind}-pilot" if options.pilot else options.kind
    return new_run_directory(options.root.resolve(), label)


def mpi_root() -> bool:
    """Return true for the one rank allowed to publish campaign metadata."""
    from openonda.runtime import detected_world_size

    if detected_world_size() == 1:
        return True
    from mpi4py import MPI

    return MPI.COMM_WORLD.Get_rank() == 0


def _explicit_mpi() -> bool:
    """Whether this process was started directly inside an MPI world.

    ``ensure_runtime`` marks its own relaunched workers with
    ``_OPENONDA_MPI_CHILD``.  Those workers intentionally retain the old
    launch contract, while an externally invoked ``mpiexec python
    run_campaign.py`` must coordinate its filesystem work here.
    """
    from openonda.runtime import detected_world_size

    return os.environ.get("_OPENONDA_MPI_CHILD") != "1" and detected_world_size() > 1


def _mpi_comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _collective_preflight(callback):
    """Run a filesystem preflight on rank zero and broadcast its decision."""
    if not _explicit_mpi():
        return callback()
    comm = _mpi_comm()
    if comm.Get_rank() == 0:
        try:
            decision = {"ok": True, "value": callback()}
        except Exception as error:  # noqa: BLE001 - the error must reach every rank
            decision = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    else:
        decision = None
    decision = comm.bcast(decision, root=0)
    if not decision["ok"]:
        raise RuntimeError(f"MPI campaign preflight failed: {decision['error']}")
    return decision["value"]


def _collective_barrier() -> None:
    if _explicit_mpi():
        _mpi_comm().Barrier()


def _collective_root_action(callback):
    """Publish a rank-zero filesystem action's success to every rank."""
    if not _explicit_mpi():
        return callback() if mpi_root() else None
    comm = _mpi_comm()
    if comm.Get_rank() == 0:
        try:
            result = {"ok": True, "value": callback()}
        except Exception as error:  # noqa: BLE001 - propagate publication failures
            result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    else:
        result = None
    result = comm.bcast(result, root=0)
    if not result["ok"]:
        raise RuntimeError(f"MPI campaign root action failed: {result['error']}")
    return result["value"]


def selected_grids(options: argparse.Namespace) -> list[tuple[str, float]]:
    if options.grid:
        values = []
        for item in options.grid:
            name, separator, spacing = item.partition("=")
            if not separator or not name or float(spacing) <= 0.0:
                raise ValueError(f"grid must be NAME=positive_spacing, got {item!r}")
            values.append((name, float(spacing)))
        return values
    grids = list(DEFAULT_CYLINDER_CASE.reference_grid())
    return grids[:1] if options.pilot else grids


def selected_overrides(options: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {}
    for item in options.override or ():
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise ValueError(f"override must be NAME=VALUE, got {item!r}")
        try:
            parsed: object = float(value)
            if parsed.is_integer():
                parsed = int(parsed)
        except ValueError:
            parsed = value
        values[name] = parsed
    return values


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _compatible_campaign_config(
    path: Path,
    expected: dict[str, object],
    *,
    allow_end_extension: bool = False,
) -> bool:
    payload = _read_json(path)
    if payload is None:
        return False
    actual = payload.get("config", {})
    for key in ("kind", "overrides"):
        if actual.get(key) != expected.get(key):
            return False
    actual_end = float(actual.get("end_time", expected.get("end_time", 0.0)))
    expected_end = float(expected.get("end_time", actual_end))
    if allow_end_extension:
        end_compatible = expected_end >= actual_end
    else:
        end_compatible = math.isclose(actual_end, expected_end, rel_tol=0.0, abs_tol=1.0e-12)
    if not end_compatible:
        return False
    expected_resolved = expected.get("resolved")
    if expected_resolved is None:
        return True
    actual_resolved = actual.get("resolved")
    if not isinstance(actual_resolved, dict):
        return False
    for key, value in expected_resolved.items():
        if key == "end_time":
            continue
        if actual_resolved.get(key) != value:
            return False
    actual_resolved_end = float(actual_resolved.get("end_time", actual_end))
    expected_resolved_end = float(expected_resolved.get("end_time", expected_end))
    return (
        expected_resolved_end >= actual_resolved_end
        if allow_end_extension
        else math.isclose(
            actual_resolved_end,
            expected_resolved_end,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )


def _grid_record_path(run_dir: Path, name: str) -> Path:
    return run_dir / "grid_manifests" / f"{name}.json"


def _grid_config(
    module, name: str, spacing: float, end_time: float, cores: int | None = None
) -> dict[str, object]:
    layers = max(4, math.ceil(module.SPAN / spacing))
    return {
        "name": name,
        "h": float(spacing),
        "hp": float(spacing),
        "span": float(module.SPAN),
        "dz": float(module.SPAN / layers),
        "end_time": float(end_time),
        "cores": getattr(module, "CORES", 6) if cores is None else cores,
        "fvm_time_step": float(module.TIME_STEP_SIZE),
        "source_hash": file_hash(REFERENCE_DIR / "setup.py"),
        "geometry_hash": file_hash(REFERENCE_DIR / "assets" / "cylinder_long.stl"),
        "software_fingerprint": software_fingerprint(),
    }


def _grid_complete(
    run_dir: Path, module, name: str, spacing: float, end_time: float, cores: int | None = None
) -> bool:
    record = _read_json(_grid_record_path(run_dir, name))
    expected = _grid_config(module, name, spacing, end_time, cores)
    if record is None or record.get("config") != expected:
        return False
    grid_run = _read_json(run_dir / "samples" / name / "grid_run.json")
    fvm_metadata = _read_json(run_dir / "solution" / name / "fvm_metadata.json")
    return bool(
        grid_run
        and grid_run.get("case") == name
        and math.isclose(
            float(grid_run.get("cell_size", 0.0)),
            spacing,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and float(grid_run.get("end_time", 0.0)) >= end_time
        and fvm_metadata
        and fvm_metadata.get("lifecycle", {}).get("status") == "complete"
        and float(fvm_metadata.get("state", {}).get("time", 0.0)) >= end_time
    )


def _grid_has_output(run_dir: Path, name: str) -> bool:
    return (run_dir / "solution" / name).exists() or (run_dir / "samples" / name).exists()


def _write_grid_record(run_dir: Path, config: dict[str, object]) -> None:
    path = _grid_record_path(run_dir, str(config["name"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": "openonda-cylinder-grid/1", "config": config}, indent=2) + "\n"
    )


def _resolved_coupled_config(
    module, end_time: float, overrides: dict[str, object]
) -> dict[str, object]:
    """Describe the resolved factory output used to authorize a restart."""
    fvm_setup, vpm_case, coupler_setup, mesh = module.build_case(
        end_time=end_time,
        overrides=overrides,
    )
    numerics = vpm_case.numerics
    viscous = numerics.viscous
    induction = numerics.induction
    midspan = next(
        sampler
        for sampler in fvm_setup.samplers
        if getattr(sampler, "file_name", None) == "fvm_midspan"
    )
    hxy = float(midspan.spacing / 2.0)
    span = float(induction.z_max - induction.z_min)
    dz_target = float(overrides.get("dz", hxy))
    axial_layers = max(4, math.ceil(span / dz_target))
    return {
        "kind": "coupled",
        "overrides": dict(overrides),
        "end_time": float(fvm_setup.time.end_time),
        "hxy": hxy,
        "fvm_bounds": list(map(float, mesh.domain.bounds)),
        "vpm_bounds": list(map(float, numerics.domain_bounds)),
        "transfer_bounds": list(map(float, coupler_setup.transfer_region_bounds)),
        "span": span,
        "dz": float(span / axial_layers),
        "hp": float(viscous.particle_spacing),
        "exchange_dt": float(numerics.time_step_size),
        "fvm_time_step": float(fvm_setup.time.time_step_size),
        "cores": int(fvm_setup.cores),
        "compute_device": str(numerics.compute_device),
        "particle_limit": int(numerics.max_n_particles),
        "transfer_region_bounds": list(coupler_setup.transfer_region_bounds),
        "transfer_method": str(coupler_setup.transfer_method),
        "boundary_condition_mode": str(coupler_setup.boundary_condition_mode),
        "interface_iterations": int(coupler_setup.interface_iterations),
        "interface_acceleration": coupler_setup.interface_acceleration,
        "source_hash": file_hash(CASE_DIR / "setup.py"),
        "geometry_hash": file_hash(CASE_DIR / "assets" / "cylinder_long.stl"),
        "software_fingerprint": software_fingerprint(),
    }


def run_reference(options: argparse.Namespace, run_dir: Path) -> None:
    if options.override:
        raise ValueError(
            "reference overrides are not supported until the reference factory exposes them"
        )
    module = load_case_module(REFERENCE_DIR)
    cores = getattr(options, "reference_cores", None)
    if cores is not None and cores < 1:
        raise ValueError("reference-cores must be positive")
    postprocess = load_case_module(REFERENCE_DIR, "postprocess_grid_study")
    grids = selected_grids(options)
    end_time = (
        4.0
        if options.pilot and options.end_time is None
        else DEFAULT_CYLINDER_CASE.reference_end_time
        if options.end_time is None
        else options.end_time
    )

    def inspect_reference_outputs() -> list[tuple[str, float]]:
        pending = []
        for name, spacing in grids:
            if _grid_complete(run_dir, module, name, spacing, end_time, cores):
                print(f"reference grid already complete: {name}")
                continue
            if _grid_has_output(run_dir, name):
                raise RuntimeError(
                    f"reference grid {name} is incomplete and has no native restart; "
                    "use a fresh --run-dir rather than resuming it"
                )
            pending.append((name, spacing))
        return pending

    pending = _collective_preflight(inspect_reference_outputs)
    if not pending:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **as_config(),
        "kind": "reference",
        "grids": [_grid_config(module, name, h, end_time, cores) for name, h in pending],
        "end_time": end_time,
        "pilot": options.pilot,
        "analysis": not options.no_analysis,
    }

    def publish_reference_manifest() -> None:
        # A previous full family may have left a campaign marker behind while
        # this invocation adds a new grid.  Per-grid records are authoritative;
        # remove the aggregate marker until the family is analysed again.
        (run_dir / "COMPLETE").unlink(missing_ok=True)
        write_manifest(
            run_dir / "campaign_manifest.json",
            status="running",
            config=config,
            inputs=REFERENCE_INPUTS,
        )

    _collective_root_action(publish_reference_manifest)
    _collective_barrier()
    if "--run-dir" not in sys.argv:
        sys.argv.extend(("--run-dir", str(run_dir)))
    for name, spacing in pending:
        solver = module.create_solver(
            name, spacing, output_root=run_dir, end_time=end_time, cores=cores
        )
        with solver:
            initialize_cylinder_perturbation(solver, module.SPAN)
            solver.run()
            module.fvm.update_grid_study(solver, spacing, profiles=("centreline",))
        _collective_root_action(
            lambda: _write_grid_record(
                run_dir, _grid_config(module, name, spacing, end_time, cores)
            )
        )
    if not options.pilot and not options.no_analysis and mpi_root():
        report = postprocess.analyse_forces(
            samples_dir=run_dir / "samples",
            output_dir=run_dir / "figures",
            start=DEFAULT_CYLINDER_CASE.reference_force_window[0],
            end=DEFAULT_CYLINDER_CASE.reference_force_window[1],
        )
        if len(report["grids"]) < 3:
            raise RuntimeError("reference campaign did not produce three valid grids")
    status = "pilot-complete" if options.pilot else "complete"
    _collective_root_action(
        lambda: write_manifest(
            run_dir / "campaign_manifest.json",
            status=status if not options.no_analysis else "grid-complete",
            config=config,
            inputs=REFERENCE_INPUTS,
        )
    )
    if status == "complete" and not options.no_analysis:
        _collective_root_action(lambda: write_complete_marker(run_dir))


def run_coupled(options: argparse.Namespace, run_dir: Path) -> None:
    module = load_case_module(CASE_DIR)
    max_steps = options.max_coupling_steps
    if options.pilot and max_steps is None:
        max_steps = 5
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max coupling steps must be positive")
    overrides = selected_overrides(options)
    physical_end = options.end_time if options.end_time is not None else module.END_TIME
    resolved_config = _resolved_coupled_config(module, physical_end, overrides)
    expected_config = {
        "kind": "coupled",
        "overrides": overrides,
        "end_time": physical_end,
        "resolved": resolved_config,
    }
    manifest_path = run_dir / "campaign_manifest.json"
    child = os.environ.get("_OPENONDA_MPI_CHILD") == "1"

    def inspect_coupled_outputs() -> dict[str, object]:
        if not child and run_dir.exists() and any(run_dir.iterdir()):
            if options.resume:
                if not _compatible_campaign_config(
                    manifest_path, expected_config, allow_end_extension=True
                ):
                    raise ValueError(
                        "coupled resume configuration differs from the existing campaign"
                    )
                existing = _read_json(manifest_path) or {}
                existing_end = float(existing.get("config", {}).get("end_time", physical_end))
                if (run_dir / "COMPLETE").is_file() and math.isclose(
                    existing_end, physical_end, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    return {"skip": True, "restart_from": None}
            elif (run_dir / "COMPLETE").is_file():
                if not _compatible_campaign_config(manifest_path, expected_config):
                    raise ValueError(
                        "completed coupled campaign configuration differs from the request"
                    )
                return {"skip": True, "restart_from": None}
            else:
                raise FileExistsError(
                    f"coupled run directory is incomplete; pass --resume with a valid backup: {run_dir}"
                )
        backup = run_dir / "solution" / "backups" / "manifest.json"
        if options.resume and backup.is_file():
            return {"skip": False, "restart_from": str(backup.parent)}
        if options.resume and not child:
            raise RuntimeError(
                "coupled resume requires the canonical solution/backups/manifest.json; "
                "use a fresh --run-dir when no coupled backup exists"
            )
        return {"skip": False, "restart_from": None}

    preflight = _collective_preflight(inspect_coupled_outputs)
    if preflight["skip"]:
        return
    restart_from = Path(preflight["restart_from"]) if preflight["restart_from"] else None
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **as_config(),
        "kind": "coupled",
        "max_coupling_steps": max_steps,
        "resume": restart_from is not None,
        "overrides": overrides,
        "end_time": physical_end,
        "resolved": resolved_config,
    }
    _collective_root_action(
        lambda: write_manifest(
            run_dir / "campaign_manifest.json", status="running", config=config, inputs=INPUTS
        )
    )
    _collective_barrier()
    if "--run-dir" not in sys.argv:
        sys.argv.extend(("--run-dir", str(run_dir)))
    completed_step = module.create_solver(
        output_root=run_dir,
        end_time=options.end_time,
        restart_from=restart_from,
        max_coupling_steps=max_steps,
        overrides=overrides,
    )
    if max_steps is None:
        expected_steps = round(physical_end / float(resolved_config["exchange_dt"]))
        if completed_step < expected_steps:
            raise RuntimeError(
                f"coupled run stopped at step {completed_step}, expected {expected_steps}"
            )
    status = "pilot-complete" if max_steps is not None else "complete"
    _collective_root_action(
        lambda: write_manifest(
            run_dir / "campaign_manifest.json", status=status, config=config, inputs=INPUTS
        )
    )
    if status == "complete":
        _collective_root_action(lambda: write_complete_marker(run_dir))


def main() -> int:
    options = arguments()
    run_dir = select_run_directory(options)
    if options.kind == "reference":
        run_reference(options, run_dir)
    else:
        run_coupled(options, run_dir)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
