#!/usr/bin/env python3
"""Qualified, restartable cylinder campaign. Use --dry-run before provisioning."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import re

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import mesh, native, setup
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow.case_definition import (
    GRIDS,
    CONTROL_DOMAINS,
    DOMAIN,
    domain_for,
    spacing_for,
)

CASE_DIR = Path(__file__).resolve().parent
MESH_CASES = (*GRIDS, *CONTROL_DOMAINS)
MAX_ACCEPTED_TRANSITION_CONCAVE_CELLS = 8


class CampaignMismatch(ValueError):
    """Existing outputs belong to another immutable campaign."""


class CampaignBusy(RuntimeError):
    """Another process owns the campaign; do not alter its reports."""


def sha256(path):
    return mesh._sha256(Path(path))


def force_digest(path, until=None):
    """CSV content fingerprint insensitive to restart writer's newline format."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for index, line in enumerate(stream):
            if index == 0 or until is None or float(line.split(b",", 1)[0]) <= until + 1e-12:
                digest.update(line.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path):
    config = json.loads(Path(path).read_text())
    expected = set(json.loads((CASE_DIR / "study_config.json").read_text()))
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError("Study configuration fields/schema do not match study_config.json")
    for key in expected - {"tolerance_percent", "schema_version"}:
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], (int, float))
            or not math.isfinite(config[key])
            or config[key] <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    if config["end_time"] <= config["discard_time"]:
        raise ValueError("end_time must exceed discard_time")
    if int(config["minimum_cycles"]) != config["minimum_cycles"] or config["minimum_cycles"] < 20:
        raise ValueError("minimum_cycles must be an integer >= 20")
    if (
        int(config["retained_checkpoints"]) != config["retained_checkpoints"]
        or config["retained_checkpoints"] < 2
    ):
        raise ValueError("retained_checkpoints must be an integer >= 2")
    if set(config["tolerance_percent"]) != set(
        json.loads((CASE_DIR / "study_config.json").read_text())["tolerance_percent"]
    ):
        raise ValueError("Unexpected convergence metrics")
    if any(
        not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0
        for x in config["tolerance_percent"].values()
    ):
        raise ValueError("Metric tolerances must be finite and positive")
    for key in ("force_interval", "checkpoint_interval", "field_interval", "end_time"):
        count = config[key] / config["dt"]
        if count < 1 or not math.isclose(count, round(count), abs_tol=1e-7, rel_tol=0):
            raise ValueError(f"{key} must be an integer multiple of dt")
    return config


def run_matrix(config):
    runs = []
    for grid, dx in GRIDS.items():
        variants = [("base", 1, False)]
        if grid != "coarse":
            variants += [("dt2", 2, False), ("dt4", 4, False), ("tight", 1, True)]
        for variant, divisor, tight in variants:
            runs.append(
                {
                    "name": grid if variant == "base" else f"{grid}_{variant}",
                    "grid": grid,
                    "dx": dx,
                    "dt": config["dt"] / divisor,
                    "tight": tight,
                    "domain": list(DOMAIN),
                }
            )
    runs.extend(
        {
            "name": name,
            "grid": name,
            "dx": GRIDS["fine"],
            "dt": config["dt"],
            "tight": False,
            "domain": list(domain),
        }
        for name, domain in CONTROL_DOMAINS.items()
    )
    return runs


def source_identity():
    paths = sorted((ROOT / "source/solvers/fvm").rglob("*.py"))
    paths += sorted(CASE_DIR.glob("*.py")) + sorted((CASE_DIR / "assets").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def resource_check(cells, config):
    required = cells * config["estimated_bytes_per_cell"] / 2**30
    if cells > config["max_cells"] or required > config["memory_budget_gib"]:
        raise RuntimeError(
            f"Resource gate: {cells:,} cells estimate {required:.1f} GiB, "
            f"budget {config['memory_budget_gib']:g} GiB / {config['max_cells']:,} cells. "
            "Provision a suitable machine; do not silently omit fine."
        )
    return required


def live_resource_check(directory, cells, config):
    import psutil

    required = resource_check(cells, config) * 2**30
    if required > 0.8 * psutil.virtual_memory().available:
        raise RuntimeError(
            "Insufficient currently available RAM; close other workloads before resuming"
        )
    # Reserve room for a new checkpoint/output alongside the rolling copies.
    if psutil.disk_usage(directory).free < config["disk_reserve_gib"] * 2**30 + cells * 1200:
        raise RuntimeError("Disk reserve reached; no new solver run/checkpoint will be written")


def prune_checkpoints(directory, keep):
    owned = sorted(
        path
        for path in directory.glob("checkpoint-*.npz")
        if len(path.stem) == 21 and path.stem[11:].isdigit()
    )
    for path in owned[: -int(keep)]:
        path.unlink()


def estimated_cells(grid):
    """Old measured D/40 volume scaling plus outer-domain/wrapper overhead."""
    dx = spacing_for(grid)
    domain = domain_for(grid)
    span = domain[5] - domain[4]
    area = (domain[1] - domain[0]) * (domain[3] - domain[2])
    old_volume = 523534 * (0.025 / dx) ** 3 * span / 1.2
    added_area = max(0.0, area - 448.0)
    outer_overhead = added_area * span / (8 * dx) ** 3 + 2 * added_area / (8 * dx) ** 2
    measured_scaling = 1.08 * (old_volume + outer_overhead)
    background = setup.background_cell_size(dx, domain=domain)
    # The thin-domain safety cap can add a full far-field lattice that the old
    # locally cropped measurement did not contain.  Reserve a conservative
    # wrapper/transition margin before launching the real mesh process.
    complete_box_floor = 1.5 * area * span / background**3
    return round(max(measured_scaling, complete_box_floor))


def verify_mesh(directory, grid):
    record = json.loads((directory / "mesh_manifest.json").read_text())
    if (
        record["case"] != grid
        or record["requested_dx"] != spacing_for(grid)
        or record.get("domain") != list(domain_for(grid))
        or record["source_stl_sha256"] != sha256(setup.CYLINDER_STL)
    ):
        raise ValueError(f"Mesh provenance mismatch: {directory}")
    for filename in ("mesh.npz", "mesh.vtu", "mesh_report.json"):
        if sha256(directory / filename) != record["output_sha256"][filename]:
            raise ValueError(f"Mesh checksum mismatch: {directory / filename}")
    aliases = {
        "mesher": ROOT / "source/solvers/fvm/mesh/cartesian/mesher.py",
        "mesh_entry": CASE_DIR / "mesh.py",
        "setup": CASE_DIR / "setup.py",
        "canonical_surface": CASE_DIR / "canonical_surface.py",
        "case_definition": CASE_DIR / "case_definition.py",
    }
    for name, digest in record["code_files"].items():
        if sha256(aliases.get(name, ROOT / name)) != digest:
            raise ValueError(f"Mesh code changed: {name}; use a fresh campaign")
    return record


def independent_check(directory):
    """Require the native checker's explicit verdict, not only exit code zero."""
    from source.solvers.fvm.io.mesh_storage import load_native_mesh
    from source.solvers.fvm.io.openfoam_poly_mesh import write_poly_mesh

    executable = native.CHECKMESH if native.CHECKMESH.is_file() else shutil.which("checkMesh")
    if not executable:
        raise RuntimeError("Independent checkMesh is unavailable; activate OpenFOAM first")
    executable = Path(executable).resolve()
    checked_hash = sha256(directory / "mesh.npz")
    marker = directory / "independent_check.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if (
            record.get("mesh_sha256") == checked_hash
            and record.get("checker_sha256") == sha256(executable)
            and record.get("passed")
            and sha256(directory / "checkMesh.log") == record.get("log_sha256")
        ):
            return record
    case = directory / "openfoam_check"
    system = case / "system"
    system.mkdir(parents=True, exist_ok=True)
    write_poly_mesh(load_native_mesh(directory / "mesh.npz"), case / "constant/polyMesh")
    (system / "controlDict").write_text(native._control_dict())
    (system / "fvSchemes").write_text(
        native._foam_header("fvSchemes", "system")
        + "ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } "
        "divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; } "
        "interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n"
    )
    (system / "fvSolution").write_text(native._foam_header("fvSolution", "system"))
    command = ([str(native.LAUNCHER)] if native.LAUNCHER.is_file() else []) + [
        str(executable),
        "-case",
        str(case),
        "-allTopology",
        "-allGeometry",
    ]
    log = directory / "checkMesh.log"
    with log.open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, timeout=1200, check=False
        )
    output = log.read_text()
    passed, accepted_concave_cells = _checkmesh_verdict(output, result.returncode)
    record = {
        "passed": passed,
        "accepted_concave_cells": accepted_concave_cells,
        "mesh_sha256": checked_hash,
        "checker_sha256": sha256(executable),
        "command": command,
        "return_code": result.returncode,
        "log_sha256": sha256(log),
    }
    write_json(marker, record)
    if not passed:
        raise RuntimeError(f"Independent mesh qualification failed; inspect {log}")
    return record


def _checkmesh_verdict(output, return_code):
    """Accept only Mesh OK or the measured star-shaped transition exception."""
    if return_code != 0:
        return False, 0
    if "Mesh OK." in output and "Failed " not in output:
        return True, 0
    match = re.search(r"\*\*\*Concave cells .* number of cells:\s*(\d+)", output)
    failed = re.search(r"Failed\s+(\d+)\s+mesh checks", output)
    warning_lines = [line for line in output.splitlines() if line.lstrip().startswith("***")]
    mandatory = (
        "Topological cell zip-up check OK.",
        "Cell volumes OK.",
        "Face pyramids OK.",
        "Face interpolation weight check OK.",
        "Cell determinant check OK.",
    )
    count = int(match.group(1)) if match else 0
    accepted = (
        failed is not None
        and int(failed.group(1)) == 1
        and len(warning_lines) == 1
        and match is not None
        and 0 < count <= MAX_ACCEPTED_TRANSITION_CONCAVE_CELLS
        and all(marker in output for marker in mandatory)
    )
    return accepted, count if accepted else 0


def flow_setup(spec, config):
    import openonda.fvm as fvm

    result = setup.solver_setup(spec["name"], spec["dx"])
    # One process per run: the present large-mesh qualification is serial.
    # Changing MPI/backend is a new campaign, not an unrecorded restart option.
    force = fvm.ForceSampler(
        patch_names=["cylinder"],
        reference_velocity=1.0,
        reference_area=setup.DIAMETER * (spec["domain"][5] - spec["domain"][4]),
        reference_length=setup.DIAMETER,
        moment_centre=[0.0, 0.0, 0.0],
        file_name="forces_history",
        schedule=fvm.RunSchedule(every_time=config["force_interval"]),
    )
    result = replace(
        result,
        cores=1,
        samplers=(force,),
        time=replace(
            result.time,
            time_step_size=spec["dt"],
            end_time=config["end_time"],
            adjustment=None,
            output_schedule=fvm.RunSchedule(every_time=config["field_interval"]),
        ),
        backup=replace(result.backup, schedule=None, write_at_end=False),
    )
    if spec["tight"]:
        result = replace(
            result,
            linear=replace(
                result.linear,
                pressure_tolerance=1e-9,
                pressure_relative_tolerance=0.0005,
                momentum_tolerance=1e-8,
                momentum_relative_tolerance=0.005,
            ),
            pimple=replace(result.pimple, n_outer_correctors=4, n_correctors=3),
        )
    return result


def run_flow(output, name):
    """Resume only an atomically published checkpoint/health pair."""
    import numpy as np
    import openonda.fvm as fvm

    manifest = json.loads((output / "campaign.json").read_text())
    config = manifest["config"]
    spec = next(x for x in manifest["runs"] if x["name"] == name)
    directory = output / "runs" / name
    directory.mkdir(parents=True, exist_ok=True)
    solution_directory = output / "solution" / name
    samples_directory = output / "samples" / name
    mesh_directory = output / "meshes" / spec["grid"]
    verify_mesh(mesh_directory, spec["grid"])
    digest = sha256(mesh_directory / "mesh.npz")
    progress_file = directory / "progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else None
    if progress and progress["mesh_sha256"] != digest:
        raise ValueError("Run's mesh identity changed")
    if progress and progress["spec"] != spec:
        raise ValueError("Run's numerical specification changed")
    if progress and progress["completed"]:
        if force_digest(samples_directory / "forces_history.csv") != progress["force_sha256"]:
            raise ValueError("Completed force history checksum changed")
        return
    health = (
        progress["health"]
        if progress
        else {
            "steps": 0,
            "max_courant": 0.0,
            "max_continuity": 0.0,
            "max_residual": 0.0,
            "max_net_flux": 0.0,
            "all_finite": True,
            "all_linear_converged": True,
        }
    )
    solver = fvm.create_fvm_solver(
        flow_setup(spec, config),
        case_dir=directory,
        solution_dir=solution_directory,
        samples_dir=samples_directory,
        mesh=mesh_directory / "mesh.npz",
    )
    try:
        if progress:
            checkpoint = directory / progress["checkpoint"]
            if sha256(checkpoint) != progress["checkpoint_sha256"]:
                raise ValueError("Restart checksum changed")
            if (
                progress["force_sha256"] is not None
                and force_digest(samples_directory / "forces_history.csv", progress["time"])
                != progress["force_sha256"]
            ):
                raise ValueError("Force history before restart was modified")
            solver.load_state(checkpoint)
        else:
            # Smooth fixed physical perturbation in the wake; identical on every
            # grid. The setter initializes all time levels and flux consistently.
            points = solver.geo_data["cell_centre"][: solver.mesh_data["n_cells"]]
            velocity = np.tile([1.0, 0.0, 0.0], (len(points), 1))
            velocity[:, 1] = 0.001 * np.exp(-((points[:, 0] - 2.0) ** 2 + points[:, 1] ** 2))
            solver.set_initial_velocity(velocity)
            if (samples_directory / "forces_history.csv").exists():
                raise ValueError(
                    "Force data exists without a published restart; refusing to append"
                )
            checkpoint = directory / "checkpoint-0000000000.npz"
            solver.save_state(checkpoint)
            write_json(
                progress_file,
                {
                    "completed": False,
                    "time": 0.0,
                    "health": health,
                    "mesh_sha256": digest,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": sha256(checkpoint),
                    "force_sha256": None,
                    "spec": spec,
                },
            )
        solver.write_run_manifest()
        checkpoint_time = solver.time
        started = time.perf_counter()
        while solver.time < config["end_time"] - 1e-9:
            solver.advance()
            diagnostic = solver.last_diagnostics
            if (
                diagnostic is None
                or diagnostic.n_nonfinite_values
                or not diagnostic.linear_solves
                or not all(x.converged for x in diagnostic.linear_solves)
            ):
                raise RuntimeError(
                    "A nonfinite state or unconverged linear solve invalidated the run"
                )
            if not math.isclose(
                diagnostic.time_step_size, spec["dt"], rel_tol=1e-6, abs_tol=1e-10
            ) and not (
                solver.time >= config["end_time"] - 1e-9
                and 0 < diagnostic.time_step_size <= spec["dt"]
            ):
                raise RuntimeError("Unexpected timestep: spatial/time errors would be mixed")
            health["steps"] += 1
            health["max_courant"] = max(health["max_courant"], diagnostic.max_courant_number)
            health["max_continuity"] = max(
                health["max_continuity"], diagnostic.max_continuity_error
            )
            health["max_residual"] = max(health["max_residual"], max(diagnostic.residuals.values()))
            health["max_net_flux"] = max(
                health["max_net_flux"], abs(diagnostic.net_boundary_volumetric_flux)
            )
            complete = solver.time >= config["end_time"] - 1e-9
            if complete or solver.time >= checkpoint_time + config["checkpoint_interval"] - 1e-9:
                if (
                    shutil.disk_usage(directory).free
                    < config["disk_reserve_gib"] * 2**30 + solver.mesh_data["n_cells"] * 1200
                ):
                    raise RuntimeError(
                        "Disk reserve reached before checkpoint; last published restart preserved"
                    )
                checkpoint = directory / f"checkpoint-{solver.step:010d}.npz"
                solver.save_state(checkpoint)
                force_path = samples_directory / "forces_history.csv"
                write_json(
                    progress_file,
                    {
                        "completed": complete,
                        "time": solver.time,
                        "health": health,
                        "mesh_sha256": digest,
                        "checkpoint": checkpoint.name,
                        "checkpoint_sha256": sha256(checkpoint),
                        "force_sha256": force_digest(force_path),
                        "spec": spec,
                    },
                )
                prune_checkpoints(directory, config["retained_checkpoints"])
                write_json(
                    output / "status.json",
                    {
                        "status": "running",
                        "phase": "flow",
                        "case": name,
                        "physical_time": solver.time,
                        "end_time": config["end_time"],
                        "steps": solver.step,
                        "max_courant": health["max_courant"],
                        "pid": os.getpid(),
                        "updated_at": time.time(),
                    },
                )
                checkpoint_time = solver.time
                print(
                    f"[study] {name}: t={solver.time:g}, steps={solver.step}, elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    finally:
        solver.close()


def campaign(output, config, *, mesh_only=False):
    import fcntl

    output.parent.mkdir(parents=True, exist_ok=True)
    # Persistent lock file lives outside the immutable campaign directory.
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignBusy("Another process already owns this campaign") from exc
        return _campaign(output, config, mesh_only=mesh_only)


def _campaign(output, config, *, mesh_only=False):
    expected = {
        "schema_version": 1,
        "config": config,
        "runs": run_matrix(config),
        "source_files": source_identity(),
        "source_stl_sha256": sha256(setup.CYLINDER_STL),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise CampaignMismatch(
                "Campaign configuration/source changed; select a new --output directory"
            )
    elif any(output.iterdir()):
        raise ValueError("Refusing to adopt a nonempty directory without a campaign manifest")
    else:
        write_json(path, expected)
    for grid in MESH_CASES:
        directory = output / "meshes" / grid
        (output / "samples" / grid).mkdir(parents=True, exist_ok=True)
        write_json(
            output / "status.json",
            {
                "status": "running",
                "phase": "mesh",
                "case": grid,
                "pid": os.getpid(),
                "updated_at": time.time(),
            },
        )
        estimate = estimated_cells(grid)
        live_resource_check(output, estimate, config)
        if not directory.exists():
            directory.parent.mkdir(parents=True, exist_ok=True)
            with (output / f"mesh-{grid}.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(CASE_DIR / "mesh.py"),
                        "--case",
                        grid,
                        "--output-dir",
                        str(directory),
                        "--backup-dir",
                        str(output / "solution" / grid),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        verify_mesh(directory, grid)
        report = json.loads((directory / "mesh_report.json").read_text())
        resource_check(report["counts"]["cells"], config)
        independent_check(directory)
    if mesh_only:
        return
    for spec in expected["runs"]:
        write_json(
            output / "status.json",
            {
                "status": "running",
                "phase": "flow",
                "case": spec["name"],
                "pid": os.getpid(),
                "updated_at": time.time(),
            },
        )
        report = json.loads((output / "meshes" / spec["grid"] / "mesh_report.json").read_text())
        live_resource_check(output, report["counts"]["cells"], config)
        print(f"[study] run {spec['name']}, dt={spec['dt']:g}", flush=True)
        with (output / f"flow-{spec['name']}.log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__)),
                    "--output",
                    str(output),
                    "--worker",
                    spec["name"],
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CASE_DIR / "study_config.json")
    parser.add_argument("--output", type=Path, default=CASE_DIR / "study_laptop")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mesh-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--worker", choices=[x["name"] for x in run_matrix({"dt": 1})], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        run_flow(output, args.worker)
        return 0
    config = read_config(args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "config": config,
                    "runs": run_matrix(config),
                    "estimated_cells": {name: estimated_cells(name) for name in MESH_CASES},
                    "estimated_memory_gib": {
                        name: estimated_cells(name) * config["estimated_bytes_per_cell"] / 2**30
                        for name in MESH_CASES
                    },
                    "note": "No solver launched. Mesh/conditioning/checkMesh and resource gates are mandatory; meshes remain unqualified.",
                },
                indent=2,
            )
        )
        return 0
    error = None
    try:
        if not args.report_only:
            campaign(output, config, mesh_only=args.mesh_only)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, file=sys.stderr)
        if isinstance(exc, (CampaignMismatch, CampaignBusy)):
            return 1
    # Always leave a report/plot on numerical or resource failure, but never
    # write into an unrelated non-campaign directory.
    if (output / "campaign.json").is_file():
        from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow.assets.study_analysis import (
            report_campaign,
        )
        from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow.assets.plot_grid_study import (
            plot_campaign,
        )

        report = report_campaign(output, failure=error)
        plot_campaign(output, report)
        status_path = output / "status.json"
        last_status = json.loads(status_path.read_text()) if status_path.exists() else {}
        write_json(
            status_path,
            {
                **last_status,
                "status": "failed"
                if error
                else ("mesh_only_complete" if args.mesh_only else report["status"]),
                "error": error,
                "selected_mesh": report["selected_mesh"],
                "pid": os.getpid(),
                "updated_at": time.time(),
            },
        )
        print(
            f"Study: {report['status']}; selected mesh: {report['selected_mesh']}; report: {output / 'grid_study.json'}"
        )
        return 1 if error else (0 if report["grid_independent"] or args.mesh_only else 2)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
