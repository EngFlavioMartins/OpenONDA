"""Run the focused unsteady cube-flow calibration matrix in isolated directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import TextIO, TypedDict

import numpy as np

from measure_trial_errors import frame, load_table, profile_record


ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
REFERENCE_TRIAL = CASE_DIR / "reference_flow" / "assets" / "run_trial.py"
COUPLED_TRIAL = ASSETS_DIR / "run_trial.py"
SEED_STEP = 200
END_TIME = 2.5


class AccuracyResult(TypedDict):
    n_metrics: int
    worst_error: float
    worst_metric: str
    max_cd_error: float
    max_profile_error: float


class CostResult(TypedDict):
    n_steps: int
    median_step_seconds: float
    median_vpm_seconds: float
    median_particles: int
    peak_particles: int


class CalibrationResult(AccuracyResult, CostResult):
    passes_accuracy: bool


def _launch(command: list[str], log_path: Path) -> tuple[subprocess.Popen[str], TextIO]:
    stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=CASE_DIR,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, stream


def _backup_step(path: Path) -> int | None:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        return int(json.loads(manifest.read_text(encoding="utf-8"))["coupling_step"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _capture_seed(baseline: Path, seed: Path) -> bool:
    # Once captured, the seed is deliberately independent from the live
    # baseline backup, whose manifest subsequently advances to t=2.5 s.
    if seed.exists():
        return _backup_step(seed) == SEED_STEP
    backup = baseline / "solution" / "backups"
    if _backup_step(backup) != SEED_STEP:
        return False
    temporary = seed.with_name(f".{seed.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(backup, temporary)
    temporary.replace(seed)
    return True


def _is_complete(case: Path) -> bool:
    force_path = case / "samples" / "forces_history.csv"
    if not force_path.is_file():
        return False
    try:
        return float(load_table(force_path)["time"][-1]) >= END_TIME - 1.0e-10
    except (IndexError, KeyError, OSError, ValueError):
        return False


def _run_command(command: list[str], log_path: Path) -> None:
    process, stream = _launch(command, log_path)
    return_code = process.wait()
    stream.close()
    if return_code:
        raise RuntimeError(f"command failed with status {return_code}: {' '.join(command)}")


def _accuracy(case: Path, reference: Path) -> AccuracyResult:
    sample_dir = case / "samples"
    reference_dir = reference / "samples"
    values: list[tuple[str, float]] = []

    candidate_force = load_table(sample_dir / "forces_history.csv")
    reference_force = load_table(reference_dir / "forces_history.csv")
    for sample_time, drag in zip(
        candidate_force["time"], candidate_force["drag_coefficient"], strict=True
    ):
        if sample_time <= 2.0:
            continue
        expected = float(
            np.interp(
                sample_time,
                reference_force["time"],
                reference_force["drag_coefficient"],
            )
        )
        values.append((f"Cd@{sample_time:g}", abs(float(drag) - expected) / abs(expected)))

    for name in ("centreline", "offaxis_y075"):
        reference_table = load_table(reference_dir / f"{name}.csv")
        for source in ("fvm", "vpm"):
            candidate_table = load_table(sample_dir / f"{source}_{name}.csv")
            for sample_time in np.unique(candidate_table["time"]):
                if sample_time <= 2.0:
                    continue
                candidate_frame = frame(candidate_table, float(sample_time))
                reference_frame = frame(reference_table, float(sample_time))
                if candidate_frame is None or reference_frame is None:
                    continue
                record = profile_record(
                    source,
                    name,
                    float(sample_time),
                    candidate_frame,
                    reference_frame,
                )
                values.append(
                    (
                        f"{source}-{name}@{sample_time:g}",
                        float(record["max_abs_over_u_inf"]),
                    )
                )

    if not values:
        raise RuntimeError(f"no unsteady accuracy metrics found in {case}")
    worst_name, worst_value = max(values, key=lambda item: item[1])
    cd = [value for name, value in values if name.startswith("Cd@")]
    profiles = [value for name, value in values if not name.startswith("Cd@")]
    return {
        "n_metrics": len(values),
        "worst_error": worst_value,
        "worst_metric": worst_name,
        "max_cd_error": max(cd),
        "max_profile_error": max(profiles),
    }


def _cost(case: Path) -> CostResult:
    diagnostics_path = case / "solution" / "coupler_diagnostics.jsonl"
    records = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if float(record["time"]) > 2.0]
    if not records:
        raise RuntimeError(f"no unsteady cost records found in {case}")
    total = [float(record["timing_seconds"]["total"]) for record in records]
    vpm = [float(record["timing_seconds"]["vpm"]) for record in records]
    # ``n_transfer_particles`` is the fixed FVM injection lattice and therefore
    # cannot measure pruning.  The next step's pre-replacement particle count
    # is the cloud left by VPM evolution/GBD and is the relevant cost signal.
    particles = [int(record["transfer"]["n_particles_before"]) for record in records]
    return {
        "n_steps": len(records),
        "median_step_seconds": statistics.median(total),
        "median_vpm_seconds": statistics.median(vpm),
        "median_particles": int(statistics.median(particles)),
        "peak_particles": max(particles),
    }


def _write_summary(workspace: Path, cases: dict[str, Path]) -> None:
    results: dict[str, CalibrationResult] = {}
    for name, path in cases.items():
        accuracy = _accuracy(path, workspace / "reference")
        results[name] = {
            **accuracy,
            **_cost(path),
            "passes_accuracy": accuracy["worst_error"] <= 0.05,
        }

    passing = [name for name, result in results.items() if result["passes_accuracy"]]
    most_accurate = min(results, key=lambda name: results[name]["worst_error"])
    fastest_passing = (
        None if not passing else min(passing, key=lambda name: results[name]["median_step_seconds"])
    )
    payload = {
        "objective": "all t>2 metrics <=5%; then minimize step time and particles",
        "most_accurate": most_accurate,
        "fastest_passing": fastest_passing,
        "results": results,
    }
    (workspace / "calibration_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Cube calibration results",
        "",
        "| ID | worst error | max Cd | max profile | median step | peak particles | pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['worst_error']:.3%} | {result['max_cd_error']:.3%} | "
            f"{result['max_profile_error']:.3%} | {result['median_step_seconds']:.1f} s | "
            f"{result['peak_particles']:,} | {'yes' if result['passes_accuracy'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"**Most accurate:** {most_accurate}",
            "",
            f"**Fastest passing:** {fastest_passing or 'none'}",
            "",
        ]
    )
    (workspace / "calibration_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-directory", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only a completed baseline/reference matrix workspace",
    )
    arguments = parser.parse_args()
    workspace = arguments.workspace_directory.resolve()
    if workspace.exists() and any(workspace.iterdir()):
        if not arguments.resume:
            raise FileExistsError(f"calibration workspace is not empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)

    reference = workspace / "reference"
    baseline = workspace / "B0_baseline"
    seed = workspace / "seed_t2"
    python = sys.executable
    reference_command = [
        python,
        str(REFERENCE_TRIAL),
        "--end-time",
        str(END_TIME),
        "--output-directory",
        str(reference),
    ]
    baseline_command = [
        python,
        str(COUPLED_TRIAL),
        "--end-time",
        str(END_TIME),
        "--case-directory",
        str(baseline),
    ]
    if arguments.resume:
        if not _is_complete(baseline) or not _capture_seed(baseline, seed):
            raise RuntimeError(
                "resume requires a completed B0 baseline and a valid exact t=2 backup seed"
            )
        print("resuming from completed B0 baseline and t=2 seed", flush=True)
    else:
        baseline_process, baseline_log = _launch(
            baseline_command, workspace / "B0_baseline_runner.log"
        )

        last_report = 0.0
        while baseline_process.poll() is None:
            _capture_seed(baseline, seed)
            now = time.monotonic()
            if now - last_report >= 60.0:
                step = _backup_step(baseline / "solution" / "backups")
                print(f"baseline running; latest backup step={step}", flush=True)
                last_report = now
            time.sleep(10.0)
        baseline_log.close()
        if baseline_process.returncode:
            raise RuntimeError(f"baseline failed with status {baseline_process.returncode}")
        if not _capture_seed(baseline, seed):
            raise RuntimeError("baseline finished without capturing the t=2 backup")

    # Keep all solver runs sequential. Concurrent CPU/GPU work would make the
    # baseline timing incomparable with the restart variants.
    if _is_complete(reference):
        print("reference already complete", flush=True)
    else:
        if reference.exists() and any(reference.iterdir()):
            raise RuntimeError("reference directory is incomplete; use a new calibration workspace")
        print("starting reference", flush=True)
        _run_command(reference_command, workspace / "reference_runner.log")

    variants = {
        "P1_full_panel": (1.0, "full"),
        "T2_threshold": (2.0, "vpm_boundary_condition"),
        "T4_threshold": (4.0, "vpm_boundary_condition"),
    }
    cases = {"B0_baseline": baseline}
    for name, (threshold_scale, panel_scope) in variants.items():
        case_path = workspace / name
        if _is_complete(case_path):
            print(f"{name} already complete", flush=True)
            cases[name] = case_path
            continue
        if case_path.exists() and any(case_path.iterdir()):
            raise RuntimeError(f"{name} directory is incomplete; use a new calibration workspace")
        command = [
            python,
            str(COUPLED_TRIAL),
            "--end-time",
            str(END_TIME),
            "--case-directory",
            str(case_path),
            "--restart-from",
            str(seed),
            "--gbd-threshold-scale",
            str(threshold_scale),
            "--panel-coupling-scope",
            panel_scope,
            "--allow-transfer-config-differences",
        ]
        print(f"starting {name}", flush=True)
        _run_command(command, workspace / f"{name}_runner.log")
        cases[name] = case_path

    _write_summary(workspace, cases)
    print((workspace / "calibration_results.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
