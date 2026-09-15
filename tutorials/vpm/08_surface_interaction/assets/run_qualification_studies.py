#!/usr/bin/env python3
"""Generate inspectable real-coupling qualification tables.

This driver intentionally runs the same declarative case builder used by the
tutorial.  It does not manufacture a response from a mock field.  Each row in
the resulting tables is derived from owner-emitted CSV/HDF5 output from a real
VPM--VLM run.

The default ``--smoke`` campaign is short and exercises every table schema.
``--full`` increases the accepted physical time and is the campaign used for
the tutorial evidence package.  Results are written below ``studies/`` in
separate case directories so no output cadence is mixed across variants.
"""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .. import setup as tutorial


CASE_DIR = Path(__file__).resolve().parents[1]
STUDY_DIR = CASE_DIR / "studies"


@dataclass(frozen=True)
class RunSpec:
    name: str
    category: str
    time_step_size: float = tutorial.TIME_STEP_SIZE
    particle_spacing: float = tutorial.PARTICLE_SPACING
    n_chordwise_panels: int = 3
    n_spanwise_panels: int = 4
    precision: str = "f64"
    boundary_response: str = "responsive"
    include_ring: bool = True
    tandem_surfaces: bool = True


def _specs(steps: int) -> tuple[RunSpec, ...]:
    return (
        RunSpec(
            "smooth_incident_plate",
            "smooth_incident_field",
            include_ring=False,
            tandem_surfaces=False,
        ),
        RunSpec("tandem_baseline", "tandem"),
        RunSpec(
            "timestep_half",
            "time_step_refinement",
            time_step_size=tutorial.TIME_STEP_SIZE / 2.0,
        ),
        RunSpec(
            "surface_refined",
            "surface_refinement",
            n_chordwise_panels=4,
            n_spanwise_panels=6,
        ),
        RunSpec(
            "particle_refined",
            "particle_refinement",
            particle_spacing=tutorial.PARTICLE_SPACING / np.sqrt(2.0),
        ),
        RunSpec("f32_cpu", "precision_backend", precision="f32"),
        RunSpec("lagged_control", "two_way_response_control", boundary_response="lagged"),
    )


def _environment() -> dict[str, object]:
    """Capture reproducibility metadata without changing the run."""
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=CASE_DIR.parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--short"],
            cwd=CASE_DIR.parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_sha, git_status = None, []
    try:
        import taichi as ti

        taichi_version = ti.__version__
    except Exception:  # pragma: no cover - environment metadata only
        taichi_version = None
    return {
        "git_sha": git_sha,
        "worktree_dirty": bool(git_status),
        "worktree_status_lines": git_status,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "taichi": taichi_version,
        "configured_device": "CPU",
        "configured_backend": "DirectInduction",
        "configured_precision": "f64 baseline; f32 comparison",
        "particle_kernel": "GAUSSIAN",
        "surface_model": "global finite-segment horseshoe VLM",
        "boundary_policy": "responsive except lagged_control",
    }


def _latest_backup(case_directory: Path) -> Path:
    backups = sorted((case_directory / "solution" / "tandem").glob("vpm_*.h5"))
    if not backups:
        raise FileNotFoundError(f"no VPM backup was written below {case_directory}")
    return backups[-1]


def _cloud_summary(checkpoint: Path) -> dict[str, float | int]:
    """Extract a final real ring/wake state from the numerical backup."""
    with h5py.File(checkpoint, "r") as file:
        position = file["particles/position"][:]
        strength = file["particles/vortex_strength"][:]
        group_id = file["particles/group_id"][:]
    ring = group_id == 7
    magnitude = np.linalg.norm(strength[ring], axis=1)
    if not ring.any() or magnitude.sum() == 0.0:
        centroid = np.full(3, np.nan)
    else:
        centroid = np.average(position[ring], axis=0, weights=magnitude)
    return {
        "n_particles_final": int(len(position)),
        "ring_particles_final": int(ring.sum()),
        "ring_centroid_x": float(centroid[0]),
        "ring_centroid_y": float(centroid[1]),
        "ring_centroid_z": float(centroid[2]),
        "max_particle_strength": float(np.linalg.norm(strength, axis=1).max())
        if len(strength)
        else 0.0,
    }


def _case_metrics(spec: RunSpec, case_directory: Path, runtime: float, steps: int) -> list[dict]:
    sample_directory = case_directory / "samples" / "tandem"
    summary = pd.read_csv(sample_directory / "qualification_summary.csv")
    cloud = _cloud_summary(_latest_backup(case_directory))
    common = {
        "case": spec.name,
        "category": spec.category,
        "time_step_size": spec.time_step_size,
        "particle_spacing": spec.particle_spacing,
        "n_panels_per_surface": spec.n_chordwise_panels * spec.n_spanwise_panels * 2,
        "precision": spec.precision,
        "compute_device": "CPU",
        "boundary_response": spec.boundary_response,
        "accepted_steps": steps,
        "runtime_seconds": runtime,
        **cloud,
    }
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                **common,
                "surface": row["surface"],
                "n_force_samples": int(row["n_force_samples"]),
                "final_lift": float(row["final_lift"]),
                "peak_abs_lift": float(row["peak_abs_lift"]),
                "integrated_lift": float(row["integrated_lift"]),
                "final_drag": float(row["final_drag"]),
            }
        )
    return rows


def _run_spec(spec: RunSpec, *, steps: int) -> tuple[list[dict], Path]:
    case_directory = STUDY_DIR / spec.name
    case_directory.mkdir(parents=True, exist_ok=True)
    run_steps = steps * (2 if spec.name == "timestep_half" else 1)
    start = time.perf_counter()
    summary_path = tutorial.run(
        n_steps=run_steps,
        compute_device="CPU",
        precision=spec.precision,
        directory=case_directory,
        sample_directory="tandem",
        name=spec.name,
        time_step_size=spec.time_step_size,
        particle_spacing=spec.particle_spacing,
        n_chordwise_panels=spec.n_chordwise_panels,
        n_spanwise_panels=spec.n_spanwise_panels,
        boundary_response=spec.boundary_response,
        include_ring=spec.include_ring,
        tandem_surfaces=spec.tandem_surfaces,
    )
    del summary_path
    runtime = time.perf_counter() - start
    return _case_metrics(spec, case_directory, runtime, run_steps), case_directory


def _restart_equivalence(steps: int) -> dict[str, object]:
    """Compare one accepted responsive step after restart to a continuous run."""
    if steps < 2:
        steps = 2
    continuous_dir = STUDY_DIR / "restart_continuous"
    segment_dir = STUDY_DIR / "restart_segment"
    resumed_dir = STUDY_DIR / "restart_resumed"
    tutorial.run(
        n_steps=steps,
        precision="f64",
        directory=continuous_dir,
        sample_directory="tandem",
        name="restart_continuous",
    )
    segment_case = tutorial.build_case(
        n_steps=steps - 1,
        precision="f64",
        directory=segment_dir,
        sample_directory="tandem",
        name="restart_segment",
    )
    first = tutorial.vpm.VPMSolver(segment_case)
    try:
        for _ in range(steps - 1):
            first.advance(defer_output=True)
        first.save_backup()
        checkpoint = segment_dir / "solution" / "tandem" / f"vpm_{steps - 1:06d}.h5"
    finally:
        first.close()
    resumed_case = tutorial.build_case(
        n_steps=steps,
        precision="f64",
        directory=resumed_dir,
        sample_directory="tandem",
        name="restart_resumed",
    )
    resumed = tutorial.vpm.VPMSolver(resumed_case)
    try:
        resumed.load_backup(checkpoint)
        resumed.advance(defer_output=True)
        resumed.save_backup()
        resumed_checkpoint = resumed_dir / "solution" / "tandem" / f"vpm_{steps:06d}.h5"
    finally:
        resumed.close()
    continuous_checkpoint = _latest_backup(continuous_dir)
    with h5py.File(continuous_checkpoint, "r") as left, h5py.File(resumed_checkpoint, "r") as right:
        position_error = float(
            np.max(np.abs(left["particles/position"][:] - right["particles/position"][:]))
        )
        strength_error = float(
            np.max(
                np.abs(left["particles/vortex_strength"][:] - right["particles/vortex_strength"][:])
            )
        )
        circulation_error = float(
            np.max(np.abs(left["solver/vlm/circulation"][:] - right["solver/vlm/circulation"][:]))
        )
        step = int(left["solver"].attrs["step"])
        time_value = float(left["solver"].attrs["time"])
    return {
        "case": "responsive_restart_equivalence",
        "continuous_checkpoint": str(continuous_checkpoint.relative_to(STUDY_DIR)),
        "resumed_checkpoint": str(resumed_checkpoint.relative_to(STUDY_DIR)),
        "step": step,
        "time": time_value,
        "max_position_abs_error": position_error,
        "max_strength_abs_error": strength_error,
        "max_vlm_circulation_abs_error": circulation_error,
        "pass": bool(
            position_error <= 1e-11 and strength_error <= 1e-11 and circulation_error <= 1e-11
        ),
    }


def run_campaign(*, steps: int, smoke: bool) -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / "baseline_environment.json").write_text(
        json.dumps(_environment(), indent=2), encoding="utf-8"
    )
    specs = _specs(steps)
    if smoke:
        specs = specs[:2]
    all_rows: list[dict] = []
    case_paths: dict[str, str] = {}
    for spec in specs:
        rows, path = _run_spec(spec, steps=steps)
        all_rows.extend(rows)
        case_paths[spec.name] = str(path.relative_to(STUDY_DIR))

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(STUDY_DIR / "qualification_metrics.csv", index=False)
    refinement = metrics[
        metrics["category"].isin(
            {"tandem", "time_step_refinement", "surface_refinement", "particle_refinement"}
        )
    ].copy()
    refinement.to_csv(STUDY_DIR / "refinement_table.csv", index=False)
    loads = metrics[
        [
            "case",
            "surface",
            "final_lift",
            "peak_abs_lift",
            "integrated_lift",
            "final_drag",
            "runtime_seconds",
        ]
    ]
    loads.to_csv(STUDY_DIR / "load_runtime_table.csv", index=False)
    restart = _restart_equivalence(steps)
    (STUDY_DIR / "restart_equivalence.json").write_text(
        json.dumps(restart, indent=2), encoding="utf-8"
    )
    (STUDY_DIR / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign": "real_vpm_vlm_surface_interaction",
                "smoke": smoke,
                "requested_steps": steps,
                "case_directories": case_paths,
                "tables": [
                    "qualification_metrics.csv",
                    "refinement_table.csv",
                    "load_runtime_table.csv",
                    "restart_equivalence.json",
                ],
                "qualification_limits": {
                    "restart_max_abs_error": 1e-11,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote qualification tables to {STUDY_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke", action="store_true", help="run only baseline and dt-refinement cases"
    )
    mode.add_argument("--full", action="store_true", help="run all comparison cases")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    smoke = not args.full
    steps = args.steps if args.steps is not None else (2 if smoke else 4)
    if steps < 1:
        raise SystemExit("--steps must be positive")
    run_campaign(steps=steps, smoke=smoke)


if __name__ == "__main__":
    main()
