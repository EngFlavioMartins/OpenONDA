"""Compare an isolated L1 lattice trial with the unchanged coupled baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as xml

import h5py
import numpy as np
import pyvista as pv


def _table(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    values = {
        name: np.asarray([float(row[name]) for row in rows]) for name in rows[0] if name != "patch"
    }
    if "step" in values:
        reset = np.flatnonzero(np.diff(values["step"]) <= 0.0)
        if reset.size:
            start = int(reset[-1] + 1)
            values = {name: value[start:] for name, value in values.items()}
    return values


def _frames(table: dict[str, np.ndarray]) -> dict[float, dict[str, np.ndarray]]:
    return {
        float(time): {key: value[selected] for key, value in table.items()}
        for time in np.unique(table["time"])
        for selected in [np.isclose(table["time"], time, rtol=0.0, atol=1.0e-12)]
    }


def _profile_comparison(candidate: Path, baseline: Path, source: str, name: str) -> dict:
    trial_frames = _frames(_table(candidate / "samples" / f"{source}_{name}.csv"))
    baseline_frames = _frames(_table(baseline / "samples" / f"{source}_{name}.csv"))
    common = sorted(set(trial_frames) & set(baseline_frames))
    records = []
    for time in common:
        actual = trial_frames[time]
        expected = baseline_frames[time]
        order = np.argsort(actual["position_x"])
        x = actual["position_x"][order]
        velocity = actual["velocity_x"][order]
        reference_order = np.argsort(expected["position_x"])
        reference_x = expected["position_x"][reference_order]
        reference_u = expected["velocity_x"][reference_order]
        valid = np.isfinite(x) & np.isfinite(velocity)
        valid &= (x >= reference_x.min()) & (x <= reference_x.max())
        if name == "centreline":
            valid &= (x < -0.5) | (x > 0.5)
        difference = velocity[valid] - np.interp(x[valid], reference_x, reference_u)
        records.append(
            {
                "time": time,
                "mean_abs_difference": float(np.mean(np.abs(difference))),
                "max_abs_difference": float(np.max(np.abs(difference))),
            }
        )
    return {
        "n_frames": len(records),
        "max_abs_difference": max(record["max_abs_difference"] for record in records),
        "records": records,
    }


def _pvd_frames(path: Path) -> dict[float, Path]:
    root = xml.parse(path).getroot()
    return {
        float(dataset.attrib["timestep"]): path.parent / dataset.attrib["file"]
        for dataset in root.findall(".//DataSet")
    }


def _wake_vorticity_comparison(candidate: Path, baseline: Path) -> dict:
    trial = _pvd_frames(candidate / "samples" / "vpm_wake_slice_z0.pvd")
    reference = _pvd_frames(baseline / "samples" / "vpm_wake_slice_z0.pvd")
    records = []
    for time in sorted(set(trial) & set(reference)):
        actual = pv.read(trial[time])
        expected = pv.read(reference[time])
        if actual.n_points != expected.n_points or not np.allclose(actual.points, expected.points):
            raise ValueError(f"wake sample geometry differs at t={time:g}")
        difference = np.asarray(actual.point_data["vorticity"], dtype=np.float64) - np.asarray(
            expected.point_data["vorticity"], dtype=np.float64
        )
        records.append(
            {
                "time": time,
                "l1_difference": float(np.linalg.norm(difference, axis=1).sum()),
                "rms_difference": float(np.sqrt(np.mean(difference**2))),
                "max_difference": float(np.max(np.abs(difference))),
            }
        )
    return {
        "n_frames": len(records),
        "max_difference": max(record["max_difference"] for record in records),
        "records": records,
    }


def _downstream_circulation(case: Path) -> dict:
    checkpoints = sorted((case / "solution" / "checkpoints").glob("vpm_*.h5"))
    if not checkpoints:
        raise FileNotFoundError(f"no VPM checkpoint found below {case}")
    path = checkpoints[-1]
    with h5py.File(path, "r") as file:
        position = np.asarray(file["particles/position"], dtype=np.float64)
        strength = np.asarray(file["particles/vortex_strength"], dtype=np.float64)
    downstream = position[:, 0] > 1.25
    selected = strength[downstream]
    return {
        "checkpoint": str(path),
        "n_particles": int(downstream.sum()),
        "gamma_l1": float(np.linalg.norm(selected, axis=1).sum(dtype=np.float64)),
        "gamma_net": selected.sum(axis=0, dtype=np.float64).tolist(),
        "vortex_strength_max": float(np.linalg.norm(selected, axis=1).max(initial=0.0)),
        "x_min": None if not downstream.any() else float(position[downstream, 0].min()),
        "x_max": None if not downstream.any() else float(position[downstream, 0].max()),
    }


def _runtime(case: Path) -> dict:
    records = [
        json.loads(line)
        for line in (case / "solution" / "coupler_diagnostics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records = [record for record in records if float(record["time"]) > 2.0]
    totals = [float(record["timing_seconds"]["total"]) for record in records]
    particles = [int(record["transfer"]["n_particles_before"]) for record in records]
    return {
        "n_steps": len(records),
        "median_step_seconds": statistics.median(totals),
        "mean_step_seconds": statistics.fmean(totals),
        "median_pretransfer_particles": int(statistics.median(particles)),
        "peak_pretransfer_particles": max(particles),
    }


def _drag_comparison(candidate: Path, baseline: Path) -> dict:
    trial = _frames(_table(candidate / "samples" / "forces_history.csv"))
    reference = _frames(_table(baseline / "samples" / "forces_history.csv"))
    records = []
    for time in sorted(set(trial) & set(reference)):
        actual = float(trial[time]["drag_coefficient"][0])
        expected = float(reference[time]["drag_coefficient"][0])
        records.append(
            {"time": time, "l1": actual, "baseline": expected, "difference": actual - expected}
        )
    return {
        "max_abs_difference": max(abs(record["difference"]) for record in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("l1_case", type=Path)
    parser.add_argument("baseline_case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    candidate = arguments.l1_case.resolve()
    baseline = arguments.baseline_case.resolve()
    payload = {
        "l1_case": str(candidate),
        "baseline_case": str(baseline),
        "velocity_profiles": {
            f"{source}_{name}": _profile_comparison(candidate, baseline, source, name)
            for source in ("fvm", "vpm")
            for name in ("centreline", "offaxis_y075")
        },
        "wake_vorticity": _wake_vorticity_comparison(candidate, baseline),
        "wake_circulation": {
            "l1": _downstream_circulation(candidate),
            "baseline": _downstream_circulation(baseline),
        },
        "drag": _drag_comparison(candidate, baseline),
        "runtime": {"l1": _runtime(candidate), "baseline": _runtime(baseline)},
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(arguments.output)


if __name__ == "__main__":
    main()
