"""Quantify peak-selection sensitivity and viscosity effects in saved ring runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from scipy.special import erf

if not __package__:
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

from . import plot_lbm_comparison as comparison
from . import postprocess


def direct_gaussian_velocity(points, position, strength, core):
    """Float64 Biot--Savart reference, independent of the tree traversal."""
    velocity = np.zeros_like(points, dtype=float)
    for start in range(0, len(points), 16):
        delta = points[start : start + 16, None, :] - position[None, :, :]
        radius = np.linalg.norm(delta, axis=2)
        q = radius / core
        fraction = erf(q) - 2 / np.sqrt(np.pi) * q * np.exp(-q * q)
        coefficient = np.divide(
            fraction, 4 * np.pi * radius**3, out=np.zeros_like(radius), where=radius > 0
        )
        velocity[start : start + 16] = np.sum(
            np.cross(strength[None, :, :], delta) * coefficient[:, :, None], axis=1
        )
    return velocity


def audit_tree_velocity(root, run, fields):
    """Check saved tree velocities near the cores against direct blob summation."""
    rows = []
    recorded = run
    for time, filename in fields.items():
        if time not in (1.5, 3.0, 4.5):
            continue
        step = int(Path(filename).stem.split("_")[-1])
        backup = root / "solution" / recorded / f"vpm_{step:06d}.h5"
        if not backup.exists():
            continue
        grid = pv.read(root / "samples" / recorded / filename)
        omega = grid.point_data["vorticity"][:, 2]
        selected = np.flatnonzero(omega > 0.2 * omega.max())
        selected = selected[np.linspace(0, len(selected) - 1, min(160, len(selected)), dtype=int)]
        with h5py.File(backup) as archive:
            particles = archive["particles"]
            exact = direct_gaussian_velocity(
                grid.points[selected].astype(float),
                particles["position"][:].astype(float),
                particles["vortex_strength"][:].astype(float),
                particles["core_radius"][:].astype(float),
            )
        approximate = grid.point_data["velocity"][selected].astype(float)
        difference = np.linalg.norm(exact - approximate, axis=1)
        rows.append(
            dict(
                time=time,
                queries=len(selected),
                backup=backup.name,
                relative_l2=float(np.linalg.norm(difference) / np.linalg.norm(exact)),
                maximum_absolute_error=float(difference.max()),
                direct_speed_max=float(np.linalg.norm(exact, axis=1).max()),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison.setup.TUTORIAL_DIR = postprocess.CASE_DIR = args.root.resolve()
    runs, fields = {}, {}
    for run in postprocess.CASES:
        if not postprocess.load_metadata(run):
            continue
        recorded = run
        peaks, sources = comparison.sampler_history(run, peak_merge_bridge=0.9)
        fields[run] = {round(row["time"], 8): row["file"] for row in sources}
        sensitivity = []
        for cutoff in (0.25, 0.5, 0.75):
            for stride in (1, 2):
                selected = peaks[peaks.step.isin(peaks.step.unique()[::stride])]
                tracks, reason = comparison.coherent_tracks(selected, secondary_peak_limit=cutoff)
                sensitivity.append(
                    dict(
                        secondary_peak_limit=cutoff,
                        snapshot_stride=stride,
                        end_time=float(tracks.time.max()),
                        termination=reason,
                        **comparison.leapfrog_events(tracks),
                    )
                )
        snapshots = []
        for path in sorted((args.root / "solution" / recorded).glob("vpm_*.h5")):
            with h5py.File(path) as archive:
                particles = archive["particles"]
                molecular = particles["kinematic_viscosity"][:].astype(float)
                extra = (
                    particles["effective_viscosity"][:].astype(float)
                    - molecular
                    - particles["eddy_viscosity"][:]
                )
                snapshots.append(
                    dict(
                        file=path.name,
                        step=int(path.stem.split("_")[-1]),
                        mean_extra_viscosity=float(extra.mean()),
                        max_extra_over_molecular=float(np.max(extra / molecular)),
                        active_fraction=float(np.mean(extra > molecular * 1e-5)),
                    )
                )
        runs[run] = dict(
            recorded_run=recorded,
            sensitivity=sensitivity,
            viscosity_snapshots=snapshots,
            direct_velocity_check=audit_tree_velocity(args.root, run, fields[run]),
            fields=sources,
            diagnostics=comparison.reported_self_diagnostics(run),
        )
    contrasts = []
    for method in runs:
        if method == "baseline":
            continue
        for time in sorted(fields["baseline"].keys() & fields[method].keys()):
            grids = [
                pv.read(args.root / "samples" / run / fields[run][time])
                for run in ("baseline", method)
            ]
            if not np.array_equal(grids[0].points, grids[1].points):
                raise ValueError("Equal-time contrasts require identical sampling grids")
            row = dict(method=method, time=time)
            for field in ("velocity", "vorticity"):
                baseline, candidate = [
                    np.asarray(grid.point_data[field], dtype=float) for grid in grids
                ]
                row[field + "_relative_l2"] = float(
                    np.linalg.norm(candidate - baseline) / np.linalg.norm(baseline)
                )
                row[field + "_bitwise_equal"] = bool(np.array_equal(candidate, baseline))
            contrasts.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_root=str(args.root.resolve()),
        runs=runs,
        equal_time_method_contrasts=contrasts,
        limitation="Method differences are not errors against a converged reference. Saved labels used nearest-source inheritance.",
    )
    (args.output / "saved_run_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(runs)} saved runs; wrote {args.output / 'saved_run_audit.json'}")


if __name__ == "__main__":
    main()
