#!/usr/bin/env python3
"""Qualify the same-mesh cube timestep control without claiming spatial convergence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "_cube_temporal_spatial_helpers", CASE_DIR / "postprocess_grid_study.py"
)
_GRID = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GRID
_SPEC.loader.exec_module(_GRID)

CD_TARGET = 0.0025
PROFILE_TARGET = 0.005
AUXILIARY_TARGET = 0.05
FINE_H = 0.045
METRICS = ("mean_drag", "rms_drag", "rms_lift", "rms_side", "strouhal_lift", "strouhal_side")


def _mesh_identity(directory: Path) -> dict:
    path = directory / "fvm" / "mesh.npz"
    digest = hashlib.sha256()
    arrays = []
    with np.load(path, allow_pickle=False) as mesh:
        # Configuration metadata can contain paths and generation timestamps.
        # Hash all actual native arrays, including connectivity and patch data.
        for name in sorted(set(mesh.files) - {"metadata"}):
            value = np.ascontiguousarray(mesh[name])
            digest.update(name.encode())
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
            arrays.append(name)
    if not {"vertex_position", "cell_sizes"}.issubset(arrays):
        raise ValueError("native mesh does not contain vertex_position and cell_sizes")
    return {"path": str(path.resolve()), "array_sha256": digest.hexdigest(), "arrays": arrays}


def _physics_and_time(context: dict) -> tuple[dict, dict]:
    physics = json.loads(json.dumps(context["configuration"]))
    time = physics.pop("time_integration")
    adjustment = dict(time.get("adjustment") or {})
    ceiling = adjustment.pop("maximum_time_step_size", time["time_step_size"])
    # The ceiling and initial dt are the only intentional numerical changes.
    physics["time_adjustment_except_ceiling"] = adjustment
    return physics, {"initial_dt": float(time["time_step_size"]), "ceiling_dt": float(ceiling)}


def _load_case(root: Path, name: str, start: float, end: float) -> dict:
    samples = root / "samples" / name
    registration = json.loads((samples / "grid_run.json").read_text())
    case = _GRID.GridCase(
        name=str(registration["case"]),
        samples_dir=samples,
        wall_cell_size=float(registration["cell_size"]),
        cell_count=int(registration["cell_count"]),
        declared_end_time=float(registration["end_time"]),
    )
    if case.name != name:
        raise ValueError("registration case name differs from its directory")
    context = _GRID._case_context(case, root / "solution")
    if context["warnings"]:
        raise ValueError("; ".join(context["warnings"]))
    if not np.isclose(context["realized_wall_cell_size"], FINE_H, rtol=0, atol=1e-12):
        raise ValueError(f"temporal control requires realized wall h={FINE_H}")
    physics, time = _physics_and_time(context)
    if not np.isfinite(list(time.values())).all() or min(time.values()) <= 0:
        raise ValueError("saved timestep controls must be finite and positive")
    history = _GRID._read_force_history(samples / "forces_history.csv")
    stats = _GRID._force_statistics(history, start, end, context=context)
    series_t, _ = _GRID._window_series(history.time, history.values["drag_coefficient"], start, end)
    if len(series_t) < 32 or np.max(np.diff(series_t)) > (end - start) / 32:
        raise ValueError("force window is too sparsely sampled (need >=32 well-spaced samples)")
    actual_dt = None
    if "accepted_time_step_size" in history.values:
        times, steps = _GRID._window_series(
            history.time, history.values["accepted_time_step_size"], start, end
        )
        if np.any(steps <= 0) or np.any(steps > time["ceiling_dt"] * (1 + 1e-6)):
            raise ValueError(
                "sampled accepted timesteps are nonpositive or exceed the saved ceiling"
            )
        actual_dt = {
            "minimum": float(steps.min()),
            "maximum": float(steps.max()),
            "median": float(np.median(steps)),
            "p90": float(np.quantile(steps, 0.9)),
            "time_weighted_mean": _GRID._time_mean(times, steps),
            "source": "accepted_time_step_size at force-sampling events; not every accepted step",
        }
    mean = max(abs(stats["mean_drag"]), 1e-14)
    block_range = float(np.ptp(stats["drag_block_means"]) / mean)
    halves = [
        _GRID._force_statistics(history, a, b, context=context)
        for a, b in ((start, (start + end) / 2), ((start + end) / 2, end))
    ]
    rms_drift = {
        key: (
            None
            if halves[0][key] is None or halves[1][key] is None
            else abs(halves[1][key] - halves[0][key]) / max(abs(stats[key]), 1e-14)
        )
        for key in ("rms_drag", "rms_lift", "rms_side")
    }
    return {
        "case": name,
        "samples_dir": str(samples.resolve()),
        "physics": physics,
        "configured_time": time,
        "sampled_accepted_dt": actual_dt,
        "mesh": _mesh_identity(root / "solution" / name),
        "realized_wall_h": context["realized_wall_cell_size"],
        "cell_count": context["global_cell_count"],
        "domain_bounds": context["domain_bounds"],
        "mesh_controls": context["mesh_controls"],
        "force_statistics": stats,
        "input_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                samples / "grid_run.json",
                samples / "forces_history.csv",
                root / "solution" / name / "fvm_metadata.json",
            )
        },
        "stationarity": {
            "drag_half_drift": stats["drag_drift_relative"],
            "drag_four_block_range": block_range,
            "target": CD_TARGET,
            "half_window_rms_relative_drift": rms_drift,
            "mean_drag_passes": stats["drag_drift_relative"] is not None
            and stats["drag_drift_relative"] <= CD_TARGET
            and block_range <= CD_TARGET,
            "fluctuation_passes": all(
                value is not None and value <= AUXILIARY_TARGET for value in rms_drift.values()
            ),
            "passes": all(
                value is not None and value <= AUXILIARY_TARGET for value in rms_drift.values()
            )
            and stats["drag_drift_relative"] is not None
            and stats["drag_drift_relative"] <= CD_TARGET
            and block_range <= CD_TARGET,
        },
    }


def _profile_coordinates(path: Path, start: float, end: float) -> np.ndarray:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(row["time"]) for row in rows])
    resets = np.flatnonzero(np.diff(times) < -1e-10)
    begin = int(resets[-1]) + 1 if len(resets) else 0
    selected = [row for row in rows[begin:] if start - 1e-10 <= float(row["time"]) <= end + 1e-10]
    unique_t = np.unique([float(row["time"]) for row in selected])
    if len(unique_t) < 8 or unique_t[0] > start + 1e-10 or unique_t[-1] < end - 1e-10:
        raise ValueError("profile has insufficient samples or does not cover the full window")
    if np.max(np.diff(unique_t)) > (end - start) / 8:
        raise ValueError("profile has a large sampling gap")
    xyz = np.array([[float(row[f"position_{axis}"]) for axis in "xyz"] for row in selected])
    return np.unique(np.round(xyz, 10), axis=0)


def _profiles(first: dict, second: dict, start: float, end: float) -> dict:
    result = {}
    for name in _GRID.PROFILE_NAMES:
        try:
            paths = [Path(case["samples_dir"]) / f"{name}.csv" for case in (first, second)]
            locations = [_profile_coordinates(path, start, end) for path in paths]
            if locations[0].shape != locations[1].shape or not np.array_equal(*locations):
                raise ValueError("the two runs do not use identical physical profile coordinates")
            profiles = [_GRID._read_profile(path, start, end) for path in paths]
            full = _GRID._profile_difference(*profiles)
            wake = _GRID._profile_difference(*profiles, interval=(0.5, 8.0))
            drifts = []
            for path in paths:
                early = _GRID._read_profile(path, start, (start + end) / 2)
                late = _GRID._read_profile(path, (start + end) / 2, end)
                drifts.append(_GRID._profile_difference(early, late)["relative_l2"])
            values = [full["relative_l2"], wake["relative_l2"], *drifts]
            result[name] = {
                "available": True,
                "full": full,
                "wake": wake,
                "half_window_relative_l2_drift": drifts,
                "position": profiles[0].position.tolist(),
                "input_sha256": {
                    str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in paths
                },
                "base_mean_velocity": profiles[0].velocity.tolist(),
                "half_dt_mean_velocity": profiles[1].velocity.tolist(),
                "target": PROFILE_TARGET,
                "passes": all(value is not None and value <= PROFILE_TARGET for value in values),
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            result[name] = {"available": False, "passes": False, "reason": str(error)}
    return result


def analyse_temporal_control(
    campaign_root: Path, output_dir: Path, *, start=60.0, end=120.0
) -> dict:
    """Write qualified/unqualified evidence even when a control is unavailable."""
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValueError("statistics window must be finite, nonnegative, and increasing")
    root = Path(campaign_root)
    report = {
        "schema": "openonda-cube-temporal-control/1",
        "window": [start, end],
        "status": "unqualified",
        "reasons": [],
        "cases": {},
        "metrics": {},
        "profiles": {},
        "targets": {
            "mean_drag_relative": CD_TARGET,
            "profile_relative_l2": PROFILE_TARGET,
            "force_rms_and_strouhal_relative_auxiliary_screen": AUXILIARY_TARGET,
        },
        "scope": "Same-mesh timestep sensitivity only; spatial grid independence is not established.",
    }
    for role, directory, name in (
        ("base", root, "grid_h0045"),
        ("half_dt", root / "temporal", "time_h0045_dt_half"),
    ):
        try:
            report["cases"][role] = _load_case(directory, name, start, end)
        except (OSError, ValueError, TypeError, KeyError) as error:
            report["reasons"].append(f"{role}: missing, incomplete, or invalid evidence: {error}")
    if len(report["cases"]) == 2:
        base, half = report["cases"]["base"], report["cases"]["half_dt"]
        identity_keys = ("physics", "domain_bounds", "mesh_controls", "cell_count")
        changed = [key for key in identity_keys if base[key] != half[key]]
        if base["mesh"]["array_sha256"] != half["mesh"]["array_sha256"]:
            changed.append("realized_mesh_array_sha256")
        report["identity"] = {"matching": not changed, "differences": changed}
        if changed:
            report["reasons"].append("Mesh/physics/solver identity differs: " + ", ".join(changed))
        for key in ("initial_dt", "ceiling_dt"):
            if not np.isclose(half["configured_time"][key] / base["configured_time"][key], 0.5):
                report["reasons"].append(f"Configured {key} is not halved")
        if any(case["sampled_accepted_dt"] is None for case in (base, half)):
            report["reasons"].append("Actual accepted-timestep evidence is missing")
        else:
            ratios = {
                key: half["sampled_accepted_dt"][key] / base["sampled_accepted_dt"][key]
                for key in ("median", "p90", "time_weighted_mean")
            }
            report["sampled_dt_refinement_ratios"] = ratios
            if ratios["p90"] > 0.75 or ratios["time_weighted_mean"] > 0.75:
                report["reasons"].append(
                    "CFL adaptation leaves insufficient observed timestep separation (ratio >0.75)"
                )
        prerequisite_pass = not report["reasons"]
        for role, case in report["cases"].items():
            if not case["stationarity"]["mean_drag_passes"]:
                report["reasons"].append(
                    f"{role}: Cd drift/block variation exceeds the 0.25% temporal signal target"
                )
            if not case["stationarity"]["fluctuation_passes"]:
                report["reasons"].append(f"{role}: force RMS half-window drift exceeds 5%")
        for metric in METRICS:
            a, b = [case["force_statistics"][metric] for case in (base, half)]
            relative = None if a is None or b is None else abs(a - b) / max(abs(b), 1e-14)
            target = CD_TARGET if metric == "mean_drag" else AUXILIARY_TARGET
            passed = relative is not None and relative <= target
            resolutions = None
            if metric.startswith("strouhal"):
                resolutions = [
                    case["force_statistics"][f"{metric}_diagnostic"].get("strouhal_resolution")
                    for case in (base, half)
                ]
                passed = passed and all(
                    value is not None and peak is not None and value <= target * abs(peak)
                    for value, peak in zip(resolutions, (a, b), strict=True)
                )
            report["metrics"][metric] = {
                "base": a,
                "half_dt": b,
                "relative_change": relative,
                "target": target,
                "passes": passed,
                "spectral_resolution": resolutions,
            }
            # Either lift or side can carry a cube's resolved shedding peak.
            if not metric.startswith("strouhal") and not passed:
                report["reasons"].append(
                    f"{metric}: missing or exceeds its declared sensitivity screen"
                )
        spectral = [report["metrics"][key] for key in ("strouhal_lift", "strouhal_side")]
        if not any(item["passes"] for item in spectral):
            report["reasons"].append(
                "No common shedding spectrum resolves and passes the 5% screen"
            )
        if any(item["relative_change"] is not None and not item["passes"] for item in spectral):
            report["reasons"].append(
                "A shedding-frequency comparison exceeds or cannot resolve the 5% screen"
            )
        report["profiles"] = _profiles(base, half, start, end)
        for name, profile in report["profiles"].items():
            if not profile["passes"]:
                report["reasons"].append(
                    f"{name}: missing, drifting, or mean-profile difference exceeds 0.5%"
                )
        primary_pass = (
            prerequisite_pass
            and report["metrics"]["mean_drag"]["passes"]
            and all(case["stationarity"]["mean_drag_passes"] for case in (base, half))
            and all(profile["passes"] for profile in report["profiles"].values())
        )
        report["mean_drag_and_profile_screen"] = {
            "status": "passes_engineering_screen" if primary_pass else "unqualified",
            "scope": "Mean Cd and mean velocity profiles only; excludes force RMS and shedding-frequency qualification.",
        }
        if not report["reasons"]:
            report["status"] = "passes_engineering_temporal_screen"
    report.setdefault(
        "mean_drag_and_profile_screen",
        {"status": "unqualified", "scope": "Missing comparison inputs"},
    )
    report["limitations"] = [
        "Two timestep ceilings cannot establish temporal order or temporal GCI.",
        "Stationarity and spectral screens are engineering checks, not confidence intervals.",
        "Small temporal changes do not establish spatial independence or turbulence-model validity.",
    ]
    report["postprocessor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["shared_helper_sha256"] = hashlib.sha256(Path(_GRID.__file__).read_bytes()).hexdigest()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "temporal_control.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Cube temporal-control comparison",
        "",
        f"Status: **{report['status']}**.",
        "",
        f"Common statistics window: {start:g}–{end:g} s; identical realized h=0.045 m required.",
        "",
        report["scope"],
        "",
        "Separate mean Cd/profile screen: **"
        + report["mean_drag_and_profile_screen"]["status"]
        + "**.",
        "",
        "Targets: Cd mean 0.25%; full/wake mean profiles and profile drift 0.5%; "
        "force RMS and resolved Strouhal 5% auxiliary screen. Cd half/block drift must be ≤0.25%.",
        "",
    ]
    if report["metrics"]:
        lines += [
            "| Metric | Base dt | Half dt | Relative change | Pass |",
            "|---|---:|---:|---:|---|",
        ]
        for name, value in report["metrics"].items():

            def fmt(x):
                return "unresolved" if x is None else f"{x:.7g}"

            lines.append(
                f"| {name} | {fmt(value['base'])} | {fmt(value['half_dt'])} | {fmt(value['relative_change'])} | {value['passes']} |"
            )
        lines.append("")
    lines += [f"- {reason}" for reason in report["reasons"]]
    lines += [
        "",
        *[f"- {limitation}" for limitation in report["limitations"]],
        "",
        "Inspect `temporal_control.json` for realized mesh hashes, saved physics, accepted dt, "
        "force drift/spectral diagnostics, and common mean velocity profiles.",
        "",
    ]
    (destination / "temporal_control.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=CASE_DIR / "campaigns/geometric_r15")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--statistics-start", type=float, default=60)
    parser.add_argument("--statistics-end", type=float, default=120)
    args = parser.parse_args()
    report = analyse_temporal_control(
        args.campaign_root,
        args.output_dir or args.campaign_root / "report",
        start=args.statistics_start,
        end=args.statistics_end,
    )
    print(f"Cube temporal control: {report['status']}")
    for reason in report["reasons"]:
        print(f"  {reason}")


if __name__ == "__main__":
    main()
