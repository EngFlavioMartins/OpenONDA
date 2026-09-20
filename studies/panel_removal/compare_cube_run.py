"""Compare a clean panel-free trajectory with saved coupled/reference cube runs.

The 4s admission gate requires both t=1..4 and t=2..4: drag relative RMS
and mean changes <=2%, both velocity-profile RMS changes <=2% U_inf, reference
agreement degraded by <=2 percentage points for drag and <=1% U_inf for
profiles. Each FVM snapshot in t=1..4 must also have RMS change <=2% U_inf.
All interface iterations from t=1..4 must converge with complete step coverage. No phase/scale
fit, filtering, or reference-time interpolation is performed.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tutorials/coupled_fvm_vpm/02_cube_flow"
COMPONENTS = ["velocity_x", "velocity_y", "velocity_z"]


def read_table(path):
    """Read a CSV path into a DataFrame, rounding saved times to eight decimals.

    The source file is read only. Missing files or invalid CSV columns raise
    through pandas; no interpolation or duplicate removal is performed."""
    result = pd.read_csv(path)
    result["time"] = result["time"].round(8)
    return result


def time_mean(time, values):
    """Return the trapezoidal time mean of a one-dimensional sampled quantity.

    ``time`` contains increasing physical times in seconds; ``values`` retains
    its input units in the returned float. Fewer than two samples or a
    nonpositive interval raises ValueError. No resampling is performed."""
    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError("A time average requires at least two distinct saved times")
    return float(np.trapezoid(values, time) / (time[-1] - time[0]))


def force_comparison(run, start, end):
    """Compare dimensionless force histories on exact common saved times.

    Parameters
    ----------
    run : pathlib.Path
        Coupled trial directory; historical panel and FVM references are fixed.
    start, end : float
        Inclusive comparison interval in seconds.

    Returns
    -------
    dict
        Trapezoidal means, relative drag errors, raw lift RMS, and exact common
        sample coverage. Relative errors are fractions, not percentages.

    Raises
    ------
    ValueError
        If histories contain duplicate times or lack two finite common samples."""
    paths = [
        run / "samples/forces_history.csv",
        BASELINE / "samples/forces_history.csv",
        BASELINE / "reference_flow/samples/fine/forces_history.csv",
    ]
    frames = []
    for label, path in zip(("trial", "baseline", "reference"), paths, strict=True):
        frame = read_table(path)
        if frame["time"].duplicated().any():
            raise ValueError(f"Ambiguous force history in {path}")
        frames.append(
            frame.set_index("time")[["drag_coefficient", "lift_coefficient"]].add_prefix(
                label + "_"
            )
        )
    aligned = frames[0].join(frames[1], how="inner").join(frames[2], how="inner")
    aligned = aligned.loc[(aligned.index >= start) & (aligned.index <= end)]
    if len(aligned) < 2 or not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Insufficient finite common force samples")
    time = aligned.index.to_numpy()
    values = {
        label: aligned[label + "_drag_coefficient"].to_numpy()
        for label in ("trial", "baseline", "reference")
    }
    means = {label: time_mean(time, value) for label, value in values.items()}
    baseline_norm = np.sqrt(time_mean(time, values["baseline"] ** 2))
    reference_norm = np.sqrt(time_mean(time, values["reference"] ** 2))
    result = {
        "start": float(time[0]),
        "end": float(time[-1]),
        "samples": len(time),
        "expected_samples": len(frames[1].join(frames[2], how="inner").loc[start:end]),
        "time_coverage_complete": list(time)
        == list(frames[1].join(frames[2], how="inner").loc[start:end].index),
        "mean_cd": means,
        "cd_mean_change_relative": abs(means["trial"] - means["baseline"]) / abs(means["baseline"]),
        "cd_rms_change_relative": np.sqrt(
            time_mean(time, (values["trial"] - values["baseline"]) ** 2)
        )
        / baseline_norm,
        "cd_max_change": float(np.max(abs(values["trial"] - values["baseline"]))),
    }
    for label in ("trial", "baseline"):
        result[label + "_cd_reference_relative_rms"] = (
            np.sqrt(time_mean(time, (values[label] - values["reference"]) ** 2)) / reference_norm
        )
        lift = aligned[label + "_lift_coefficient"].to_numpy()
        result[label + "_cl_rms"] = np.sqrt(time_mean(time, lift**2))
    return result


def profile_comparison(run, name, start, end, *, prefix="fvm_", wake_only=False):
    """Compare vector snapshots with trapezoidal spatial and temporal weights.

    Parameters
    ----------
    run : pathlib.Path
        Coupled trial directory containing line samples.
    name : str
        Saved centreline or off-axis line name.
    start, end : float
        Inclusive physical-time interval in seconds.
    prefix : str
        Coupled sampler prefix, normally fvm_ or vpm_.
    wake_only : bool
        Restrict support to x/D greater than 0.5 when true.

    Returns
    -------
    dict
        Snapshot errors, covered lengths and time coverage. RMS errors combine
        all three velocity components, integrated along each fluid segment
        and then over exact saved times. Reference values are interpolated in
        space only, within their fluid support. Uinf/D labels assume these
        unit-speed, unit-diameter cases.

    Raises
    ------
    ValueError
        If coordinates differ, samples are ambiguous, or support is incomplete."""
    trial, baseline, reference = [
        read_table(path)
        for path in (
            run / f"samples/{prefix}{name}.csv",
            BASELINE / f"samples/{prefix}{name}.csv",
            BASELINE / f"reference_flow/samples/fine/{name}.csv",
        )
    ]
    groups = [dict(tuple(frame.groupby("time"))) for frame in (trial, baseline, reference)]
    times = sorted(t for t in set(groups[0]) & set(groups[1]) if start <= t <= end)
    expected_times = sorted(t for t in groups[1] if start <= t <= end)
    metrics = []
    for time in times:
        a, b = [group[time].sort_values("position_x") for group in groups[:2]]
        r = groups[2].get(time)
        if r is not None:
            r = r.sort_values("position_x")
        x = a.position_x.to_numpy()
        if len(x) != len(b) or not np.allclose(x, b.position_x, rtol=0, atol=1e-8):
            raise ValueError("Trial/baseline profile coordinates differ")
        if any(frame.position_x.duplicated().any() for frame in (a, b)) or (
            r is not None and r.position_x.duplicated().any()
        ):
            raise ValueError("Duplicate coordinates within a profile snapshot")
        valid = abs(x) > 0.5 + 1e-8 if name == "centreline" else np.ones(len(x), dtype=bool)
        if wake_only:
            valid &= x > 0.5 + 1e-8
        pa, pb = a[COMPONENTS].to_numpy(), b[COMPONENTS].to_numpy()
        pr = np.full_like(pa, np.nan)
        sides = [x < -0.5, x > 0.5] if name == "centreline" else [np.ones(len(x), dtype=bool)]
        weights, direct_weights = np.zeros(len(x)), np.zeros(len(x))
        for side in sides:
            indices = np.flatnonzero(side & valid)
            if not len(indices):
                continue
            dx = np.diff(x[indices])
            direct_weights[indices[:-1]] += dx / 2
            direct_weights[indices[1:]] += dx / 2
            if r is None:
                continue
            source = r
            if name == "centreline":
                source = (
                    r.loc[r.position_x < -0.5] if x[indices[0]] < 0 else r.loc[r.position_x > 0.5]
                )
            sx = source.position_x.to_numpy()
            if not len(sx):
                raise ValueError("Incomplete reference profile coverage")
            indices = indices[(x[indices] >= sx[0] - 1e-8) & (x[indices] <= sx[-1] + 1e-8)]
            if len(indices) < 2:
                raise ValueError("Incomplete reference profile coverage")
            for j, component in enumerate(COMPONENTS):
                pr[indices, j] = np.interp(x[indices], sx, source[component])
            dx = np.diff(x[indices])
            weights[indices[:-1]] += dx / 2
            weights[indices[1:]] += dx / 2
        row = {
            "time": time,
            "direct_length": float(direct_weights.sum()),
            "reference_length": float(weights.sum()) if r is not None else None,
        }
        pairs = [("change", pa, pb)]
        if r is not None:
            pairs += [("trial_reference", pa, pr), ("baseline_reference", pb, pr)]
        for label, left, right in pairs:
            selected_weights = direct_weights if label == "change" else weights
            selected = selected_weights > 0
            if (
                not np.any(selected)
                or not np.isfinite(left[selected]).all()
                or not np.isfinite(right[selected]).all()
            ):
                raise ValueError("Nonfinite velocity or empty support in fluid profile")
            error = np.linalg.norm(left[selected] - right[selected], axis=1)
            row[label + "_mse"] = float(np.average(error**2, weights=selected_weights[selected]))
            row[label + "_rms_Uinf"] = float(np.sqrt(row[label + "_mse"]))
            row[label + "_max"] = float(np.max(error))
        metrics.append(row)
    if len(metrics) < 2:
        raise ValueError("Insufficient common profile times")
    result = {
        "start": times[0],
        "end": times[-1],
        "samples": len(times),
        "expected_samples": len(expected_times),
        "time_coverage_complete": times == expected_times,
        "missing_times": sorted(set(expected_times) - set(times)),
        "snapshots": metrics,
        "direct_covered_length_D": metrics[-1]["direct_length"],
    }
    reference_rows = [row for row in metrics if row["reference_length"] is not None]
    result["reference_covered_length_D"] = (
        reference_rows[-1]["reference_length"] if reference_rows else None
    )
    result["reference_samples"] = len(reference_rows)
    for label in ("change", "trial_reference", "baseline_reference"):
        selected = [row for row in metrics if label + "_mse" in row]
        if len(selected) < 2:
            raise ValueError("Insufficient exact reference profile times")
        sample_times = np.array([row["time"] for row in selected])
        result[label + "_rms_Uinf"] = np.sqrt(
            time_mean(sample_times, np.array([row[label + "_mse"] for row in selected]))
        )
        result[label + "_max_Uinf"] = max(row[label + "_max"] for row in selected)
        result[label + "_max_snapshot_rms_Uinf"] = max(row[label + "_rms_Uinf"] for row in selected)
    return result


def _differences(left, right, path=""):
    """Return recursively differing mapping paths for the saved identity audit."""
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            item
            for key in sorted(set(left) | set(right))
            for item in _differences(left.get(key), right.get(key), path + "." + key)
        ]
    return [] if left == right else [{"path": path, "baseline": left, "trial": right}]


def identity_audit(run):
    """Audit saved inputs, permitting only explicit experiment/output changes."""
    allowed = {
        ".fvm.cores": "MPI process count; runtime comparisons require a matched allocation",
        ".fvm.time.end_time": "run endpoint",
        ".fvm.time.output_schedule": "visualization output cadence",
        ".vpm.numerics.panel_solver": "panel removal",
        ".vpm.numerics.bodies": "panel removal; solid mask remains owned by FVM geometry",
        ".vpm.run.steps": "run endpoint",
        ".coupler.backup_interval_steps": "checkpoint output cadence",
        ".coupler.interface_iterations": "higher sweep cap with unchanged convergence tolerances",
    }
    try:
        data = []
        for directory in (BASELINE, run):
            solution = directory / "solution"
            fvm = json.loads((solution / "fvm_metadata.json").read_text())["configuration"]
            vpm = json.loads((solution / "vpm_metadata.json").read_text())["configuration"]
            coupler = json.loads((solution / "run_metadata.json").read_text())["coupler"]
            # Removing surface visualization is permitted; keep force/line sampler
            # coordinates, normalization and cadence in the actual identity check.
            fvm = copy.deepcopy(fvm)
            vpm = copy.deepcopy(vpm)
            fvm["samplers"] = [
                s for s in fvm.get("samplers", []) if s.get("type") != "SurfaceSampler"
            ]
            if "samplers" in vpm:
                vpm["samplers"]["items"] = [
                    s for s in vpm["samplers"].get("items", []) if s.get("type") != "SurfaceSampler"
                ]
            # Treat each output schedule as one allowed change, preserving the
            # actual integration settings and sampling schedules separately.
            fvm["time"]["output_schedule"] = json.dumps(
                fvm["time"].get("output_schedule"), sort_keys=True
            )
            mesh_path = solution / "fvm/mesh.npz"
            if not mesh_path.exists():
                mesh_path = solution / "mesh.npz"
            with mesh_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            data.append(({"fvm": fvm, "vpm": vpm, "coupler": coupler}, digest))
        differences = _differences(data[0][0], data[1][0])
        for entry in differences:
            entry["allowed_reason"] = allowed.get(entry["path"])
        unexpected = [entry for entry in differences if entry["allowed_reason"] is None]
        panel_free = data[1][0]["vpm"]["numerics"].get("panel_solver") is None and not data[1][0][
            "vpm"
        ]["numerics"].get("bodies")
        return {
            "available": True,
            "matching_mesh": data[0][1] == data[1][1],
            "mesh_sha256": {"baseline": data[0][1], "trial": data[1][1]},
            "panel_free_saved_configuration": panel_free,
            "differences": differences,
            "unexpected_differences": unexpected,
            "verified": data[0][1] == data[1][1] and not unexpected and panel_free,
            "additional_allowed_change": "Surface visualization samplers omitted; force and line samplers checked exactly",
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"available": False, "verified": False, "reason": str(error)}


def timing_comparison(run, end):
    """Compare equal accepted times; these historical timings are not a benchmark."""
    records = []
    try:
        for directory in (BASELINE, run):
            rows = [
                json.loads(line)
                for line in (directory / "solution/coupler_diagnostics.jsonl")
                .read_text()
                .splitlines()
            ]
            selected = {round(row["time"], 8): row for row in rows if 0 < row["time"] <= end + 1e-8}
            records.append(selected)
        times = sorted(set(records[0]) & set(records[1]))
        result = {
            "available": bool(times),
            "samples": len(times),
            "end": times[-1] if times else None,
        }
        totals = []
        for name, rows in zip(("baseline", "trial"), records, strict=True):
            values = [rows[t] for t in times]
            totals.append(sum(row["timing_seconds"]["total"] for row in values))
            result[name] = {
                "total_seconds": totals[-1],
                "component_seconds": {
                    key: sum(row["timing_seconds"][key] for row in values)
                    for key in ("vpm", "vpm_boundary_condition", "fvm", "transfer")
                },
                "total_interface_sweeps": sum(
                    row["interface_iteration"]["sweeps"] for row in values
                ),
            }
        result["trial_to_baseline_time_ratio"] = totals[1] / totals[0] if totals[0] else None
        result["caveat"] = (
            "Not evidence of speedup: historical hardware load, JIT/cache state, output volume and source revision are uncontrolled. "
            "Extra interface sweeps count as real work; benchmark fresh panel/no-panel runs with matched output and resources."
        )
        return result
    except (OSError, KeyError, ValueError) as error:
        return {"available": False, "reason": str(error)}


def interface_coverage(run, start, end, dt=0.05):
    """Audit ordered accepted interface records against every expected step.

    ``run`` is the trial directory; ``start``, ``end`` and ``dt`` are seconds.
    The result separately reports exact time/step coverage and convergence;
    missing records never count as converged."""
    rows = [
        json.loads(line)
        for line in (run / "solution/coupler_diagnostics.jsonl").read_text().splitlines()
    ]
    rows = [row for row in rows if start - 1e-8 <= row["time"] <= end + 1e-8]
    times = [round(row["time"], 8) for row in rows]
    expected_steps = list(range(round(start / dt), int(np.floor((end + 1e-8) / dt)) + 1))
    expected_times = [round(step * dt, 8) for step in expected_steps]
    steps = [row.get("step") for row in rows]
    return {
        "steps": len(rows),
        "expected_steps": len(expected_steps),
        "complete": times == expected_times and steps == expected_steps,
        "converged": bool(rows) and all(row["interface_iteration"]["converged"] for row in rows),
        "unconverged_steps": sum(not row["interface_iteration"]["converged"] for row in rows),
        "missing_times": sorted(set(expected_times) - set(times)),
    }


def report(run, gate_time=4.0):
    """Build the cube admission report without modifying samples or evidence.

    ``run`` is the trial directory and ``gate_time`` is seconds (default 4).
    The returned mapping distinguishes waiting/partial states from a gate
    result. Admission combines identity, exact coverage, converged interfaces,
    and force/profile agreement over the declared one-to-four and two-to-four
    second windows; a short available interval cannot establish the gate."""
    force_path = run / "samples/forces_history.csv"
    if not force_path.exists():
        return {"status": "waiting_for_samples", "gate_passed": False}
    latest = float(read_table(force_path).time.max())
    end = min(latest, gate_time)
    audit = identity_audit(run)
    if latest < 0.5:
        return {
            "status": "running_before_comparison_window",
            "latest_time": latest,
            "gate_passed": False,
            "identity_audit": audit,
        }
    names = ("centreline", "offaxis_y075")
    full_profiles = {name: profile_comparison(run, name, 0.05, end) for name in names}
    full_forces = force_comparison(run, 0.05, end)
    vpm = {}
    for name in names:
        try:
            vpm[name] = {
                region: profile_comparison(
                    run, name, 0.05, end, prefix="vpm_", wake_only=region == "wake_only"
                )
                for region in ("full_domain", "wake_only")
            }
        except (OSError, ValueError) as error:
            vpm[name] = {"available": False, "reason": str(error)}
    experiment = run / "experiment.json"
    identity = json.loads(experiment.read_text()) if experiment.exists() else {}
    criteria = {
        "four_seconds_reached": latest >= gate_time - 1e-8,
        "verified_panel_free_experiment": identity.get("panel") is False and run != BASELINE,
        "verified_mesh_and_configuration": audit["verified"],
    }
    windows = {}
    for label, start in (("late", 2.0), ("sustained", 1.0)):
        if end < start + 0.25 - 1e-8:
            criteria[label + "_window_available"] = False
            continue
        forces = force_comparison(run, start, end)
        profiles = {name: profile_comparison(run, name, start, end) for name in names}
        windows[label] = {"forces": forces, "profiles": profiles}
        prefix = "" if label == "late" else "sustained_"
        criteria[prefix + "cd_mean_change_below_2pct"] = forces["cd_mean_change_relative"] <= 0.02
        criteria[prefix + "cd_rms_change_below_2pct"] = forces["cd_rms_change_relative"] <= 0.02
        criteria[prefix + "cd_reference_degradation_below_2points"] = (
            forces["trial_cd_reference_relative_rms"]
            <= forces["baseline_cd_reference_relative_rms"] + 0.02
        )
        criteria[prefix + "complete_force_window"] = (
            forces["start"] <= start
            and forces["end"] >= gate_time
            and forces["time_coverage_complete"]
        )
        for name, profile in profiles.items():
            criteria[prefix + name + "_complete_window"] = (
                profile["start"] <= start
                and profile["end"] >= gate_time
                and profile["time_coverage_complete"]
            )
            criteria[prefix + name + "_change_below_2pct_Uinf"] = profile["change_rms_Uinf"] <= 0.02
            criteria[prefix + name + "_reference_degradation_below_1pct_Uinf"] = (
                profile["trial_reference_rms_Uinf"] <= profile["baseline_reference_rms_Uinf"] + 0.01
            )
            if label == "sustained":
                criteria[prefix + name + "_every_snapshot_below_2pct_Uinf"] = (
                    profile["change_max_snapshot_rms_Uinf"] <= 0.02
                )
    sustained_vpm = {}
    if end >= 1.25 - 1e-8:
        for name in names:
            sustained_vpm[name] = {}
            for region in ("full_domain", "wake_only"):
                key = f"sustained_vpm_{name}_{region}"
                try:
                    p = profile_comparison(
                        run, name, 1.0, end, prefix="vpm_", wake_only=region == "wake_only"
                    )
                    sustained_vpm[name][region] = p
                    criteria[key + "_complete_window"] = (
                        p["start"] <= 1.0 and p["end"] >= gate_time and p["time_coverage_complete"]
                    )
                    criteria[key + "_change_below_2pct_Uinf"] = p["change_rms_Uinf"] <= 0.02
                    criteria[key + "_every_snapshot_below_2pct_Uinf"] = (
                        p["change_max_snapshot_rms_Uinf"] <= 0.02
                    )
                    criteria[key + "_reference_degradation_below_1pct_Uinf"] = (
                        p["trial_reference_rms_Uinf"] <= p["baseline_reference_rms_Uinf"] + 0.01
                    )
                except (OSError, ValueError) as error:
                    sustained_vpm[name][region] = {"available": False, "reason": str(error)}
                    criteria[key + "_complete_window"] = False
    else:
        criteria["sustained_vpm_window_available"] = False
    coverage = (
        interface_coverage(run, 1.0, end)
        if end >= 1.0
        else {"steps": 0, "complete": False, "converged": False, "unconverged_steps": 0}
    )
    criteria["interface_converged"] = coverage["converged"]
    criteria["complete_interface_window"] = coverage["complete"] and end >= gate_time
    criteria = {name: bool(value) for name, value in criteria.items()}
    return {
        "trial_directory": str(run),
        "baseline_directory": str(BASELINE),
        "status": "gate_evaluated" if latest >= gate_time else "provisional",
        "latest_time": latest,
        "gate_passed": bool(all(criteria.values())),
        "criteria": criteria,
        "forces": windows.get("late", {}).get("forces"),
        "profiles": windows.get("late", {}).get("profiles"),
        "sustained_window": windows.get("sustained"),
        "full_transient_forces": full_forces,
        "full_transient_profiles": full_profiles,
        "vpm_full_transient_profiles": vpm,
        "vpm_sustained_profiles": sustained_vpm,
        "identity_audit": audit,
        "interface_steps": coverage["steps"],
        "unconverged_steps": coverage["unconverged_steps"],
        "interface_coverage": coverage,
        "historical_timing_comparison": timing_comparison(run, end),
        "limitation": "Early transient admission only; the 0.05..1 s startup is reported separately and may differ materially. Mature statistics, full-run stability and matched runtime benchmarking remain separate.",
    }


def main():
    """Write the requested comparison report and optional PNG/PDF figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plot", type=Path, help="Write unsmoothed force and profile comparison figures"
    )
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    result = report(args.run.resolve())
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    if args.plot:
        plot_comparison(args.run.resolve(), args.plot, figure_format=args.format)
    print(text)


def plot_comparison(run: Path, output: Path, *, figure_format: str = "png") -> None:
    """Export fixed-size force and saved-profile comparisons without smoothing.

    Parameters
    ----------
    run : pathlib.Path
        Coupled cube run containing the saved samples through four seconds.
    output : pathlib.Path
        Main force figure path; profile and wake companions share its stem.
    figure_format : {"png", "pdf"}, default="png"
        Export format, with identical data and physical size in both formats.

    Raises
    ------
    ValueError, RuntimeError
        If source data, the requested format, thesis fonts, or layout are invalid.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from openonda.plotting import (
        CM,
        COLORS,
        DEFAULT_DPI,
        centered_subplots_adjust,
        figure_path,
        fit_thesis_y_label_margins,
        set_thesis_style,
        validate_thesis_figure,
    )

    set_thesis_style()
    output = figure_path(output, figure_format)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = (
        (BASELINE / "reference_flow/samples/fine", "Reference", COLORS["RefGray"], "", "--"),
        (BASELINE / "samples", "With panel", COLORS["TUDcyan"], "fvm_", ":"),
        (run / "samples", "No panel", COLORS["VPMpurple"], "fvm_", "-"),
    )

    def save(figure, axes, suffix):
        """Measure the final labels and export without a subsequent layout pass."""
        for axis in axes:
            axis.grid(True)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        handles, labels = axes[-1].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            columnspacing=1.1,
            handlelength=1.7,
        )
        centered_subplots_adjust(
            figure, outer=0.18, top=0.84, bottom=1.35 * CM / figure.get_figheight(), hspace=0.72
        )
        fit_thesis_y_label_margins(figure, axes)
        validate_thesis_figure(figure, axes)
        figure.savefig(output.with_stem(output.stem + suffix), dpi=DEFAULT_DPI, bbox_inches=None)
        plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, 9.4 * CM))
    for directory, label, color, _prefix, linestyle in sources:
        forces = read_table(directory / "forces_history.csv")
        forces = forces.loc[forces.time <= 4.0]
        for axis, column in zip(axes, ("drag_coefficient", "lift_coefficient"), strict=True):
            axis.plot(forces.time, forces[column], label=label, color=color, linestyle=linestyle)
    axes[0].set(ylabel=r"$C_D$", xlim=(0, 4))
    axes[1].set(xlabel=r"$t$ [s]", ylabel=r"$C_L$", xlim=(0, 4))
    save(figure, axes, "")
    for time in (2.0, 4.0):
        figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, 10.5 * CM))
        for axis, name, location in zip(
            axes, ("centreline", "offaxis_y075"), ("0", "0.75"), strict=True
        ):
            for directory, label, color, prefix, linestyle in sources:
                frame = read_table(directory / f"{prefix}{name}.csv")
                profile = frame.loc[frame.time == time].sort_values("position_x")
                x = profile.position_x.to_numpy()
                u = profile.velocity_x.to_numpy().copy()
                if name == "centreline":
                    u[abs(x) <= 0.5] = np.nan
                axis.plot(x, u, color=color, label=label, linestyle=linestyle)
            axis.set(
                xlim=(-1.5, 1.5),
                ylabel=r"$u_x/U_\infty$",
                title=rf"$y/D={location}$, $t={time:g}$ s",
            )
        axes[-1].set_xlabel(r"$x/D$")
        save(figure, axes, f"_profiles_t{time:g}")

    figure, axes = plt.subplots(3, 1, figsize=(12.5 * CM, 13.5 * CM))
    trial = read_table(run / "samples/forces_history.csv").set_index("time")
    baseline = read_table(BASELINE / "samples/forces_history.csv").set_index("time")
    common = trial.index.intersection(baseline.index)
    common = common[(common >= 1.0) & (common <= 4.0)]
    axes[0].plot(
        common,
        100
        * (trial.loc[common].drag_coefficient - baseline.loc[common].drag_coefficient)
        / baseline.loc[common].drag_coefficient,
        color=COLORS["VPMpurple"],
    )
    axes[0].axhline(0, color=COLORS["RefGray"], lw=0.7, linestyle="--")
    axes[0].set(xlabel=r"$t$ [s]", ylabel=r"$\Delta C_D$ [\%]")
    for axis, name, location in zip(
        axes[1:], ("centreline", "offaxis_y075"), ("0", "0.75"), strict=True
    ):
        for directory, label, color, prefix, linestyle in sources:
            filename = name if not prefix else "vpm_" + name
            frame = read_table(directory / f"{filename}.csv")
            profile = frame.loc[frame.time == 4.0].sort_values("position_x")
            x = profile.position_x.to_numpy()
            u = profile.velocity_x.to_numpy().copy()
            if name == "centreline":
                u[abs(x) <= 0.5] = np.nan
            axis.plot(x, u, color=color, label=label, linestyle=linestyle)
        axis.set(xlabel=r"$x/D$", ylabel=r"$u_x/U_\infty$", title=rf"$y/D={location}$, $t=4$ s")
    save(figure, axes, "_wake")
    output.with_suffix(".md").write_text(
        "Cube comparison: reference is the fully meshed FVM; with/without panel are "
        "the coupled FVM-VPM runs. Histories retain every saved startup sample through "
        "4 s. Profile companions show exact snapshots at 2 and 4 s; solid points are "
        "masked. The wake companion shows the unsmoothed drag change relative to the "
        "panel run from 1 to 4 s and VPM/reference velocity profiles at 4 s. "
        "These figures contain no averaging, time interpolation, or phase alignment. "
        "Qualification is recorded separately in cube_gate.json.\n"
    )


if __name__ == "__main__":
    main()
