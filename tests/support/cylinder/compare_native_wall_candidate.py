"""Compare a completed native planar wall trial with frozen production forces.

Prepare the production timing window while the isolated candidate is running::

    python -m tests.support.cylinder.compare_native_wall_candidate --prepare-only

After its ordinary native continuation is complete, reuse the returned output::

    python -m tests.support.cylinder.compare_native_wall_candidate --output PATH

No solver is created. Force curves retain their physical accepted times. The
first two seconds after the common 46 s restart are excluded from amplitudes and
timing. Sampling costs are explicit: the candidate writes forces only, whereas
production also writes FVM/VPM profiles and surfaces.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze_force_cycles import FIELDS, complete_cycles, window_statistics
from .freeze_force_comparison import freeze_complete_interval

REPOSITORY = Path(__file__).resolve().parents[3]
CASE = REPOSITORY / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
STUDY = Path(__file__).resolve().parent / "force_boundary_study/20261005T214514Z"
BASELINE = STUDY / "force_baseline_48_58_20261005T224454Z"
CANDIDATE = Path("/tmp/openonda-planar-wall-candidate-cycles-20261006")
PHASES = (
    "vpm",
    "vpm_boundary_condition",
    "fvm",
    "transfer",
    "evolution_total",
    "state_checks_and_samplers",
    "backup",
    "total",
)
LABELS = {
    "reference": "Reference flow",
    "baseline": "Production baseline",
    "candidate": "Native wall candidate",
}
COLOURS = {"reference": "#252525", "baseline": "#27649b", "candidate": "#d18324"}
STYLES = {"reference": "-", "baseline": (0, (5, 2)), "candidate": (0, (2, 2))}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def copy_immutable(source, destination, expected=None):
    before = digest(source)
    if expected is not None and before != expected:
        raise ValueError(f"Frozen source hash differs: {source}")
    shutil.copyfile(source, destination)
    if digest(source) != before or digest(destination) != before:
        raise ValueError(f"Immutable input changed: {source}")
    return {"source": str(source), "snapshot": str(destination), "sha256": before}


def freeze_diagnostics(source, destination, start, end):
    captured_at = datetime.now(UTC).isoformat()
    raw = source.read_bytes()
    complete = raw[: raw.rfind(b"\n") + 1]
    selected = [(line, json.loads(line)) for line in complete.splitlines(keepends=True)]
    selected = [
        (line, value) for line, value in selected if start + 1e-8 < value["time"] <= end + 1e-8
    ]
    records = [value for line, value in selected]
    expected = round((end - start) / 0.04)
    if (
        len(records) != expected
        or abs(records[-1]["time"] - end) > 1e-8
        or any(b["step"] != a["step"] + 1 for a, b in zip(records[:-1], records[1:], strict=True))
    ):
        raise ValueError("Accepted diagnostic records do not completely cover the requested window")
    for record in records:
        if (
            not record["interface_iteration"]["converged"]
            or record["backup_phase"]["status"] == "failed"
        ):
            raise ValueError("Candidate or baseline contains failed interface/backup diagnostics")
        if not all(np.isfinite(record["timing_seconds"][name]) for name in PHASES):
            raise ValueError("Non-finite phase timing")
    content = b"".join(line for line, value in selected)
    destination.write_bytes(content)
    return records, {
        "source": str(source),
        "snapshot": str(destination),
        "sha256": hashlib.sha256(content).hexdigest(),
        "captured_at": captured_at,
        "exchange_count": len(records),
        "interval": [start, end],
        "selection": "start < accepted time <= end, tolerance 1e-8; complete lines from one read of an appending JSONL",
    }


def sampler_scope(fvm_metadata, vpm_metadata):
    return {
        "fvm": [
            {
                "type": item["type"],
                "file_name": item.get("file_name"),
                "schedule": item.get("schedule"),
            }
            for item in fvm_metadata["configuration"]["samplers"]
        ],
        "vpm": [
            {
                "type": item["type"],
                "file_name": item.get("file_name"),
                "schedule": item.get("schedule"),
            }
            for item in vpm_metadata["configuration"]["samplers"]["items"]
        ],
    }


def execution_scope(fvm_metadata, vpm_metadata):
    config = fvm_metadata["configuration"]
    numerics = vpm_metadata["configuration"]["numerics"]
    return {
        "fvm_cores": config["cores"],
        "fvm_execution": config["execution"],
        "fvm_cell_count": fvm_metadata["mesh"]["n_cells"],
        "vpm_configured_device": numerics["compute_device"],
        "vpm_capacity": numerics["max_n_particles"],
        "vpm_precision": numerics["precision"],
        "vpm_induction": numerics["induction"],
    }


def prepare(args, output):
    path = output / "preparation.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data["window"] != [args.start, args.end] or data["candidate_directory"] != str(
            args.candidate.resolve()
        ):
            raise ValueError("Existing comparison has different physical inputs")
        for information in data["sources"].values():
            if digest(information["snapshot"]) != information["sha256"]:
                raise ValueError("Comparison input changed after preparation")
        return data
    snapshots = output / "snapshots"
    snapshots.mkdir()
    baseline = json.loads((args.baseline / "force_comparison.json").read_text())
    if baseline["window"] != [args.start, args.end]:
        raise ValueError("Baseline force window differs")
    sources = {
        "baseline_report": copy_immutable(
            args.baseline / "force_comparison.json", snapshots / "baseline_force_comparison.json"
        )
    }
    for source_label, label in (("coupled", "baseline"), ("reference", "reference")):
        source = baseline["sources"][source_label]
        sources[f"{label}_forces"] = copy_immutable(
            Path(source["snapshot"]),
            snapshots / f"{label}_forces_history.csv",
            source["immutable_snapshot_sha256"],
        )
    _, sources["baseline_diagnostics"] = freeze_diagnostics(
        args.production / "solution/coupler_diagnostics.jsonl",
        snapshots / "baseline_coupler_diagnostics.jsonl",
        args.start,
        args.end,
    )
    metadata = {}
    for label, directory in (("baseline", args.production), ("candidate", args.candidate)):
        metadata[label] = {}
        for solver in ("fvm", "vpm"):
            name = f"{label}_{solver}_metadata"
            sources[name] = copy_immutable(
                directory / f"solution/{solver}_metadata.json", snapshots / f"{name}.json"
            )
            metadata[label][solver] = json.loads(Path(sources[name]["snapshot"]).read_text())
    sources["candidate_inputs"] = copy_immutable(
        args.candidate / "inputs/inputs.json", snapshots / "candidate_inputs.json"
    )
    inputs = json.loads(Path(sources["candidate_inputs"]["snapshot"]).read_text())
    if abs(inputs["initial_time"] - (args.start - 2)) > 1e-8:
        raise ValueError("Candidate restart/settling interval differs from the planned comparison")
    execution = {
        label: execution_scope(values["fvm"], values["vpm"]) for label, values in metadata.items()
    }
    if execution["baseline"] != execution["candidate"]:
        raise ValueError("Candidate and baseline execution configuration differ")
    data = {
        "schema": "openonda-native-wall-candidate-preparation/1",
        "captured_at": datetime.now(UTC).isoformat(),
        "window": [args.start, args.end],
        "restart_time": inputs["initial_time"],
        "settling_seconds_excluded": 2,
        "candidate_directory": str(args.candidate.resolve()),
        "production_directory": str(args.production.resolve()),
        "baseline_directory": str(args.baseline.resolve()),
        "sources": sources,
        "execution": execution,
        "sampling": {
            label: sampler_scope(values["fvm"], values["vpm"]) for label, values in metadata.items()
        },
        "scope": "Native isolated candidate versus immutable actual production forces at identical accepted times; no time shift or solver initialization.",
    }
    write_json(path, data)
    return data


def timing_statistics(records):
    result = {}
    for name in PHASES:
        values = np.asarray([record["timing_seconds"][name] for record in records])
        result[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "sum": float(np.sum(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    particles = [record["n_transfer_particles"] for record in records]
    return {
        "exchange_count": len(records),
        "particles_minimum": min(particles),
        "particles_maximum": max(particles),
        "phase_seconds": result,
    }


def sweep_key(record, *, population=False):
    interface = record["interface_iteration"]
    key = (
        interface["sweeps"],
        interface["picard_sweeps"],
        bool(interface.get("prediction", {}).get("attempted", False)),
        record["n_fvm_substeps"],
    )
    return (*key, record["n_transfer_particles"] // 500) if population else key


def matched_timings(first, second, *, population=False):
    groups = {}
    for label, records in (("baseline", first), ("candidate", second)):
        groups[label] = {}
        for record in records:
            groups[label].setdefault(sweep_key(record, population=population), []).append(record)
    all_keys = sorted(set(groups["baseline"]) | set(groups["candidate"]))
    rows = []
    for key in all_keys:
        values = {
            label: timing_statistics(groups[label][key]) for label in groups if key in groups[label]
        }
        row = {
            "total_interface_evaluations": key[0],
            "picard_sweeps": key[1],
            "prediction_attempted": key[2],
            "fvm_substeps_per_exchange": key[3],
            "statistics": values,
            "both_runs_represented": len(values) == 2,
        }
        if population:
            row["particle_count_bin"] = [key[4] * 500, (key[4] + 1) * 500 - 1]
        if len(values) == 2:
            row["candidate_to_baseline_phase_mean_ratios"] = {
                name: values["candidate"]["phase_seconds"][name]["mean"]
                / values["baseline"]["phase_seconds"][name]["mean"]
                for name in PHASES
            }
        rows.append(row)
    return rows


def read_forces(path):
    return np.atleast_1d(np.genfromtxt(io.BytesIO(path.read_bytes()), delimiter=",", names=True))


def measured_statistics(values, start, end):
    if (
        np.max(abs(2 * values["total_force_x"] - values["drag_coefficient"])) > 1e-11
        or np.max(abs(2 * values["total_force_y"] - values["lift_coefficient"])) > 1e-11
    ):
        raise ValueError("Candidate force normalization differs")
    cycles = {field: complete_cycles(values, field, 2.0) for field in FIELDS}
    result = window_statistics(values, cycles, start, end, 2.0)
    if any(result[field]["complete_cycle_count"] < 1 for field in FIELDS):
        raise ValueError("No complete candidate/reference cycle for a required coefficient")
    return result


def amplitude_ratios(statistics, denominator):
    return {
        label: {
            field: {
                quantity: values[field][quantity] / statistics[denominator][field][quantity]
                for quantity in ("median_peak_to_peak_raw", "median_peak_to_peak_drift_corrected")
            }
            for field in FIELDS
        }
        for label, values in statistics.items()
        if label != denominator
    }


def render(histories, statistics, output, restart, start, end):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(13.5, 10.5))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1, 1, 1.12],
        left=0.065,
        right=0.97,
        top=0.84,
        bottom=0.15,
        hspace=0.56,
        wspace=0.27,
    )
    figure.suptitle(
        "Planar solid-wall correction: native coupled trial",
        x=0.065,
        y=0.975,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    figure.text(
        0.065,
        0.938,
        f"Ordinary native continuation from the identical {restart:g} s state; amplitudes measured only in {start:g}–{end:g} s",
        fontsize=11,
    )
    figure.text(
        0.065,
        0.91,
        "Curves retain physical accepted times. Candidate writes forces only; production also writes FVM/VPM profiles and surfaces.",
        fontsize=10,
        color="#505050",
    )
    axes = []
    for row, field in enumerate(FIELDS):
        axis = figure.add_subplot(grid[row, 0])
        axes.append(axis)
        symbol = "Cd" if row == 0 else "Cl"
        for label, values in histories.items():
            selected = (values["time"] >= restart - 1e-8) & (values["time"] <= end + 1e-8)
            axis.plot(
                values["time"][selected],
                values[field][selected],
                color=COLOURS[label],
                linestyle=STYLES[label],
                linewidth=1.3,
                label=LABELS[label],
            )
        axis.axvspan(restart, start, color="#e7e8eb", alpha=0.7)
        axis.axvline(start, color="#777777", linewidth=0.7, linestyle=":")
        axis.set_xlim(restart, end)
        axis.set_xlabel("Accepted flow time (s)")
        axis.set_ylabel(symbol)
        axis.set_title(f"{symbol}: unshifted force histories", loc="left", fontsize=11)
        axis.grid(color="#e4e6e9", linewidth=0.5)
        trend = figure.add_subplot(grid[row, 1])
        for label, values in statistics.items():
            cycles = values[field]["cycles"]
            trend.plot(
                [cycle["peak_time"] for cycle in cycles],
                [cycle["peak_to_peak_drift_corrected"] for cycle in cycles],
                marker="o",
                markersize=4.5,
                color=COLOURS[label],
                linestyle=STYLES[label],
                linewidth=1.1,
            )
        trend.set_xlim(start, end)
        trend.set_xlabel("Peak time of complete cycle (s)")
        trend.set_ylabel(f"{symbol} peak-to-peak")
        trend.set_title(f"{symbol}: complete-cycle trend, drift-corrected", loc="left", fontsize=11)
        trend.grid(color="#e4e6e9", linewidth=0.5)
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower left",
        bbox_to_anchor=(0.065, 0.858),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=3,
    )
    table_axis = figure.add_subplot(grid[2, :])
    table_axis.axis("off")
    table_axis.set_title(
        f"Complete-cycle medians and ratios · identical {start:g}–{end:g} s window",
        loc="left",
        fontsize=12,
        weight="bold",
        pad=12,
    )
    rows = []
    for label, values in statistics.items():
        rows.append(
            [
                LABELS[label],
                *(
                    f"{values[field][quantity]:.6f}"
                    for field in FIELDS
                    for quantity in (
                        "median_peak_to_peak_raw",
                        "median_peak_to_peak_drift_corrected",
                    )
                ),
                f"{values['drag_coefficient']['complete_cycle_count']} / {values['lift_coefficient']['complete_cycle_count']}",
            ]
        )
    for label in ("baseline", "candidate"):
        rows.append(
            [
                f"{LABELS[label]} / reference",
                *(
                    f"{100 * statistics[label][field][quantity] / statistics['reference'][field][quantity]:.1f}%"
                    for field in FIELDS
                    for quantity in (
                        "median_peak_to_peak_raw",
                        "median_peak_to_peak_drift_corrected",
                    )
                ),
                "—",
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Case",
            "Cd raw",
            "Cd drift-corrected",
            "Cl raw",
            "Cl drift-corrected",
            "Cd / Cl cycles",
        ],
        colWidths=[0.32, 0.11, 0.16, 0.11, 0.16, 0.14],
        cellLoc="center",
        bbox=[0, 0, 1, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d7da")
        cell.set_linewidth(0.5)
        cell.set_facecolor("#f0f1f3" if row == 0 else "white" if row < 4 else "#f6f6f6")
        if row == 0 or row >= 4:
            cell.set_text_props(weight="bold")
        if column == 0:
            cell.set_text_props(ha="left")
    figure.text(
        0.065,
        0.097,
        f"Gray interval {restart:g}–{start:g} s is excluded from all amplitude and timing statistics. Each cycle must be entirely contained in {start:g}–{end:g} s.",
        fontsize=9,
    )
    figure.text(
        0.065,
        0.073,
        "Raw = actual min–max of a complete cycle; drift correction subtracts the interpolated trough baseline. No phase shift is used.",
        fontsize=9,
    )
    figure.text(
        0.065,
        0.049,
        "Timing groups retain actual interface sweeps and sampling scope. A finite continuation window does not establish saturation.",
        fontsize=9,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"native_wall_candidate_comparison.{suffix}", dpi=180, facecolor="white"
        )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--production", type=Path, default=CASE)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=float, default=48)
    parser.add_argument("--end", type=float, default=58)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = (
        args.output.resolve()
        if args.output
        else args.study.resolve()
        / f"native_wall_comparison_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output.mkdir(exist_ok=True)
    preparation = prepare(args, output)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "output": str(output),
                    "comparison_window": [args.start, args.end],
                },
                indent=2,
            )
        )
        return
    trial_path = args.candidate / "continuation_report.json"
    trial_bytes = trial_path.read_bytes()
    trial = json.loads(trial_bytes)
    if (
        trial["status"] != "complete"
        or trial["mode"] != "native"
        or not trial["native_strict_restart_passed"]
        or not trial["numerical_sources_and_original_inputs_unchanged"]
        or trial["boundary_trace_capture_enabled"]
        or trial["predictor_phases"]
        or abs(trial["accepted_time"] - args.end) > 1e-8
    ):
        raise ValueError(
            "Candidate must finish its unchanged, ordinary native continuation before comparison"
        )
    snapshots = output / "snapshots"
    (snapshots / "candidate_continuation_report.json").write_bytes(trial_bytes)
    candidate, candidate_source = freeze_complete_interval(
        args.candidate / "samples/forces_history.csv",
        snapshots / "candidate_forces_history.csv",
        args.end,
    )
    if (
        digest(args.candidate / "samples/forces_history.csv") != trial["force_history_sha256"]
        or abs(trial["final_checkpoint_time"] - args.end) > 1e-8
    ):
        raise ValueError("Candidate final force/checkpoint receipt differs")
    candidate_diagnostics, diagnostic_source = freeze_diagnostics(
        args.candidate / "solution/coupler_diagnostics.jsonl",
        snapshots / "candidate_coupler_diagnostics.jsonl",
        args.start,
        args.end,
    )
    baseline_diagnostics = [
        json.loads(line)
        for line in (snapshots / "baseline_coupler_diagnostics.jsonl").read_bytes().splitlines()
    ]
    histories = {
        "reference": read_forces(snapshots / "reference_forces_history.csv"),
        "baseline": read_forces(snapshots / "baseline_forces_history.csv"),
        "candidate": candidate,
    }
    statistics = {
        label: measured_statistics(values, args.start, args.end)
        for label, values in histories.items()
    }
    render(histories, statistics, output, preparation["restart_time"], args.start, args.end)
    report = {
        "schema": "openonda-native-planar-wall-candidate-comparison/1",
        "created_at": datetime.now(UTC).isoformat(),
        "output": str(output),
        "preparation": preparation,
        "candidate_force_source": candidate_source,
        "candidate_diagnostic_source": diagnostic_source,
        "candidate_continuation_receipt_sha256": hashlib.sha256(trial_bytes).hexdigest(),
        "analysis_source": str(Path(__file__).resolve()),
        "analysis_source_sha256": digest(Path(__file__)),
        "window": [args.start, args.end],
        "complete_cycle_statistics": statistics,
        "relative_to_reference": amplitude_ratios(statistics, "reference"),
        "candidate_relative_to_baseline": amplitude_ratios(statistics, "baseline")["candidate"],
        "timing": {
            "scope": "Accepted exchanges after settling, before final timing row publication; numerical evolution phases exclude sampling and backup. Grouping retains total interface evaluations, Picard sweeps, prediction attempts, FVM substeps and optionally 500-particle bins.",
            "baseline": timing_statistics(baseline_diagnostics),
            "candidate": timing_statistics(candidate_diagnostics),
            "matched_interface_sweeps": matched_timings(
                baseline_diagnostics, candidate_diagnostics
            ),
            "matched_interface_sweeps_and_population": matched_timings(
                baseline_diagnostics, candidate_diagnostics, population=True
            ),
            "execution_configuration_equal": True,
            "sampling_configuration_equal": preparation["sampling"]["baseline"]
            == preparation["sampling"]["candidate"],
        },
        "final_native_checkpoint": {
            "time": trial["final_checkpoint_time"],
            "checkpoint_info_sha256": trial["final_checkpoint_sha256"],
            "native_strict_restart": True,
            "candidate_source_and_input_hash_verification": trial[
                "numerical_sources_and_original_inputs_unchanged"
            ],
        },
        "figure_sha256": {
            suffix: digest(output / f"native_wall_candidate_comparison.{suffix}")
            for suffix in ("png", "pdf", "svg")
        },
        "limitations": [
            "The first 2 s after the identical native restart are excluded; later complete cycles remain a finite transient and do not prove final saturation.",
            "Force curves use physical accepted time, without phase shifting or selecting amplitudes across different windows.",
            "The candidate writes FVM forces only, while production also writes FVM/VPM profiles and surfaces. Total timing differences cannot be assigned solely to the numerical correction.",
            "GPU and CPU are shared with concurrent production, so matching configured execution and particle bins does not remove resource contention.",
            "Only a final native checkpoint hash receipt is reported here; independent native field decoding/audit remains a separate validation.",
        ],
    }
    write_json(output / "candidate_comparison.json", report)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "relative_to_reference": report["relative_to_reference"],
                "candidate_relative_to_baseline": report["candidate_relative_to_baseline"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
