#!/usr/bin/env python3
"""Run matched reference/coupled grids, force qualification and sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from openonda.cylinder_campaign import collect_cost, compare_profiles, profile_statistics, run_trial
from openonda.cylinder_case import new_run_directory
from openonda.tutorial_runner import load_case_module

CASE_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).with_name("run_campaign.py")


def plot_results(report: dict, directory: Path, output_format: str = "png") -> None:
    """Save grid metrics with sampling uncertainty and matched span profiles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from openonda.plotting import set_thesis_style

    directory.mkdir(parents=True, exist_ok=True)
    set_thesis_style()
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    for label, grids in (
        ("FVM reference", report["reference"]["grids"]),
        ("FVM–VPM", report["coupled_grids"]),
    ):
        for axis, metric, title in zip(
            axes, ("mean_drag", "rms_lift", "strouhal"), ("Mean drag", "Lift RMS", "Strouhal")
        ):
            axis.errorbar(
                [row["h"] for row in grids],
                [row[metric] for row in grids],
                yerr=[row["uncertainty_95"].get(metric) or 0 for row in grids],
                marker="o",
                capsize=3,
                label=label,
            )
            axis.set(xlabel="h/D", title=title)
            axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("Grid comparison with 95 percent cycle-block sampling intervals")
    figure.savefig(directory / f"grid_comparison.{output_format}", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    for axis, (name, row) in zip(axes, report["span_profiles"].items()):
        for label in ("reference", "coupled"):
            profile = row[label]
            axis.plot([value[0] for value in profile["mean_velocity"]], profile["y"], label=label)
        axis.set(xlabel="Mean u/U", ylabel="y/D", title=name.replace("_", " "))
        axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("Matched x/D=1 mean velocity profiles, tU/D=40–100")
    figure.savefig(directory / f"span_profiles.{output_format}", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CASE_DIR / "study_results" / "cylinder")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=43200, help="seconds per individual case")
    parser.add_argument("--grids", type=float, nargs="+", default=(0.1, 0.08, 0.064))
    parser.add_argument("--sensitivity", choices=("full", "screen", "none"), default="full")
    parser.add_argument(
        "--compute-device", choices=("AUTO", "CPU", "CUDA", "VULKAN", "METAL"), default="CPU"
    )
    parser.add_argument("--coupled-cores", type=int, default=4)
    parser.add_argument("--reference-cores", type=int, default=6)
    args = parser.parse_args()
    if min(args.coupled_cores, args.reference_cores) < 1:
        raise ValueError("coupled-cores and reference-cores must be positive")
    if (
        len(args.grids) < 3
        or any(h <= 0 for h in args.grids)
        or any(a <= b for a, b in zip(args.grids[:-1], args.grids[1:]))
    ):
        raise ValueError("provide at least three positive, strictly decreasing mesh spacings")
    root = (
        args.run_dir.resolve()
        if args.run_dir
        else new_run_directory(args.root.resolve(), "cylinder")
    )
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError(f"use --resume for existing campaign {root}")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    post = load_case_module(CASE_DIR / "reference_flow", "postprocess_grid_study")
    grids = args.grids[:1] if args.pilot else args.grids
    force_grids = []
    report = {
        "schema": "openonda-cylinder-pipeline/2",
        "pilot": args.pilot,
        "per_case_wall_limit_seconds": args.timeout,
        "compute_device": args.compute_device,
        "coupled_cores": args.coupled_cores,
        "reference_cores": args.reference_cores,
        "runs": records,
    }
    for kind in ("reference",) if args.reference_only else ("reference", "coupled"):
        for spacing in grids:
            name = "grid_h" + format(spacing, ".8g").replace(".", "p")
            run_dir = root / kind if kind == "reference" else root / kind / name
            command = [sys.executable, str(LAUNCHER), "--kind", kind, "--run-dir", str(run_dir)]
            if run_dir.exists():
                command.append("--resume")
            if kind == "reference":
                command.extend(("--grid", f"{name}={spacing}", "--no-analysis"))
                command.extend(("--reference-cores", str(args.reference_cores)))
                if args.pilot:
                    command.extend(("--pilot", "--end-time", ".16"))
            else:
                command.extend(
                    (
                        "--override",
                        f"hxy={spacing}",
                        "--override",
                        f"cores={args.coupled_cores}",
                        "--override",
                        f"compute_device={args.compute_device}",
                    )
                )
                if args.pilot:
                    command.extend(("--max-coupling-steps", "3"))
            print(f"{kind} h={spacing:g}: {run_dir}", flush=True)
            record = {
                "kind": kind,
                "name": name,
                "hxy": spacing,
                **run_trial(
                    command, root / "logs" / kind / name, cwd=CASE_DIR, wall_limit=args.timeout
                ),
            }
            cost_root = run_dir / "solution" / name if kind == "reference" else run_dir
            record["cost"] = collect_cost(cost_root)
            records.append(record)
            (root / "pipeline_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
            if record["returncode"]:
                print(
                    f"Case failed or exceeded its budget. See {record['console_log']}", flush=True
                )
                return record["returncode"]
            if kind == "coupled" and not args.pilot:
                histories = list((run_dir / "samples").rglob("forces_history.csv"))
                if len(histories) != 1:
                    raise ValueError(f"expected one coupled force record in {run_dir}")
                force_grids.append(
                    {"name": name, "h": spacing, **post.force_statistics(histories[0], 40, 100)}
                )
        if kind == "reference" and not args.pilot:
            report["reference"] = post.analyse_forces(
                root / "reference" / "samples", root / "reference" / "figures", 40, 100
            )
            # Preserve diagnostics and still compute coupled results when a
            # statistical grid gate is inconclusive; never call it independent.
            report["reference_qualified"] = report["reference"]["force_grid_qualified"]
    if force_grids:
        report["coupled_grids"] = force_grids
        report["coupled_convergence"] = {
            metric: post.richardson_gci(force_grids, metric) for metric in post.METRICS
        }
        drag_convergence = report["coupled_convergence"]["mean_drag"]
        report["coupled_force_grid_qualified"] = bool(
            all(grid["qualified_statistics"] for grid in force_grids)
            and drag_convergence["valid"]
            and drag_convergence["fine_gci"] <= 0.02
            and post.relative_change(force_grids[-2]["rms_lift"], force_grids[-1]["rms_lift"])
            <= 0.05
            and post.relative_change(force_grids[-2]["strouhal"], force_grids[-1]["strouhal"])
            <= 0.02
        )
        fine_reference = report["reference"]["grids"][-1]
        report["coupled_reference_relative_difference"] = {
            metric: post.relative_change(force_grids[-1][metric], fine_reference[metric])
            for metric in ("mean_drag", "rms_lift", "strouhal")
        }
        fine_coupled = root / "coupled" / force_grids[-1]["name"] / "samples"
        fine_reference_directory = root / "reference" / "samples" / fine_reference["name"]
        report["span_profiles"] = {}
        for name in ("span_lower", "span_middle", "span_upper"):
            matches = list(fine_coupled.rglob(name + ".csv"))
            if len(matches) != 1:
                raise ValueError(f"expected one {name} profile in {fine_coupled}")
            candidate = profile_statistics(matches[0])
            reference = profile_statistics(fine_reference_directory / (name + ".csv"))
            report["span_profiles"][name] = {
                "coupled": candidate,
                "reference": reference,
                "errors": compare_profiles(candidate, reference),
            }
        differences = report["coupled_reference_relative_difference"]
        report["accuracy_targets_met"] = bool(
            report["reference_qualified"]
            and report["coupled_force_grid_qualified"]
            and not any(
                row["cost"]["unconverged_stationary_intervals"]
                for row in records
                if row["kind"] == "coupled"
            )
            and all(grid["qualified_statistics"] for grid in force_grids)
            and differences["mean_drag"] <= 0.02
            and differences["rms_lift"] <= 0.05
            and differences["strouhal"] <= 0.02
            and all(
                row["errors"]["mean_velocity_l2"] <= 0.03
                for row in report["span_profiles"].values()
            )
        )
        report["qualification_scope"] = (
            "These force/profile targets do not establish temporal, domain or span independence; inspect their separate sensitivity results."
        )
        plot_results(report, root / "figures")
    (root / "pipeline_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.pilot and not args.reference_only and args.sensitivity != "none":
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_sensitivity.py")),
            "--run-dir",
            str(root / "sensitivity"),
            "--timeout",
            str(args.timeout),
            "--compute-device",
            args.compute_device,
            "--coupled-cores",
            str(args.coupled_cores),
        ]
        if args.sensitivity == "screen":
            command.append("--screen")
        if args.resume:
            command.append("--resume")
        # Each sensitivity worker enforces its own per-case limit. This driver
        # must not apply the 12-hour limit to the entire multi-case campaign.
        import subprocess

        report["sensitivity_returncode"] = subprocess.call(command, cwd=CASE_DIR)
        (root / "pipeline_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(root, flush=True)
    return (
        0
        if args.pilot
        or (
            (
                report.get("reference_qualified", False)
                if args.reference_only
                else report.get("accuracy_targets_met", False)
            )
            and not report.get("sensitivity_returncode", 0)
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
