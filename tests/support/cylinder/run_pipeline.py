#!/usr/bin/env python3
"""Run matched reference/coupled grids, force qualification and sensitivity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from openonda.cylinder_campaign import collect_cost, compare_profiles, profile_statistics, run_trial
from openonda.cylinder_case import new_run_directory
from openonda.tutorial_runner import load_case_module

CASE_DIR = (
    Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
)
LAUNCHER = Path(__file__).with_name("run_campaign.py")
_DOMAIN_KEYS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def recorded_fvm_domain(run_dir: Path) -> dict | None:
    """Read realized mesh bounds, never the requested mesher domain."""
    try:
        metadata = json.loads((run_dir / "solution/run_metadata.json").read_text())
        domain = metadata["fvm_solver"]["fvm_domain"]
        bounds = [float(domain[key]) for key in _DOMAIN_KEYS]
        if not all(math.isfinite(value) for value in bounds) or any(
            bounds[index] >= bounds[index + 1] for index in (0, 2, 4)
        ):
            return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return dict(zip(_DOMAIN_KEYS, bounds, strict=True))


def coupled_grid_domain_check(grids: list[dict]) -> dict:
    """Require one recorded physical domain for grid-only GCI qualification."""
    missing = [grid["name"] for grid in grids if grid["resolved_fvm_domain"] is None]
    if missing:
        return {
            "valid": False,
            "reason": "missing or invalid recorded FVM domain: " + ", ".join(missing),
        }
    reference = grids[0]["resolved_fvm_domain"]
    if any(
        not math.isclose(grid["resolved_fvm_domain"][key], reference[key], rel_tol=0, abs_tol=1e-10)
        for grid in grids[1:]
        for key in _DOMAIN_KEYS
    ):
        return {
            "valid": False,
            "reason": "resolved FVM domains differ; comparison includes domain sensitivity",
        }
    return {"valid": True, "reason": "recorded resolved FVM domains match"}


def plot_results(report: dict, directory: Path, output_format: str = "both") -> None:
    """Save grid metrics with sampling uncertainty and matched span profiles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from openonda.plotting import (
        COLORS,
        centered_subplots_adjust,
        export_figure,
        figure_size,
        fit_thesis_y_label_margins,
        prepare_figure,
        set_thesis_style,
        validate_thesis_figure,
    )

    directory.mkdir(parents=True, exist_ok=True)
    set_thesis_style()
    figure, axes = plt.subplots(3, 1, figsize=figure_size("stacked"))
    for label, grids, color in (
        ("FVM reference", report["reference"]["grids"], COLORS["reference"]),
        ("FVM–VPM", report["coupled_grids"], COLORS["hybrid"]),
    ):
        for axis, metric, title in zip(
            axes,
            ("mean_drag", "rms_lift", "strouhal"),
            ("Mean drag", "Lift RMS", "Strouhal"),
            strict=True,
        ):
            axis.errorbar(
                [row["h"] for row in grids],
                [row[metric] for row in grids],
                yerr=[row["uncertainty_95"].get(metric) or 0 for row in grids],
                marker="o",
                capsize=3,
                label=label,
                color=color,
                linestyle="--" if "reference" in label else "-",
            )
            axis.set(xlabel="h/D", title=title)
            axis.margins(x=0.1)
            axis.grid(False)
    axes[0].legend()
    print("Grid comparison: 95% cycle-block sampling intervals")
    centered_subplots_adjust(figure, outer=0.15, bottom=0.12, top=0.94, hspace=0.85)
    prepare_figure(figure)
    fit_thesis_y_label_margins(figure, axes)
    validate_thesis_figure(figure, axes)
    export_figure(figure, directory / "grid_comparison", figure_format=output_format)
    plt.close(figure)
    figure, axes = plt.subplots(3, 1, figsize=figure_size("stacked"))
    for axis, (name, row) in zip(axes, report["span_profiles"].items(), strict=True):
        for label, color in (("reference", COLORS["reference"]), ("coupled", COLORS["hybrid"])):
            profile = row[label]
            axis.plot(
                [value[0] for value in profile["mean_velocity"]],
                profile["y"],
                label=label,
                color=color,
                linestyle="--" if "reference" in label else "-",
            )
        axis.set(xlabel="Mean u/U", ylabel="y/D", title=name.replace("_", " "))
        axis.grid(False)
    axes[0].legend()
    print("Mean velocity at x/D=1; tU/D=40–100")
    centered_subplots_adjust(figure, outer=0.15, bottom=0.12, top=0.94, hspace=0.85)
    prepare_figure(figure)
    fit_thesis_y_label_margins(figure, axes)
    validate_thesis_figure(figure, axes)
    export_figure(figure, directory / "span_profiles", figure_format=output_format)
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
        or any(a <= b for a, b in zip(args.grids[:-1], args.grids[1:], strict=True))
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
    post = load_case_module(Path(__file__).resolve().parent, "postprocess_grid_study")
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
                    {
                        "name": name,
                        "h": spacing,
                        "resolved_fvm_domain": recorded_fvm_domain(run_dir),
                        **post.force_statistics(histories[0], 40, 100),
                    }
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
        domain_check = coupled_grid_domain_check(force_grids)
        report["coupled_grid_domain_check"] = domain_check
        report["coupled_convergence"] = {
            metric: post.richardson_gci(force_grids, metric)
            if domain_check["valid"]
            else {
                "valid": False,
                "order": None,
                "extrapolated": None,
                "fine_gci": None,
                "reason": domain_check["reason"],
            }
            for metric in post.METRICS
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
        # Sensitivity workers enforce the per-case time budget.
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
