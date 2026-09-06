#!/usr/bin/env python3
"""Plot the cylinder grid-study force statistics."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openonda-matplotlib-cache")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CASE_DIR = Path(__file__).resolve().parents[1]


def plot_campaign(output: Path, report: dict) -> None:
    """Always plot available evidence; an empty campaign gets a clear status card."""
    from .study_analysis import METRICS, read_history

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = ("coarse", "medium", "fine")
    colors = {"coarse": "#3569aa", "medium": "#c77919", "fine": "#168070"}
    records = {row["case"]: row for row in report["cases"]}
    subtitle = (
        report.get("evidence_label", "")
        + f"{report['status'].upper()} — selected mesh: {report['selected_mesh'] or 'none'}"
    )

    def save(figure, name):
        for suffix in ("png", "svg"):
            figure.savefig(figures / f"{name}.{suffix}", dpi=170)
        plt.close(figure)

    if not records:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.axis("off")
        import textwrap

        message = "\n\n".join(textwrap.fill(reason, 110) for reason in report["reasons"][:3])
        ax.text(0.02, 0.95, "Cylinder grid study: no qualified statistics", fontsize=16, va="top")
        ax.text(0.02, 0.80, message or "Runs have not completed.", fontsize=10, va="top", wrap=True)
        ax.text(
            0.02,
            0.06,
            "No grid-independence claim. See REPORT.md and individual mesh/flow logs.",
            fontsize=10,
        )
        save(fig, "grid_study")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for axis, metric in zip(axes.flat, METRICS):
            for name in names:
                if name not in records:
                    continue
                row = records[name]
                axis.errorbar(
                    row["mesh_cells"],
                    row[metric],
                    yerr=row["sampling_uncertainty"][metric],
                    fmt="o",
                    color=colors[name],
                    capsize=4,
                    label=f"{name} (D/{round(1 / row['dx'])})",
                )
            rows = [records[name] for name in names if name in records]
            axis.plot(
                [row["mesh_cells"] for row in rows],
                [row[metric] for row in rows],
                color="#888888",
                linewidth=0.7,
            )
            fit = report["grid_convergence"].get(metric, {})
            limit = fit.get("richardson_extrapolated_value")
            if limit is not None:
                axis.axhline(limit, ls=":", color="#777777", label="Estimated Richardson limit")
            axis.set(title=metric.replace("_", " "), xlabel="Cell count", xscale="log")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(subtitle + "\nError bars: cycle-block + force-sampling estimates")
        save(fig, "grid_study")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True, sharex=True)
    plotted = False
    for name in names:
        path = output / "samples" / name / "forces_history.csv"
        try:
            history = read_history(path)
        except (OSError, ValueError):
            continue
        plotted = True
        for axis, key in zip(axes, ("drag_coefficient", "lift_coefficient")):
            axis.plot(history["time"], history[key], color=colors[name], lw=0.6, label=name)
            if name in records:
                selected = records[name]["window"]
                axis.axvspan(selected["start"], selected["end"], color=colors[name], alpha=0.06)
    for axis, ylabel in zip(axes, ("Cd", "Cl")):
        axis.axvline(
            report["config"]["discard_time"],
            color="black",
            ls="--",
            lw=0.8,
            label="Earliest statistics time",
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    if plotted:
        axes[0].legend(fontsize=8, ncol=4)
    else:
        axes[0].text(0.5, 0.5, "No force histories yet", transform=axes[0].transAxes, ha="center")
    axes[-1].set_xlabel("Convective time t U / D")
    fig.suptitle(subtitle + "\nFull histories; faint bands show selected whole-cycle windows")
    save(fig, "force_histories")

    if report["candidate_meshes"]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        terms = (
            "spatial_gci_absolute",
            "temporal_absolute",
            "iterative_absolute",
            "sampling_absolute",
            "domain_absolute",
            "span_absolute",
        )
        for axis, metric in zip(axes.flat, METRICS):
            bottom = np.zeros(2)
            for term in terms:
                values = []
                for name in ("medium", "fine"):
                    row = report["candidate_meshes"][name]["metrics"][metric]
                    values.append(
                        np.nan
                        if row[term] is None or row["allowance"] <= 0
                        else 100 * row[term] / row["allowance"]
                    )
                axis.bar(
                    ["medium", "fine"], values, bottom=bottom, label=term.replace("_absolute", "")
                )
                bottom += np.nan_to_num(values)
            axis.axhline(100, color="black", ls="--", lw=0.8)
            axis.set(title=metric.replace("_", " "), ylabel="% of allowed error budget")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(subtitle + "\nEstimated error contributions; missing estimates do not pass")
        save(fig, "error_budgets")
    else:
        # Replace any old completed-campaign plot when evidence is now missing.
        fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
        axis.axis("off")
        axis.text(0.5, 0.6, subtitle, transform=axis.transAxes, ha="center", fontsize=14)
        axis.text(
            0.5,
            0.4,
            "Insufficient qualified runs for error-budget estimates. See REPORT.md.",
            transform=axis.transAxes,
            ha="center",
            fontsize=11,
        )
        save(fig, "error_budgets")


def main() -> None:
    report = json.loads((CASE_DIR / "solution" / "grid_study.json").read_text(encoding="utf-8"))
    records = report["cases"]
    production_names = set(report["production_cases"])
    production = [record for record in records if record["case"] in production_names]
    dx = np.asarray([record["dx"] for record in records])
    production_dx = np.asarray([record["dx"] for record in production])

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    for axis, metric, label in zip(
        axes,
        ("mean_cd", "cl_rms", "strouhal"),
        (r"$\overline{C_D}$", r"$C_{L,\mathrm{rms}}$", r"$St$"),
        strict=True,
    ):
        values = np.asarray([record[metric] for record in records])
        production_values = np.asarray([record[metric] for record in production])
        axis.plot(dx, values, "o--", color="#9aa4b2", linewidth=1.0, label="all-grid trend")
        axis.plot(
            production_dx,
            production_values,
            "o-",
            color="#1769aa",
            linewidth=1.8,
            label="r=2 production grids",
        )
        convergence = report["grid_convergence"][metric]
        extrapolated = convergence["richardson_extrapolated_value"]
        if extrapolated is not None:
            axis.axhline(
                extrapolated,
                color="#d1495b",
                linewidth=1.0,
                linestyle=":",
                label="Richardson limit",
            )
        gci = convergence["fine_grid_gci_percent"]
        status = (
            f"GCI$_f$={gci:.2f}%"
            if gci is not None
            else convergence["status"].replace("_", " ").capitalize()
        )
        axis.set_title(status, fontsize=9)
        axis.set_xlabel(r"wall spacing $\Delta x/D$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
    axes[0].legend(fontsize=7, loc="best")

    figures = CASE_DIR / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"grid_study.{suffix}", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
