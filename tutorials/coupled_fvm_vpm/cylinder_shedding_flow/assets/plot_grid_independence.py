#!/usr/bin/env python3
"""Plot force convergence and measured solver cost for G0, G1, and G2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_CASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_CASE_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CASE_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402


CASE = Path(__file__).resolve().parents[1]
REFERENCE = CASE / "reference_flow"
VERIFICATION = REFERENCE / "solution" / "verification"
GRID_NAMES = ("g0", "g1", "g2")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _performance(path: Path, end_step: int) -> dict[str, float | int]:
    step_seconds = []
    ranks = set()
    previous_step = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            step = int(record["step"])
            if step > end_step:
                continue
            if step != previous_step + 1:
                raise ValueError(
                    f"Non-contiguous performance history in {path}: "
                    f"expected step {previous_step + 1}, found {step}"
                )
            previous_step = step
            step_seconds.append(float(record["step_seconds"]["max"]))
            ranks.add(int(record["n_ranks"]))
    if previous_step != end_step or len(ranks) != 1:
        raise ValueError(f"Incomplete or inconsistent performance history in {path}")
    wall_seconds = float(np.sum(step_seconds))
    n_ranks = ranks.pop()
    return {
        "steps": end_step,
        "mpi_ranks": n_ranks,
        "solver_wall_seconds": wall_seconds,
        "solver_wall_hours": wall_seconds / 3600.0,
        "solver_core_hours": wall_seconds * n_ranks / 3600.0,
        "mean_step_seconds": float(np.mean(step_seconds)),
    }


def _paths(grid: str) -> tuple[Path, Path]:
    if grid == "g1":
        return (
            REFERENCE / "solution" / "benchmark_metadata.json",
            REFERENCE / "solution" / "performance.jsonl",
        )
    return (
        VERIFICATION / f"{grid}_metadata.json",
        VERIFICATION / f"{grid}_performance.jsonl",
    )


def convergence_data(report: dict) -> list[dict]:
    common_end = float(report["common_window"]["end"])
    result = []
    for grid in GRID_NAMES:
        metadata_path, performance_path = _paths(grid)
        metadata = _read_json(metadata_path)
        dt = float(metadata["time"]["fvm_time_step"])
        end_step = int(round(common_end / dt))
        mesh = metadata["mesh"]
        result.append(
            {
                "grid": grid.upper(),
                "surface_h_over_d": float(mesh["surface_cell_size"]),
                "surface_cells_per_d": 1.0 / float(mesh["surface_cell_size"]),
                "first_cell_height_over_d": float(mesh["first_cell_height"]),
                "near_wake_h_over_d": float(mesh["near_wake_cell_size"]),
                "downstream_wake_h_over_d": float(mesh["downstream_wake_cell_size"]),
                "background_h_over_d": float(mesh["background_cell_size"]),
                "cell_count": int(mesh["cell_count"]),
                **_performance(performance_path, end_step),
                **{
                    metric: float(value)
                    for metric, value in report["metrics"][grid].items()
                    if isinstance(value, (int, float))
                },
            }
        )
    return result


def _metric_panel(ax, x: np.ndarray, values: np.ndarray, label: str, limit: float) -> None:
    finest = float(values[-1])
    tolerance = abs(finest) * limit
    ax.axhspan(
        finest - tolerance,
        finest + tolerance,
        color=util.COLORS["background_strong"],
        alpha=0.35,
        linewidth=0.0,
    )
    ax.plot(
        x,
        values,
        color=util.COLORS["reference"],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=util.COLORS["reference"],
        linewidth=1.2,
    )
    ax.set_ylabel(label)
    ax.grid(True, axis="y", alpha=0.3)


def plot(figure_format: str) -> Path:
    report = _read_json(REFERENCE / "solution" / "grid_independence.json")
    rows = convergence_data(report)
    (REFERENCE / "solution" / "grid_costs.json").write_text(
        json.dumps({"schema": 1, "grids": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    x = np.asarray([row["surface_cells_per_d"] for row in rows], dtype=float)
    labels = [f'{row["grid"]}\n{row["surface_cells_per_d"]:g}' for row in rows]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=util.figure_size(15.0),
        dpi=util.FIGURE_DPI,
        sharex=True,
        constrained_layout=True,
    )
    metric_panels = (
        ("strouhal", "$St$"),
        ("mean_cd", r"$\overline{C_D}$"),
        ("cd_peak_to_peak", r"$C_{D,\mathrm{p-p}}$"),
        ("cl_first_harmonic", r"$C_{L,1}$"),
    )
    for ax, (name, label) in zip(axes.flat[:4], metric_panels, strict=True):
        values = np.asarray([row[name] for row in rows], dtype=float)
        _metric_panel(ax, x, values, label, float(report["limits"][name]))

    cost = np.asarray([row["solver_wall_hours"] for row in rows], dtype=float)
    axes[2, 0].plot(
        x,
        cost,
        color=util.COLORS["accent"],
        marker="s",
        markerfacecolor="white",
        markeredgecolor=util.COLORS["accent"],
        linewidth=1.2,
    )
    axes[2, 0].set_ylabel("solver wall time [h]")
    axes[2, 0].grid(True, axis="y", alpha=0.3)

    cells = np.asarray([row["cell_count"] for row in rows], dtype=float)
    axes[2, 1].plot(
        x,
        cells / 1000.0,
        color=util.COLORS["accent"],
        marker="s",
        markerfacecolor="white",
        markeredgecolor=util.COLORS["accent"],
        linewidth=1.2,
    )
    axes[2, 1].set_ylabel("cells [$10^3$]")
    axes[2, 1].grid(True, axis="y", alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("surface resolution $D/h_s$")
    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlim(0.85 * x.min(), 1.05 * x.max())
    fig.suptitle("Cylinder reference: grid convergence and computational cost")
    return util.save(fig, "grid_independence", figure_format, util.FIGURE_DPI)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    print(plot(args.format))


if __name__ == "__main__":
    main()
