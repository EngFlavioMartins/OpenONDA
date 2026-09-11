#!/usr/bin/env python3
"""Regenerate thesis-ready figures from the accepted, frozen result tables."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "openonda-vpm-time-integration-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "openonda-vpm-time-integration-xdg"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
from style import METHOD_STYLE, PALETTE, THESIS_WIDTH, apply_style, finish_layout, save_all

ROOT = Path(__file__).resolve().parents[1]
FIXED = ROOT / "data" / "fixed_core"
COMPARATOR = ROOT / "data" / "comparator"
FIGURES = ROOT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_fixed_core_temporal(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(THESIS_WIDTH, 3.0), sharey=True)
    cases = (("regular", "regular"), ("cluster_void", "cluster / void"))
    methods = ("coupled_ssprk3", "historical_lie", "symmetric_strang")
    for axis, (cloud, title) in zip(axes, cases, strict=True):
        for method in methods:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["cloud"] == cloud
                    and float(row["sigma_over_ell"]) == 0.6
                    and row["method"] == method
                ),
                key=lambda row: float(row["dt"]),
            )
            style = METHOD_STYLE[method]
            axis.loglog(
                [float(row["dt"]) for row in selected],
                [float(row["scaled_rms_error"]) for row in selected],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
            )
        axis.set_title(f"{title}, $\\sigma/h=0.6$")
        axis.set_xlabel("time step $\\Delta t$")
        axis.grid(True, which="both")
        axis.set_xlim(0.0055, 0.057)
        axis.xaxis.set_major_locator(mticker.FixedLocator([0.00625, 0.0125, 0.025, 0.05]))
        axis.xaxis.set_major_formatter(mticker.FixedFormatter([".00625", ".0125", ".025", ".05"]))
        axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[0].set_ylabel("scaled full-state RMS error")
    axes[0].set_ylim(1.5e-12, 3.0e-4)
    axes[0].yaxis.set_major_locator(mticker.FixedLocator([10.0**power for power in range(-11, -3)]))
    axes[0].yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        columnspacing=0.8,
        handlelength=1.6,
        handletextpad=0.4,
    )
    finish_layout(fig, axes, rect=(0.0, 0.0, 1.0, 0.86), w_pad=1.2)
    save_all(fig, FIGURES / "fixed_core_temporal_convergence")


def plot_order_summary(rows: list[dict[str, str]]) -> None:
    methods = ("historical_lie", "symmetric_strang", "coupled_ssprk3")
    expected = {"historical_lie": 1.0, "symmetric_strang": 2.0, "coupled_ssprk3": 3.0}
    fig, axis = plt.subplots(figsize=(THESIS_WIDTH, 2.7))
    rng = np.random.default_rng(711)
    for index, method in enumerate(methods):
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            if row["method"] == method:
                groups.setdefault((row["cloud"], row["sigma_over_ell"]), []).append(row)
        orders = []
        for group in groups.values():
            finest = min(group, key=lambda row: float(row["dt"]))
            orders.append(float(finest["observed_order_from_previous"]))
        style = METHOD_STYLE[method]
        jitter = rng.uniform(-0.08, 0.08, len(orders))
        axis.scatter(
            index + jitter,
            orders,
            s=21,
            facecolor="white",
            edgecolor=style["color"],
            marker=style["marker"],
            linewidth=0.9,
            zorder=3,
        )
        axis.plot(
            [index - 0.22, index + 0.22],
            [np.median(orders), np.median(orders)],
            color=style["color"],
            linewidth=2.1,
        )
        axis.plot(
            [index - 0.30, index + 0.30],
            [expected[method], expected[method]],
            color=PALETTE["gray"],
            linestyle=":",
            linewidth=0.9,
        )
    axis.set_xticks(range(3), [METHOD_STYLE[method]["label"] for method in methods])
    axis.set_ylabel("finest-pair observed order")
    axis.set_ylim(0.88, 3.12)
    axis.grid(True, axis="y")
    axis.set_title("Observed order across 12 cases")
    finish_layout(fig, axis)
    save_all(fig, FIGURES / "fixed_core_order_summary")


def plot_geometry(rows: list[dict[str, str]]) -> None:
    cloud_style = {
        "regular": (PALETTE["dark"], "o", "regular"),
        "jitter_a": (PALETTE["purple"], "s", "jitter A"),
        "jitter_b": (PALETTE["teal"], "^", "jitter B"),
        "cluster_void": (PALETTE["orange"], "D", "cluster / void"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(THESIS_WIDTH, 3.0))
    for cloud, (color, marker, label) in cloud_style.items():
        selected = sorted(
            (row for row in rows if row["cloud"] == cloud),
            key=lambda row: float(row["sigma_over_volume_spacing"]),
        )
        x = [float(row["sigma_over_volume_spacing"]) for row in selected]
        axes[0].plot(
            x,
            [float(row["zeroth_moment_abs_rms"]) for row in selected],
            color=color,
            marker=marker,
            label=label,
        )
        axes[1].plot(
            x,
            [float(row["first_moment_norm_rms"]) for row in selected],
            color=color,
            marker=marker,
            label=label,
        )
    axes[0].set_ylabel(r"zeroth-moment RMS $|m_0|$")
    axes[1].set_ylabel(r"first-moment RMS $\|m_1\|$")
    for axis in axes:
        axis.set_xlabel(r"core ratio $\sigma/h$")
        axis.set_yscale("log")
        axis.grid(True, which="both")
        axis.set_xticks([0.6, 1.0, 1.5])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        columnspacing=0.8,
        handlelength=1.6,
        handletextpad=0.4,
    )
    finish_layout(fig, axes, rect=(0.0, 0.0, 1.0, 0.84), w_pad=0.8)
    save_all(fig, FIGURES / "geometry_moment_defects")


def plot_tangent(rows: list[dict[str, str]]) -> None:
    cases = (
        ("regular", 0.6, "regular\n0.6"),
        ("regular", 1.5, "regular\n1.5"),
        ("cluster_void", 0.6, "cluster/void\n0.6"),
        ("cluster_void", 1.5, "cluster/void\n1.5"),
    )
    methods = ("coupled_ssprk3", "historical_lie", "symmetric_strang")
    fig, axis = plt.subplots(figsize=(THESIS_WIDTH, 3.0))
    centers = np.arange(len(cases), dtype=float)
    offsets = (-0.16, 0.0, 0.16)
    for offset, method in zip(offsets, methods, strict=True):
        values = []
        for cloud, sigma, _ in cases:
            row = next(
                item
                for item in rows
                if item["cloud"] == cloud
                and float(item["sigma_over_ell"]) == sigma
                and item["method"] == method
            )
            values.append(1.0e6 * (float(row["numerical_over_refined_tangent_estimate"]) - 1.0))
        style = METHOD_STYLE[method]
        axis.scatter(
            centers + offset,
            values,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            s=24,
            zorder=3,
        )
    axis.axhline(0.0, color=PALETTE["dark"], linewidth=0.7)
    axis.set_yscale("symlog", linthresh=0.001, linscale=0.7)
    axis.set_xticks(centers, [case[2] for case in cases])
    axis.set_xlabel("cloud and $\\sigma/h$")
    axis.set_ylabel("tangent excess (ppm; symlog)")
    axis.set_ylim(-3.0, 100.0)
    axis.yaxis.set_major_locator(mticker.FixedLocator([-1.0, 0.0, 1.0, 10.0, 100.0]))
    axis.yaxis.set_major_formatter(mticker.FixedFormatter(["-1", "0", "1", "10", "100"]))
    axis.yaxis.set_minor_formatter(mticker.NullFormatter())
    axis.grid(True, axis="y", which="both")
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        columnspacing=0.8,
        handlelength=1.6,
        handletextpad=0.4,
    )
    finish_layout(fig, axis, rect=(0.0, 0.0, 1.0, 0.86))
    save_all(fig, FIGURES / "tangent_excess")


def plot_comparator(rows: list[dict]) -> None:
    cases = (
        ("oscillator", "exact linear oscillator"),
        ("growing", "exact linear growing mode"),
        ("two_particle_fixed_core", "synthetic two-particle field"),
    )
    methods = ("ssp3_coupled", "historical_lie", "strang_rk4")
    fig, axes = plt.subplots(3, 1, figsize=(THESIS_WIDTH, 4.7), sharey=False)
    for axis, (system, title) in zip(axes, cases, strict=True):
        for method in methods:
            selected = sorted(
                (row for row in rows if row["system"] == system and row["method"] == method),
                key=lambda row: float(row["dt"]),
            )
            style = METHOD_STYLE[method]
            axis.loglog(
                [float(row["dt"]) for row in selected],
                [float(row["relative_error"]) for row in selected],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
            )
        axis.set_title(title)
        axis.set_xlabel("$\\Delta t$")
        axis.grid(True, which="both")
        ticks = sorted({float(row["dt"]) for row in rows if row["system"] == system})
        labels = [f"{value:g}" for value in ticks]
        axis.set_xlim(ticks[0] / 1.35, ticks[-1] * 1.35)
        axis.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        axis.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
        axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    for axis in axes:
        axis.set_ylabel("relative error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        columnspacing=0.8,
        handlelength=1.6,
        handletextpad=0.4,
    )
    finish_layout(fig, axes, rect=(0.0, 0.0, 1.0, 0.89), h_pad=0.6)
    save_all(fig, FIGURES / "comparator_convergence")


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def plot_oscillator_stability() -> None:
    q_values = np.linspace(0.0, 2.4, 481)
    coupled = np.sqrt((1.0 - 0.5 * q_values**2) ** 2 + (q_values - q_values**3 / 6.0) ** 2)
    lie = np.array([spectral_radius(np.array([[1.0, q], [-q, 1.0 - q * q]])) for q in q_values])
    strang = np.array(
        [
            spectral_radius(
                np.array([[1.0 - 0.5 * q * q, q * (1.0 - 0.25 * q * q)], [-q, 1.0 - 0.5 * q * q]])
            )
            for q in q_values
        ]
    )
    fig, axis = plt.subplots(figsize=(THESIS_WIDTH, 2.45))
    axis.plot(q_values, coupled, color=PALETTE["purple"], label="Coupled SSPRK3")
    axis.plot(q_values, lie, color=PALETTE["orange"], linestyle="--", label="Lie exact subflows")
    axis.plot(
        q_values, strang, color=PALETTE["teal"], linestyle="-.", label="Strang exact subflows"
    )
    axis.axhline(1.0, color=PALETTE["dark"], linewidth=0.7)
    axis.axvline(math.sqrt(3.0), color=PALETTE["purple"], linestyle=":", linewidth=0.9)
    axis.axvline(2.0, color=PALETTE["gray"], linestyle=":", linewidth=0.9)
    axis.text(
        math.sqrt(3.0) - 0.03, 2.55, r"$\sqrt{3}$", ha="right", va="top", color=PALETTE["purple"]
    )
    axis.text(2.03, 2.55, "$2$", ha="left", va="top", color=PALETTE["gray"])
    axis.set_xlim(0.0, 2.4)
    axis.set_ylim(0.92, 2.65)
    axis.set_xlabel(r"oscillator parameter $q=\omega\Delta t$")
    axis.set_ylabel("one-step spectral radius")
    axis.set_title("Oscillator stability counterexample")
    axis.grid(True)
    axis.legend(loc="upper left")
    finish_layout(fig, axis)
    save_all(fig, FIGURES / "oscillator_stability")


def main() -> None:
    apply_style()
    fixed_temporal = read_csv(FIXED / "temporal.csv")
    fixed_geometry = read_csv(FIXED / "geometry.csv")
    fixed_tangent = read_csv(FIXED / "tangent.csv")
    comparator_rows = json.loads((COMPARATOR / "results.json").read_text())["rows"]
    plot_fixed_core_temporal(fixed_temporal)
    plot_order_summary(fixed_temporal)
    plot_geometry(fixed_geometry)
    plot_tangent(fixed_tangent)
    plot_comparator(comparator_rows)
    plot_oscillator_stability()
    print(f"wrote 18 figure files under {FIGURES}")


if __name__ == "__main__":
    main()
