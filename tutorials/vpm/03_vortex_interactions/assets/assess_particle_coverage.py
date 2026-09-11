#!/usr/bin/env python3
"""Audit particle coverage in saved seeded vortex-ring runs.

The audit is deliberately offline: it reads native HDF5 backups and sampler
CSVs without reconstructing a new flow field or changing solver state.  Local
coverage is measured within each original ring group so interpenetration of
the two clouds cannot make an under-resolved filament look well sampled.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


BACKUP_PATTERN = re.compile(r"vpm_(\d+)\.h5$")
N_NEIGHBOURS = 6
N_AZIMUTH_BINS = 64


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        return np.full(quantiles.shape, np.nan)
    order = np.argsort(values[usable])
    selected = values[usable][order]
    selected_weights = weights[usable][order]
    probability = (np.cumsum(selected_weights) - 0.5 * selected_weights) / np.sum(selected_weights)
    return np.interp(quantiles, probability, selected, left=selected[0], right=selected[-1])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        return float("nan")
    return float(np.average(values[usable], weights=weights[usable]))


def _misalignment_degrees(strength: np.ndarray, vorticity: np.ndarray) -> np.ndarray:
    strength_norm = np.linalg.norm(strength, axis=1)
    vorticity_norm = np.linalg.norm(vorticity, axis=1)
    denominator = strength_norm * vorticity_norm
    cosine = np.divide(
        np.einsum("ij,ij->i", strength, vorticity),
        denominator,
        out=np.full(len(strength), np.nan),
        where=denominator > np.finfo(float).tiny,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _same_group_spacing(position: np.ndarray, group_id: np.ndarray) -> np.ndarray:
    spacing = np.full(len(position), np.nan)
    for group in np.unique(group_id):
        selected = np.flatnonzero(group_id == group)
        if len(selected) <= N_NEIGHBOURS:
            continue
        tree = cKDTree(position[selected], compact_nodes=False)
        distance, _ = tree.query(position[selected], k=N_NEIGHBOURS + 1, workers=-1)
        spacing[selected] = distance[:, 1:].mean(axis=1)
    return spacing


def _cross_group_rho(
    position: np.ndarray, core_radius: np.ndarray, group_id: np.ndarray
) -> np.ndarray:
    rho = np.full(len(position), np.nan)
    groups = np.unique(group_id)
    if len(groups) != 2:
        return rho
    for group, other in ((groups[0], groups[1]), (groups[1], groups[0])):
        selected = np.flatnonzero(group_id == group)
        other_selected = np.flatnonzero(group_id == other)
        tree = cKDTree(position[other_selected], compact_nodes=False)
        distance, local_nearest = tree.query(position[selected], k=1, workers=-1)
        nearest = other_selected[local_nearest]
        pair_radius = 0.5 * (core_radius[selected] + core_radius[nearest])
        rho[selected] = distance / pair_radius
    return rho


def _azimuth_metrics(
    position: np.ndarray, group_id: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    coverage = []
    minimum_effective_count = []
    for group in np.unique(group_id):
        selected = group_id == group
        group_position = position[selected]
        group_weights = weights[selected]
        centroid_y = np.average(group_position[:, 1], weights=group_weights)
        centroid_z = np.average(group_position[:, 2], weights=group_weights)
        angle = np.mod(
            np.arctan2(group_position[:, 2] - centroid_z, group_position[:, 1] - centroid_y),
            2.0 * np.pi,
        )
        bin_id = np.minimum(
            (angle / (2.0 * np.pi) * N_AZIMUTH_BINS).astype(int), N_AZIMUTH_BINS - 1
        )
        bin_weight = np.bincount(bin_id, weights=group_weights, minlength=N_AZIMUTH_BINS)
        bin_weight_sq = np.bincount(
            bin_id, weights=group_weights * group_weights, minlength=N_AZIMUTH_BINS
        )
        effective = np.divide(
            bin_weight * bin_weight,
            bin_weight_sq,
            out=np.zeros_like(bin_weight),
            where=bin_weight_sq > 0.0,
        )
        coverage.append(float(np.count_nonzero(bin_weight) / N_AZIMUTH_BINS))
        minimum_effective_count.append(float(np.min(effective)))
    return float(min(coverage)), float(min(minimum_effective_count))


def _snapshot_metrics(path: Path, run: str) -> tuple[dict[str, float | int | str], list[dict]]:
    with h5py.File(path, "r") as handle:
        particles = handle["particles"]
        position = particles["position"][:].astype(np.float64)
        strength = particles["vortex_strength"][:].astype(np.float64)
        vorticity = particles["vorticity"][:].astype(np.float64)
        core_radius = particles["core_radius"][:].astype(np.float64)
        group_id = particles["group_id"][:].astype(np.int64)
        step = int(handle["solver"].attrs["step"])
        time = float(handle["solver"].attrs["time"])

    weight = np.linalg.norm(strength, axis=1)
    spacing = _same_group_spacing(position, group_id)
    overlap = spacing / core_radius
    angle = _misalignment_degrees(strength, vorticity)
    # The pair radius used by particle-to-particle induction is the mean of
    # target and source radii.  This nearest-pair metric asks only whether the
    # two represented vorticity supports have begun to overlap; velocity
    # induction itself is nonlocal.
    cross_rho = _cross_group_rho(position, core_radius, group_id)
    q50, q90, q95, q99 = _weighted_quantiles(overlap, weight, (0.50, 0.90, 0.95, 0.99))
    a50, a90, a95 = _weighted_quantiles(angle, weight, (0.50, 0.90, 0.95))
    c01, c50 = _weighted_quantiles(cross_rho, weight, (0.01, 0.50))
    strong = weight >= 0.01 * np.max(weight)
    finite = strong & np.isfinite(overlap) & np.isfinite(angle)
    correlation = (
        float(spearmanr(overlap[finite], angle[finite]).statistic)
        if np.sum(finite) >= 3
        else np.nan
    )
    high_angle = finite & (angle >= _weighted_quantiles(angle[finite], weight[finite], (0.90,))[0])
    other_angle = finite & ~high_angle
    azimuth_coverage, azimuth_effective_min = _azimuth_metrics(position, group_id, weight)

    row: dict[str, float | int | str] = {
        "run": run,
        "file": path.name,
        "step": step,
        "time": time,
        "n_particles": len(position),
        "core_radius_strength_weighted_mean": _weighted_mean(core_radius, weight),
        "overlap_weighted_mean": _weighted_mean(overlap, weight),
        "overlap_p50": q50,
        "overlap_p90": q90,
        "overlap_p95": q95,
        "overlap_p99": q99,
        "overlap_fraction_gt_1": _weighted_mean((overlap > 1.0).astype(float), weight),
        "misalignment_weighted_mean_degrees": _weighted_mean(angle, weight),
        "misalignment_p50_degrees": a50,
        "misalignment_p90_degrees": a90,
        "misalignment_p95_degrees": a95,
        "coverage_misalignment_spearman_strong": correlation,
        "overlap_mean_top_misalignment_decile": _weighted_mean(
            overlap[high_angle], weight[high_angle]
        ),
        "overlap_mean_other_strong_particles": _weighted_mean(
            overlap[other_angle], weight[other_angle]
        ),
        "cross_group_rho_p01": c01,
        "cross_group_rho_p50": c50,
        "cross_group_fraction_within_4sigma": _weighted_mean(
            (cross_rho < 4.0).astype(float), weight
        ),
        "azimuth_bin_coverage_min": azimuth_coverage,
        "azimuth_effective_particles_per_bin_min": azimuth_effective_min,
    }

    region_rows: list[dict] = []
    for group in np.unique(group_id):
        selected = group_id == group
        centred_x = position[selected, 0] - np.average(
            position[selected, 0], weights=weight[selected]
        )
        edges = _weighted_quantiles(centred_x, weight[selected], (0.2, 0.4, 0.6, 0.8))
        quintile = np.digitize(centred_x, edges)
        selected_index = np.flatnonzero(selected)
        for region in range(5):
            index = selected_index[quintile == region]
            oq50, oq95 = _weighted_quantiles(overlap[index], weight[index], (0.50, 0.95))
            aq50, aq90 = _weighted_quantiles(angle[index], weight[index], (0.50, 0.90))
            region_rows.append(
                {
                    "run": run,
                    "file": path.name,
                    "step": step,
                    "time": time,
                    "group_id": int(group),
                    "axial_quintile": region + 1,
                    "overlap_p50": oq50,
                    "overlap_p95": oq95,
                    "misalignment_p50_degrees": aq50,
                    "misalignment_p90_degrees": aq90,
                }
            )
    return row, region_rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_flow_integrals(samples_root: Path, runs: list[str]) -> dict[str, np.ndarray]:
    histories = {}
    for run in runs:
        path = samples_root / run / "flow_integrals.csv"
        histories[run] = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return histories


def _plot(history_rows: list[dict], region_rows: list[dict], histories, output: Path) -> None:
    labels = {
        "cs_breakdown_baseline": "Baseline",
        "cs_breakdown_stretching_viscosity": "Stretching viscosity",
        "cs_breakdown_coverage_h05_fixed_sigma": r"Coverage: $h=.05$, $\sigma_0=.06$",
        "cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation": (
            r"Coverage: $h=.05$, $\sigma_0=.06$"
        ),
    }
    colours = {
        "cs_breakdown_baseline": "#2f5f8f",
        "cs_breakdown_stretching_viscosity": "#d18b28",
        "cs_breakdown_coverage_h05_fixed_sigma": "#2c8c6b",
        "cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation": "#2c8c6b",
    }
    styles = {
        "cs_breakdown_baseline": "-",
        "cs_breakdown_stretching_viscosity": "--",
        "cs_breakdown_coverage_h05_fixed_sigma": "-.",
        "cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation": "-.",
    }
    runs = list(histories)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), constrained_layout=True)
    for run in runs:
        data = histories[run]
        colour = colours.get(run, "#555555")
        style = styles.get(run, "-")
        label = labels.get(run, run)
        axes[0, 0].plot(
            data["time"], data["mean_overlap_ratio"], style, color=colour, label=f"{label}: mean"
        )
        axes[0, 0].plot(
            data["time"], data["max_overlap_ratio"], ":", color=colour, label=f"{label}: max"
        )
        axes[0, 1].plot(
            data["time"], data["mean_particle_core_radius"], style, color=colour, label=label
        )
        axes[1, 0].plot(
            data["time"],
            data["vortex_strength_misalignment_degrees"],
            style,
            color=colour,
            label=label,
        )

    axes[0, 0].axhline(1.0, color="#333333", linewidth=1.0, label="Consistency threshold")
    axes[0, 0].set(
        title="Global particle overlap", ylabel=r"$h_{nn}/\sigma$", xlabel="Physical time (s)"
    )
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].set(
        title="Core spreading",
        ylabel=r"Mean numerical core $\sigma$ (m)",
        xlabel="Physical time (s)",
    )
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].axhline(25.0, color="#333333", linewidth=1.0, label="Native health guard")
    axes[1, 0].set(
        title="Strength–vorticity alignment",
        ylabel="Weighted mean angle (deg)",
        xlabel="Physical time (s)",
    )
    axes[1, 0].legend(fontsize=8)

    final_by_run = {}
    for run in runs:
        selected = [row for row in region_rows if row["run"] == run]
        final_step = max(row["step"] for row in selected)
        final_by_run[run] = [row for row in selected if row["step"] == final_step]
    x = np.arange(1, 6)
    offsets = np.linspace(-0.15, 0.15, max(len(runs), 1))
    for offset, run in zip(offsets, runs, strict=True):
        rows = final_by_run[run]
        # Pool the two original rings without hiding their range.
        median = np.array(
            [np.mean([row["overlap_p50"] for row in rows if row["axial_quintile"] == q]) for q in x]
        )
        high = np.array(
            [np.max([row["overlap_p95"] for row in rows if row["axial_quintile"] == q]) for q in x]
        )
        axes[1, 1].plot(
            x + offset,
            median,
            marker="o",
            linestyle=styles.get(run, "-"),
            color=colours.get(run, "#555555"),
            label=f"{labels.get(run, run)}: median",
        )
        axes[1, 1].plot(
            x + offset,
            high,
            marker="x",
            linestyle=":",
            color=colours.get(run, "#555555"),
            label=f"{labels.get(run, run)}: worst p95",
        )
    axes[1, 1].axhline(1.0, color="#333333", linewidth=1.0)
    axes[1, 1].set(
        title="Final local coverage by axial quintile",
        xlabel="Within-ring axial quintile (ends: 1 and 5)",
        ylabel=r"Same-ring $h_{nn}/\sigma$",
        xticks=x,
    )
    axes[1, 1].legend(fontsize=7, ncol=2)

    for axis in axes.flat:
        axis.grid(color="#d9dde2", linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Seeded CS particle coverage, core growth, and native health")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tutorial-directory", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=("cs_breakdown_baseline", "cs_breakdown_stretching_viscosity"),
    )
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    tutorial = args.tutorial_directory.resolve()
    output = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else tutorial / "figures" / "coverage_diagnosis"
    )
    history_rows: list[dict] = []
    region_rows: list[dict] = []
    for run in args.runs:
        solution = tutorial / "solution" / run
        backups = sorted(
            (path for path in solution.glob("vpm_*.h5") if BACKUP_PATTERN.search(path.name)),
            key=lambda path: int(BACKUP_PATTERN.search(path.name).group(1)),
        )
        if not backups:
            raise FileNotFoundError(f"no native backups found for {run}: {solution}")
        for backup in backups:
            row, regions = _snapshot_metrics(backup, run)
            history_rows.append(row)
            region_rows.extend(regions)

    _write_rows(output / "particle_coverage_history.csv", history_rows)
    _write_rows(output / "particle_coverage_by_axial_quintile.csv", region_rows)
    histories = _load_flow_integrals(tutorial / "samples", list(args.runs))
    _plot(history_rows, region_rows, histories, output / "particle_coverage_diagnosis.png")


if __name__ == "__main__":
    main()
