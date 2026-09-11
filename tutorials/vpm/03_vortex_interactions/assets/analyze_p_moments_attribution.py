#!/usr/bin/env python3
"""Attribute the seeded p-moments contrast using completed native outputs.

This is read-only post-processing.  It compares particle geometry, strength-
weighted proxies, and the two native sampled field planes at exact saved times.
One explicitly labelled frozen-state event separates the local Pedrizzetti
blend from its global moment restoration; it is not a time-evolution result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "cs_breakdown_baseline"
REALIGNED = "cs_breakdown_p_moments_cpu_t6_qualification"
PARTICLE_STEPS = (100, 200, 300)
FIELD_STEPS = {"core_section": (100, 200, 300), "cross_section": (80, 200, 280)}
MODE = 8
BIN_COUNT = 128
RING_RADIUS = 1.0
RING_CIRCULATION = np.pi
REYNOLDS_CIRCULATION = 3415.0
INITIAL_PHYSICAL_CORE = 0.1
INITIAL_PARTICLE_CORE = 0.06
RELAXATION_FACTOR = 0.0028851361104375
FIELD_CORE_CUTOFF = 0.6

# These three implementations are unchanged between the archived installed
# generation used for the run and the checkout used for this offline audit.
FROZEN_OPERATOR_HASHES = {
    "operators.py": "48523231105271489d08f328717c46c5cbc53bfb66363524ea76570e4267fe87",
    "divergence_relaxation.py": (
        "0ac7340da35a53a63c6e2f710563b7dd73da86e376311b995df268039ad2b96c"
    ),
    "filament_refinement.py": ("c10409187d6c9ee27e053c7eb1a04d3b20f1441de2975ac53eed6a8547e0ead9"),
}
REALIGNED_FINAL_H5_SHA256 = "38815ff626012ab26fdf4d1587923a74b12c3ae5fe99e054f7b4d01d695ba507"


def _load_particles(run: str, step: int) -> dict[str, np.ndarray | float]:
    path = ROOT / "solution" / run / f"vpm_{step:06d}.h5"
    with h5py.File(path, "r") as file:
        particles = file["particles"]
        result = {
            name: np.asarray(particles[name], dtype=np.float64)
            for name in (
                "position",
                "vortex_strength",
                "vorticity",
                "core_radius",
                "particle_volume",
            )
        }
        result["group_id"] = np.asarray(particles["group_id"], dtype=np.int32)
        result["time"] = float(file["solver"].attrs["time"])
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complex_modes(position: np.ndarray, weight: np.ndarray) -> tuple[complex, complex]:
    """Return the centered equal-azimuth-bin axial and radial coefficients."""
    theta = np.mod(np.arctan2(position[:, 2], position[:, 1]), 2.0 * np.pi)
    radius = np.hypot(position[:, 1], position[:, 2])
    bins = np.minimum((theta * BIN_COUNT / (2.0 * np.pi)).astype(int), BIN_COUNT - 1)
    weight_by_bin = np.bincount(bins, weights=weight, minlength=BIN_COUNT)
    occupied = weight_by_bin > np.finfo(float).tiny
    cosine = (
        np.bincount(bins, weights=weight * np.cos(MODE * theta), minlength=BIN_COUNT)[occupied]
        / weight_by_bin[occupied]
    )
    sine = (
        np.bincount(bins, weights=weight * np.sin(MODE * theta), minlength=BIN_COUNT)[occupied]
        / weight_by_bin[occupied]
    )
    design = np.column_stack((np.ones(np.count_nonzero(occupied)), cosine, sine))
    coefficients = []
    for values in (position[:, 0], radius):
        by_bin = (
            np.bincount(bins, weights=weight * values, minlength=BIN_COUNT)[occupied]
            / weight_by_bin[occupied]
        )
        fit, *_ = np.linalg.lstsq(design, by_bin / RING_RADIUS, rcond=None)
        coefficients.append(0.5 * (fit[1] - 1j * fit[2]))
    return coefficients[0], coefficients[1]


def _cloud_geometry(position: np.ndarray) -> tuple[np.ndarray, float, float]:
    centroid = position.mean(axis=0)
    centered = position - centroid
    covariance = centered.T @ centered / len(position)
    eigenvalues = np.linalg.eigvalsh(covariance)
    major_radius = float(np.sqrt(max(eigenvalues[-1] + eigenvalues[-2], 0.0)))
    mean_radius = float(np.mean(np.hypot(position[:, 1], position[:, 2])))
    return centroid, major_radius, mean_radius


def _particle_moments(
    position: np.ndarray, strength: np.ndarray, core_radius: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    total = strength.sum(axis=0, dtype=np.float64)
    variation = float(np.linalg.norm(strength, axis=1).sum(dtype=np.float64))
    impulse = 0.5 * np.cross(position, strength).sum(axis=0, dtype=np.float64)
    angular = (
        np.cross(position, np.cross(position, strength)).sum(axis=0, dtype=np.float64) / 3.0
        - (core_radius[:, None] ** 2 * strength).sum(axis=0, dtype=np.float64) / 3.0
    )
    return total, variation, impulse, angular


def _weighted_geometry(
    position: np.ndarray, strength: np.ndarray, core_radius: np.ndarray
) -> dict[str, float]:
    weight = np.linalg.norm(strength, axis=1)
    denominator = max(float(weight.sum()), np.finfo(float).tiny)
    centroid = np.einsum("i,ij->j", weight, position) / denominator
    centered = position - centroid
    covariance = (centered * weight[:, None]).T @ centered / denominator
    eigenvalues = np.linalg.eigvalsh(covariance)
    major_radius = float(np.sqrt(max(eigenvalues[-1] + eigenvalues[-2], 0.0)))
    total, variation, impulse, angular = _particle_moments(position, strength, core_radius)
    axial, radial = _complex_modes(position, weight)
    unweighted_axial, unweighted_radial = _complex_modes(position, np.ones(len(position)))
    return {
        "weighted_centroid_x": float(centroid[0]),
        "weighted_mean_radius": float(
            np.dot(weight, np.hypot(position[:, 1], position[:, 2])) / denominator
        ),
        "weighted_major_radius": major_radius,
        "strength_variation": variation,
        "net_strength_norm": float(np.linalg.norm(total)),
        "linear_impulse_x": float(impulse[0]),
        "linear_impulse_norm": float(np.linalg.norm(impulse)),
        "angular_impulse_norm": float(np.linalg.norm(angular)),
        "impulse_radius": float(2.0 * np.linalg.norm(impulse) / denominator),
        "tube_circulation_proxy": float(denominator / (2.0 * np.pi * major_radius)),
        "axial_mode_amplitude": float(2.0 * abs(axial)),
        "axial_mode_real": float(axial.real),
        "axial_mode_imag": float(axial.imag),
        "radial_mode_amplitude": float(2.0 * abs(radial)),
        "radial_mode_real": float(radial.real),
        "radial_mode_imag": float(radial.imag),
        "unweighted_axial_mode_amplitude": float(2.0 * abs(unweighted_axial)),
        "unweighted_radial_mode_amplitude": float(2.0 * abs(unweighted_radial)),
    }


def _particle_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    attribution_rows = []
    for step in PARTICLE_STEPS:
        data = {run: _load_particles(run, step) for run in (BASELINE, REALIGNED)}
        if not np.array_equal(data[BASELINE]["group_id"], data[REALIGNED]["group_id"]):
            raise ValueError(f"particle group ordering differs at step {step}")
        for run, values in data.items():
            groups = values["group_id"]
            for group in sorted(np.unique(groups)):
                selected = groups == group
                position = values["position"][selected]
                strength = values["vortex_strength"][selected]
                core_radius = values["core_radius"][selected]
                centroid, major_radius, mean_radius = _cloud_geometry(position)
                raw_rows.append(
                    {
                        "run": run,
                        "step": step,
                        "time": values["time"],
                        "group_id": int(group),
                        "n_particles": int(np.count_nonzero(selected)),
                        "unweighted_centroid_x": float(centroid[0]),
                        "unweighted_centroid_y": float(centroid[1]),
                        "unweighted_centroid_z": float(centroid[2]),
                        "unweighted_major_radius": major_radius,
                        "unweighted_mean_radius": mean_radius,
                        **_weighted_geometry(position, strength, core_radius),
                    }
                )

        baseline = data[BASELINE]
        realigned = data[REALIGNED]
        for group in sorted(np.unique(baseline["group_id"])):
            selected = baseline["group_id"] == group
            position_b = baseline["position"][selected]
            position_p = realigned["position"][selected]
            weight_b = np.linalg.norm(baseline["vortex_strength"][selected], axis=1)
            weight_p = np.linalg.norm(realigned["vortex_strength"][selected], axis=1)
            combinations = {
                "bb": _complex_modes(position_b, weight_b),
                "pb": _complex_modes(position_p, weight_b),
                "bp": _complex_modes(position_b, weight_p),
                "pp": _complex_modes(position_p, weight_p),
            }
            row = {
                "step": step,
                "time": baseline["time"],
                "group_id": int(group),
            }
            for component, index in (("axial", 0), ("radial", 1)):
                amplitude = {key: 2.0 * abs(value[index]) for key, value in combinations.items()}
                geometry = 0.5 * (
                    amplitude["pb"] - amplitude["bb"] + amplitude["pp"] - amplitude["bp"]
                )
                weighting = 0.5 * (
                    amplitude["bp"] - amplitude["bb"] + amplitude["pp"] - amplitude["pb"]
                )
                row.update(
                    {
                        f"{component}_baseline_amplitude": amplitude["bb"],
                        f"{component}_realigned_amplitude": amplitude["pp"],
                        f"{component}_amplitude_change": amplitude["pp"] - amplitude["bb"],
                        f"{component}_geometry_contribution": geometry,
                        f"{component}_strength_weight_contribution": weighting,
                        f"{component}_baseline_real": float(combinations["bb"][index].real),
                        f"{component}_baseline_imag": float(combinations["bb"][index].imag),
                        f"{component}_realigned_real": float(combinations["pp"][index].real),
                        f"{component}_realigned_imag": float(combinations["pp"][index].imag),
                    }
                )
            attribution_rows.append(row)
    return pd.DataFrame(raw_rows), pd.DataFrame(attribution_rows)


def _ring_centers(run: str, step: int) -> np.ndarray:
    rings = pd.read_csv(ROOT / "samples" / run / "ring_diagnostics.csv")
    rows = rings[rings.step == step].sort_values("group_id")
    if len(rows) != 2:
        raise ValueError(f"expected two ring rows for {run} step {step}")
    return rows[["vortex_centroid_x", "major_radius"]].to_numpy(dtype=np.float64)


def _field_geometry_rows() -> pd.DataFrame:
    rows = []
    for plane, steps in FIELD_STEPS.items():
        for step in steps:
            for run in (BASELINE, REALIGNED):
                path = ROOT / "samples" / run / f"{plane}_{step:06d}.vts"
                grid = pv.read(path)
                points = np.asarray(grid.points, dtype=np.float64)
                vorticity = np.asarray(grid.point_data["vorticity"], dtype=np.float64)
                coordinates = (
                    points[:, :2]
                    if plane == "core_section"
                    else np.column_stack((points[:, 0], np.abs(points[:, 2])))
                )
                centers = _ring_centers(run, step)
                distance_sq = np.sum((coordinates[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                assignment = np.argmin(distance_sq, axis=1)
                for group in range(2):
                    selected = assignment == group
                    local_coordinates = coordinates[selected]
                    local_distance_sq = np.sum((local_coordinates - centers[group]) ** 2, axis=1)
                    weight = np.linalg.norm(vorticity[selected], axis=1)
                    weight *= local_distance_sq <= FIELD_CORE_CUTOFF**2
                    denominator = float(weight.sum())
                    centroid = np.einsum("i,ij->j", weight, local_coordinates) / denominator
                    width = float(
                        np.sqrt(
                            np.dot(
                                weight,
                                np.sum((local_coordinates - centroid) ** 2, axis=1),
                            )
                            / denominator
                        )
                    )
                    time = float(
                        pd.read_csv(ROOT / "samples" / run / "ring_diagnostics.csv")
                        .query("step == @step")
                        .time.iloc[0]
                    )
                    rows.append(
                        {
                            "run": run,
                            "plane": plane,
                            "step": step,
                            "time": time,
                            "group_id": group,
                            "field_centroid_x": float(centroid[0]),
                            "field_major_radius": float(centroid[1]),
                            "field_core_rms_width": width,
                            "field_peak_vorticity": float(
                                np.linalg.norm(vorticity[selected], axis=1).max()
                            ),
                            "core_cutoff_radius": FIELD_CORE_CUTOFF,
                        }
                    )
    return pd.DataFrame(rows)


def _invariant_rows(position: np.ndarray, core_radius: np.ndarray) -> np.ndarray:
    n = len(position)
    identity = np.eye(3)
    rows = np.zeros((9, n, 3), dtype=np.float64)
    rows[0:3] = identity[:, None, :]
    skew = np.zeros((n, 3, 3), dtype=np.float64)
    skew[:, 0, 1], skew[:, 0, 2] = -position[:, 2], position[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = position[:, 2], -position[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -position[:, 1], position[:, 0]
    rows[3:6] = 0.5 * np.transpose(skew, (1, 0, 2))
    skew_sq = (
        np.einsum("pa,pb->pab", position, position)
        - np.einsum("pa,pa->p", position, position)[:, None, None] * identity
    )
    rows[6:9] = np.transpose(
        skew_sq / 3.0 - core_radius[:, None, None] ** 2 * identity / 3.0,
        (1, 0, 2),
    )
    return rows


def _restore_moments(
    position: np.ndarray,
    relaxed: np.ndarray,
    core_radius: np.ndarray,
    volume: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    before = _particle_moments(position, reference, core_radius)
    current = _particle_moments(position, relaxed, core_radius)
    change = np.concatenate(
        (before[0] - current[0], before[2] - current[2], before[3] - current[3])
    )
    rows = _invariant_rows(position, core_radius)
    sqrt_volume = np.sqrt(volume)
    scaled_rows = rows * sqrt_volume[None, :, None]
    gram = np.einsum("mpa,npa->mn", scaled_rows, scaled_rows)
    multipliers = np.linalg.pinv(gram, rcond=1.0e-12) @ change
    transformed = np.einsum("m,mpa->pa", multipliers, scaled_rows)
    return relaxed + sqrt_volume[:, None] * transformed


def _frozen_event_rows() -> pd.DataFrame:
    state = _load_particles(BASELINE, 300)
    position = state["position"]
    strength = state["vortex_strength"]
    vorticity = state["vorticity"]
    core_radius = state["core_radius"]
    volume = state["particle_volume"]
    strength_norm = np.linalg.norm(strength, axis=1)
    vorticity_norm = np.linalg.norm(vorticity, axis=1)
    valid = (strength_norm > 1.0e-30) & (vorticity_norm > 1.0e-30)
    local = strength.copy()
    local[valid] = (1.0 - RELAXATION_FACTOR) * strength[valid] + (
        RELAXATION_FACTOR * strength_norm[valid] / vorticity_norm[valid]
    )[:, None] * vorticity[valid]
    corrected = _restore_moments(position, local, core_radius, volume, strength)
    rows = []
    stages = {"before": strength, "local_blend": local, "global_restored": corrected}
    for stage, values in stages.items():
        for group in (-1, 0, 1):
            selected = (
                np.ones(len(position), dtype=bool) if group == -1 else state["group_id"] == group
            )
            total, variation, impulse, angular = _particle_moments(
                position[selected], values[selected], core_radius[selected]
            )
            rows.append(
                {
                    "state": "baseline_step300_t2.25",
                    "stage": stage,
                    "group_id": group,
                    "strength_variation": variation,
                    "net_strength_x": float(total[0]),
                    "net_strength_y": float(total[1]),
                    "net_strength_z": float(total[2]),
                    "linear_impulse_x": float(impulse[0]),
                    "linear_impulse_y": float(impulse[1]),
                    "linear_impulse_z": float(impulse[2]),
                    "linear_impulse_norm": float(np.linalg.norm(impulse)),
                    "angular_impulse_x": float(angular[0]),
                    "angular_impulse_y": float(angular[1]),
                    "angular_impulse_z": float(angular[2]),
                    "angular_impulse_norm": float(np.linalg.norm(angular)),
                }
            )
    return pd.DataFrame(rows)


def _core_growth_rows(particle: pd.DataFrame, field: pd.DataFrame) -> pd.DataFrame:
    rows = []
    viscosity = RING_CIRCULATION / REYNOLDS_CIRCULATION
    for run in (BASELINE, REALIGNED):
        flow = pd.read_csv(ROOT / "samples" / run / "flow_integrals.csv")
        for time in (0.0, 0.6, 0.75, 1.5, 2.1, 2.25):
            sample = flow[np.isclose(flow.time, time)]
            if sample.empty:
                continue
            mean_sigma = float(sample.iloc[-1].mean_core_radius)
            row = {
                "run": run,
                "time": time,
                "mean_particle_core": mean_sigma,
                "molecular_only_particle_core": float(
                    np.sqrt(INITIAL_PARTICLE_CORE**2 + 4.0 * viscosity * time)
                ),
                "molecular_physical_vortex_core": float(
                    np.sqrt(INITIAL_PHYSICAL_CORE**2 + 4.0 * viscosity * time)
                ),
                "mean_eddy_viscosity": float(sample.iloc[-1].mean_eddy_viscosity),
            }
            rows.append(row)
    result = pd.DataFrame(rows)
    particle_mode = particle[["run", "time", "group_id", "weighted_major_radius"]].copy()
    particle_mode["mode8_wavelength"] = 2.0 * np.pi * particle_mode.weighted_major_radius / MODE
    particle_mode = particle_mode.merge(result, on=["run", "time"], how="left")
    particle_mode["particle_core_over_mode8_wavelength"] = (
        particle_mode.mean_particle_core / particle_mode.mode8_wavelength
    )
    core_field = field[field.plane == "core_section"][
        ["run", "time", "group_id", "field_core_rms_width"]
    ]
    return particle_mode.merge(core_field, on=["run", "time", "group_id"], how="left")


def _plot(
    particle: pd.DataFrame,
    attribution: pd.DataFrame,
    field: pd.DataFrame,
    growth: pd.DataFrame,
    output: Path,
) -> None:
    labels = {BASELINE: "Baseline", REALIGNED: "Realignment from onset"}
    colors = {BASELINE: "#202838", REALIGNED: "#6f4a8e"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    group_zero = particle[particle.group_id == 0]
    for run in (BASELINE, REALIGNED):
        rows = group_zero[group_zero.run == run].sort_values("time")
        axes[0, 0].plot(
            rows.time,
            rows.radial_mode_amplitude,
            "o-",
            color=colors[run],
            label=f"{labels[run]}, strength weighted",
        )
        axes[0, 0].plot(
            rows.time,
            rows.unweighted_radial_mode_amplitude,
            "s--",
            color=colors[run],
            alpha=0.8,
            label=f"{labels[run]}, positions only",
        )
    axes[0, 0].set(title="Group 0 radial mode-8", ylabel="amplitude / R0")
    axes[0, 0].legend(frameon=False, fontsize=7)

    final = attribution[np.isclose(attribution.time, 2.25)].sort_values("group_id")
    locations = np.arange(len(final))
    axes[0, 1].bar(
        locations - 0.18,
        final.radial_geometry_contribution,
        0.36,
        label="particle geometry",
        color="#2c8c6b",
    )
    axes[0, 1].bar(
        locations + 0.18,
        final.radial_strength_weight_contribution,
        0.36,
        label="strength weighting",
        color="#c06c42",
    )
    axes[0, 1].set(
        title="Radial-amplitude change at t=2.25 s",
        ylabel="realigned - baseline",
        xticks=locations,
        xticklabels=[f"group {group}" for group in final.group_id],
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].legend(frameon=False, fontsize=8)

    for run in (BASELINE, REALIGNED):
        rows = growth[(growth.run == run) & (growth.group_id == 0)].sort_values("time")
        axes[1, 0].plot(
            rows.time,
            rows.mean_particle_core,
            "o-",
            color=colors[run],
            label=labels[run],
        )
    reference = growth[growth.run == BASELINE].sort_values("time").drop_duplicates("time")
    axes[1, 0].plot(
        reference.time,
        reference.molecular_only_particle_core,
        ":",
        color="#2c8c6b",
        label="particle core, molecular only",
    )
    axes[1, 0].plot(
        reference.time,
        reference.molecular_physical_vortex_core,
        "--",
        color="#c06c42",
        label="physical Gaussian core, molecular",
    )
    axes[1, 0].set(title="CS/LES core growth", ylabel="core scale / R0")
    axes[1, 0].legend(frameon=False, fontsize=7)

    core = field[field.plane == "core_section"]
    cross = field[field.plane == "cross_section"]
    for run in (BASELINE, REALIGNED):
        for plane, rows, marker in (
            ("meridional", core, "o"),
            ("orthogonal", cross, "s"),
        ):
            group = rows[(rows.run == run) & (rows.group_id == 0)].sort_values("time")
            axes[1, 1].plot(
                group.time,
                group.field_core_rms_width,
                marker + ("-" if plane == "meridional" else "--"),
                color=colors[run],
                label=f"{labels[run]}, {plane}",
            )
    axes[1, 1].set(title="Group 0 reconstructed core", ylabel="vorticity RMS width / R0")
    axes[1, 1].legend(frameon=False, fontsize=7)
    for axis in axes.flat:
        axis.set_xlabel("physical time [s]")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Seeded p-moments attribution; completed native states only", y=0.995)
    figure.tight_layout()
    figure.savefig(output / "p_moments_attribution.png", dpi=180)
    plt.close(figure)


def _write_summary(
    particle: pd.DataFrame,
    attribution: pd.DataFrame,
    field: pd.DataFrame,
    frozen: pd.DataFrame,
    growth: pd.DataFrame,
    output: Path,
) -> None:
    final_particle = particle[np.isclose(particle.time, 2.25)].set_index(["run", "group_id"])
    final_attribution = attribution[np.isclose(attribution.time, 2.25)].set_index("group_id")
    baseline = final_particle.loc[(BASELINE, 0)]
    realigned = final_particle.loc[(REALIGNED, 0)]
    radial = final_attribution.loc[0]
    decrease = radial.radial_amplitude_change
    geometry_fraction = radial.radial_geometry_contribution / decrease
    before = frozen[(frozen.stage == "before") & (frozen.group_id == -1)].iloc[0]
    local = frozen[(frozen.stage == "local_blend") & (frozen.group_id == -1)].iloc[0]
    corrected = frozen[(frozen.stage == "global_restored") & (frozen.group_id == -1)].iloc[0]
    baseline_growth = growth[
        (growth.run == BASELINE) & np.isclose(growth.time, 2.25) & (growth.group_id == 0)
    ].iloc[0]
    baseline_field = field[
        (field.run == BASELINE)
        & (field.plane == "core_section")
        & np.isclose(field.time, 2.25)
        & (field.group_id == 0)
    ].iloc[0]
    lines = [
        "# Seeded p-moments offline attribution",
        "",
        "All comparisons use completed old-generation native outputs. The frozen event is an operator decomposition, not a time-evolution counterfactual.",
        "",
        f"- At t=2.25 s, group-0 strength-weighted radial mode changes from {radial.radial_baseline_amplitude:.6g} to {radial.radial_realigned_amplitude:.6g} R0. The symmetric two-factor decomposition assigns {geometry_fraction:.1%} of the amplitude reduction to changed particle positions and {1.0 - geometry_fraction:.1%} to changed strength weights.",
        f"- The positions-only radial mode changes from {baseline.unweighted_radial_mode_amplitude:.6g} to {realigned.unweighted_radial_mode_amplitude:.6g} R0. The weighted group strength changes by {(realigned.strength_variation / baseline.strength_variation - 1.0):+.2%}.",
        f"- Group-0 impulse radius changes by {(realigned.impulse_radius / baseline.impulse_radius - 1.0):+.2%}, while its impulse norm changes by {(realigned.linear_impulse_norm / baseline.linear_impulse_norm - 1.0):+.2%}; the large radius-proxy change is dominated by its strength-variation denominator.",
        f"- Baseline mean particle sigma at t=2.25 is {baseline_growth.mean_particle_core:.6g} R0 versus {baseline_growth.molecular_only_particle_core:.6g} for molecular-only particle-core spreading. The native meridional reconstructed group-0 core RMS width is {baseline_field.field_core_rms_width:.6g} R0; the molecular physical Gaussian-core expectation is {baseline_growth.molecular_physical_vortex_core:.6g} R0.",
        f"- In one frozen baseline t=2.25 event, the local blend changes total strength variation by {local.strength_variation - before.strength_variation:+.6g}; global moment restoration adds {corrected.strength_variation - local.strength_variation:+.6g}. The restored event changes global linear impulse by {corrected.linear_impulse_norm - before.linear_impulse_norm:+.3e}, while moving equal-and-opposite impulse between the two labelled rings.",
        "",
        "Frozen-event source hashes:",
        "",
        *[f"- `{name}`: `{digest}`" for name, digest in FROZEN_OPERATOR_HASHES.items()],
        "",
        "Native field timing is not interpolated: meridional planes use t=0.75/1.5/2.25 s and orthogonal planes use t=0.6/1.5/2.1 s.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "cs_breakdown_p_moments_attribution",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    realigned_final = ROOT / "solution" / REALIGNED / "vpm_000300.h5"
    observed_final_hash = _sha256(realigned_final)
    if observed_final_hash != REALIGNED_FINAL_H5_SHA256:
        raise ValueError(
            "realigned final state does not match the qualified native artifact: "
            f"{observed_final_hash}"
        )
    particle, attribution = _particle_rows()
    field = _field_geometry_rows()
    frozen = _frozen_event_rows()
    growth = _core_growth_rows(particle, field)
    particle.to_csv(args.output / "particle_geometry.csv", index=False)
    attribution.to_csv(args.output / "mode_attribution.csv", index=False)
    field.to_csv(args.output / "field_core_geometry.csv", index=False)
    frozen.to_csv(args.output / "frozen_event.csv", index=False)
    growth.to_csv(args.output / "core_growth.csv", index=False)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "baseline": BASELINE,
                "realigned": REALIGNED,
                "particle_steps": PARTICLE_STEPS,
                "field_steps": FIELD_STEPS,
                "frozen_operator_hashes": FROZEN_OPERATOR_HASHES,
                "frozen_event_state": "baseline/vpm_000300.h5",
                "frozen_event_factor": RELAXATION_FACTOR,
                "field_core_cutoff": FIELD_CORE_CUTOFF,
                "realigned_final_h5_sha256": REALIGNED_FINAL_H5_SHA256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot(particle, attribution, field, growth, args.output)
    _write_summary(particle, attribution, field, frozen, growth, args.output)
    print(f"wrote p-moments attribution to {args.output}")


if __name__ == "__main__":
    main()
