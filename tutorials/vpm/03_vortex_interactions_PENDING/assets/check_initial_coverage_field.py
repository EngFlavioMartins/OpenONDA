#!/usr/bin/env python3
"""Compare initial ring sampling, tails, and mutual induction offline.

All cases represent the same two seeded Gaussian rings.  The isolated coverage
contrast changes particle spacing while holding the numerical Gaussian radius
at 0.06 R0.  The refined h=0.04 cloud is used only as a discrete-quadrature
reference for common physical probe points; it is not an LBM reference.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import openonda.vpm as vpm
from source.solvers.vpm.kernels import make_vortex_kernel


R0 = 1.0
GAMMA = np.pi
CORE_RADIUS = 0.1
SEPARATION = 1.0
REYNOLDS_NUMBER = 3415.0
DISTURBANCE_AMPLITUDE = 0.05
DISTURBANCE_MODE = 8
BASELINE_SMAGORINSKY = 0.20
REFERENCE_NAME = "h040_sigma060_tail1e4"


@dataclass(frozen=True)
class Representation:
    name: str
    spacing: float
    particle_core_radius: float
    tail_fraction: float


REPRESENTATIONS = (
    Representation("h060_sigma060_tail1e4", 0.06, 0.06, 1.0e-4),
    Representation("h060_sigma060_tail1e6", 0.06, 0.06, 1.0e-6),
    Representation("h050_sigma060_tail1e4", 0.05, 0.06, 1.0e-4),
    Representation("h050_sigma050_tail1e4", 0.05, 0.05, 1.0e-4),
    Representation(REFERENCE_NAME, 0.04, 0.06, 1.0e-4),
)


def _build_pair(specification: Representation):
    represented_core = np.sqrt(CORE_RADIUS**2 - specification.particle_core_radius**2)
    tube_radius = represented_core * np.sqrt(-np.log(specification.tail_fraction))
    disturbance = vpm.WidnallDisturbance.single_mode(
        amplitude=DISTURBANCE_AMPLITUDE,
        mode=DISTURBANCE_MODE,
        direction="axial",
    )
    rings = []
    for group, centre_x in enumerate((-0.5 * SEPARATION, 0.5 * SEPARATION)):
        ring = vpm.VortexRing(
            centre=(centre_x, 0.0, 0.0),
            radius=R0,
            circulation=GAMMA,
            vortex_core_radius=CORE_RADIUS,
            kinematic_viscosity=GAMMA / REYNOLDS_NUMBER,
            disturbance=disturbance,
            core_compensation=vpm.ParticleCoreCompensation(),
            distribution=vpm.ToroidalDistribution(
                centre=(centre_x, 0.0, 0.0),
                ring_radius=R0,
                tube_radius=tube_radius,
                spacing=specification.spacing,
                core_radius_ratio=specification.particle_core_radius / specification.spacing,
                disturbance=disturbance,
            ),
            group_id=group,
        ).build()
        rings.append(ring)
    return tuple(rings)


def _concatenate(rings):
    return {
        field: np.concatenate([getattr(ring, field) for ring in rings])
        for field in ("position", "vortex_strength", "core_radius", "particle_volume", "group_id")
    }


def _scalar_circulation(ring, centre_x: float) -> float:
    relative = ring.position - np.array([centre_x, 0.0, 0.0])
    radial = np.hypot(relative[:, 1], relative[:, 2])
    tangent = np.column_stack((np.zeros(len(radial)), -relative[:, 2], relative[:, 1]))
    tangent /= radial[:, None]
    return float(
        np.sum(np.einsum("ij,ij->i", ring.vortex_strength, tangent) / radial) / (2.0 * np.pi)
    )


def _linear_impulse(position: np.ndarray, strength: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.cross(position, strength), axis=0)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average(values * values, weights=weights)))


def _evaluate_field(target: np.ndarray, sources: dict) -> tuple[np.ndarray, np.ndarray]:
    kernel = make_vortex_kernel("GAUSSIAN")
    source_position = sources["position"].astype(np.float64)
    source_strength = sources["vortex_strength"].astype(np.float64)
    source_core = sources["core_radius"].astype(np.float64)
    target = np.asarray(target, dtype=np.float64)
    velocity = np.zeros((len(target), 3))
    gradient = np.zeros((len(target), 3, 3))
    target_chunk = 32
    source_chunk = 2048
    for target_start in range(0, len(target), target_chunk):
        target_end = min(target_start + target_chunk, len(target))
        points = target[target_start:target_end]
        local_velocity = np.zeros((len(points), 3))
        local_gradient = np.zeros((len(points), 3, 3))
        for source_start in range(0, len(source_position), source_chunk):
            source_end = min(source_start + source_chunk, len(source_position))
            position = source_position[source_start:source_end]
            strength = source_strength[source_start:source_end]
            core = source_core[source_start:source_end]
            displacement = points[:, None, :] - position[None, :, :]
            # Passing source core on both sides makes the pair-radius helper
            # exactly reproduce the source-radius arbitrary-target operator.
            local_velocity += np.sum(
                kernel.velocity_pair(
                    displacement,
                    strength[None, :, :],
                    core[None, :],
                    core[None, :],
                ),
                axis=1,
            )
            local_gradient += np.sum(
                kernel.gradient_pair(
                    displacement,
                    strength[None, :, :],
                    core[None, :],
                    core[None, :],
                ),
                axis=1,
            )
        velocity[target_start:target_end] = local_velocity
        gradient[target_start:target_end] = local_gradient
    return velocity, gradient


def _centreline_probes(centre_x: float, count: int = 64) -> np.ndarray:
    polar = 2.0 * np.pi * np.arange(count) / count
    axial = centre_x + DISTURBANCE_AMPLITUDE * np.sin(DISTURBANCE_MODE * polar)
    return np.column_stack((axial, np.cos(polar), np.sin(polar)))


def _gap_probes() -> np.ndarray:
    axial = np.linspace(-0.4, 0.4, 5)
    radial = np.linspace(0.8, 1.2, 5)
    polar = 2.0 * np.pi * np.arange(16) / 16
    return np.asarray(
        [
            (x, radius * np.cos(angle), radius * np.sin(angle))
            for x in axial
            for radius in radial
            for angle in polar
        ],
        dtype=np.float64,
    )


def _relative_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((candidate - reference) ** 2))
        / max(np.sqrt(np.mean(reference**2)), np.finfo(float).tiny)
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(errors: list[dict], output: Path) -> None:
    selected = [
        row
        for row in errors
        if row["comparison"] == "refined_quadrature" and row["representation"] != REFERENCE_NAME
    ]
    display_names = {
        "h060_sigma060_tail1e4": r"$h=.06,\ \sigma=.06$",
        "h060_sigma060_tail1e6": r"$h=.06,\ \sigma=.06,$ tail $10^{-6}$",
        "h050_sigma060_tail1e4": r"$h=.05,\ \sigma=.06$",
        "h050_sigma050_tail1e4": r"$h=.05,\ \sigma=.05$",
    }
    names = [display_names[row["representation"]] for row in selected]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    colours = ("#2f5f8f", "#d18b28")
    for offset, metric, label, colour in (
        (-0.17, "centreline_velocity_relative_rms", "Mutual velocity", colours[0]),
        (0.17, "centreline_gradient_relative_rms", "Mutual gradient", colours[1]),
    ):
        axes[0].bar(
            x + offset, [row[metric] for row in selected], width=0.32, label=label, color=colour
        )
    axes[0].set_yscale("log")
    axes[0].set(
        title="Cross-ring induction error",
        ylabel="Relative RMS vs h=0.04",
        xticks=x,
        xticklabels=names,
    )
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].legend(fontsize=8)

    for offset, metric, label, colour in (
        (-0.17, "gap_velocity_relative_rms", "Gap velocity", colours[0]),
        (0.17, "gap_gradient_relative_rms", "Gap gradient", colours[1]),
    ):
        axes[1].bar(
            x + offset, [row[metric] for row in selected], width=0.32, label=label, color=colour
        )
    axes[1].set_yscale("log")
    axes[1].set(
        title="Inter-ring probe error", ylabel="Relative RMS vs h=0.04", xticks=x, xticklabels=names
    )
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", color="#d9dde2", linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Initial represented-field sensitivity at fixed seeded ring physics")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "coverage_diagnosis",
    )
    args = parser.parse_args()
    output = args.output_directory.resolve()

    pairs = {specification.name: _build_pair(specification) for specification in REPRESENTATIONS}
    clouds = {name: _concatenate(pair) for name, pair in pairs.items()}
    representation_rows = []
    baseline_delta_rms = None
    for specification in REPRESENTATIONS:
        cloud = clouds[specification.name]
        strength_weight = np.linalg.norm(cloud["vortex_strength"], axis=1)
        delta = np.cbrt(cloud["particle_volume"])
        delta_rms = _weighted_rms(delta, strength_weight)
        if specification.name == "h060_sigma060_tail1e4":
            baseline_delta_rms = delta_rms
        impulse = _linear_impulse(cloud["position"], cloud["vortex_strength"])
        representation_rows.append(
            {
                "representation": specification.name,
                "spacing": specification.spacing,
                "particle_core_radius": specification.particle_core_radius,
                "tail_fraction": specification.tail_fraction,
                "n_particles": len(cloud["position"]),
                "scalar_circulation_group_0": _scalar_circulation(
                    pairs[specification.name][0], -0.5
                ),
                "scalar_circulation_group_1": _scalar_circulation(
                    pairs[specification.name][1], 0.5
                ),
                "linear_impulse_x": impulse[0],
                "linear_impulse_y": impulse[1],
                "linear_impulse_z": impulse[2],
                "filter_delta_strength_weighted_rms": delta_rms,
                "smagorinsky_for_matched_rms_CsDelta": (
                    BASELINE_SMAGORINSKY * baseline_delta_rms / delta_rms
                    if baseline_delta_rms is not None
                    else BASELINE_SMAGORINSKY
                ),
            }
        )

    reference = clouds[REFERENCE_NAME]
    centreline = np.vstack((_centreline_probes(-0.5), _centreline_probes(0.5)))
    gap = _gap_probes()
    reference_centreline = []
    for target_group, source_group in ((0, 1), (1, 0)):
        source = {
            key: value[reference["group_id"] == source_group]
            for key, value in reference.items()
            if key != "group_id"
        }
        target = centreline[target_group * 64 : (target_group + 1) * 64]
        reference_centreline.append(_evaluate_field(target, source))
    reference_centreline_velocity = np.vstack([item[0] for item in reference_centreline])
    reference_centreline_gradient = np.vstack([item[1] for item in reference_centreline])
    reference_gap_velocity, reference_gap_gradient = _evaluate_field(gap, reference)

    error_rows = []
    fields = {}
    for specification in REPRESENTATIONS:
        cloud = clouds[specification.name]
        centreline_results = []
        for target_group, source_group in ((0, 1), (1, 0)):
            source = {
                key: value[cloud["group_id"] == source_group]
                for key, value in cloud.items()
                if key != "group_id"
            }
            target = centreline[target_group * 64 : (target_group + 1) * 64]
            centreline_results.append(_evaluate_field(target, source))
        centreline_velocity = np.vstack([item[0] for item in centreline_results])
        centreline_gradient = np.vstack([item[1] for item in centreline_results])
        gap_velocity, gap_gradient = _evaluate_field(gap, cloud)
        fields[specification.name] = (
            centreline_velocity,
            centreline_gradient,
            gap_velocity,
            gap_gradient,
        )
        error_rows.append(
            {
                "comparison": "refined_quadrature",
                "representation": specification.name,
                "centreline_velocity_relative_rms": _relative_rms(
                    centreline_velocity, reference_centreline_velocity
                ),
                "centreline_gradient_relative_rms": _relative_rms(
                    centreline_gradient, reference_centreline_gradient
                ),
                "gap_velocity_relative_rms": _relative_rms(gap_velocity, reference_gap_velocity),
                "gap_gradient_relative_rms": _relative_rms(gap_gradient, reference_gap_gradient),
            }
        )

    baseline_name = "h060_sigma060_tail1e4"
    tail_name = "h060_sigma060_tail1e6"
    baseline = fields[baseline_name]
    tail = fields[tail_name]
    error_rows.append(
        {
            "comparison": "tail_extension_vs_baseline",
            "representation": tail_name,
            "centreline_velocity_relative_rms": _relative_rms(tail[0], baseline[0]),
            "centreline_gradient_relative_rms": _relative_rms(tail[1], baseline[1]),
            "gap_velocity_relative_rms": _relative_rms(tail[2], baseline[2]),
            "gap_gradient_relative_rms": _relative_rms(tail[3], baseline[3]),
        }
    )

    # Appending particles with exactly zero strength is a direct null check.
    # It cannot alter exact induction because every source contribution is
    # linear in source strength.  Use the extended-tail positions/geometry to
    # exercise that statement on the same probes.
    baseline_cloud = clouds[baseline_name]
    tail_cloud = clouds[tail_name]
    null_cloud = {
        "position": np.concatenate((baseline_cloud["position"], tail_cloud["position"])),
        "vortex_strength": np.concatenate(
            (baseline_cloud["vortex_strength"], np.zeros_like(tail_cloud["vortex_strength"]))
        ),
        "core_radius": np.concatenate((baseline_cloud["core_radius"], tail_cloud["core_radius"])),
    }
    null_centreline = []
    for target_group, source_group in ((0, 1), (1, 0)):
        baseline_selected = baseline_cloud["group_id"] == source_group
        tail_selected = tail_cloud["group_id"] == source_group
        source = {
            "position": np.concatenate(
                (
                    baseline_cloud["position"][baseline_selected],
                    tail_cloud["position"][tail_selected],
                )
            ),
            "vortex_strength": np.concatenate(
                (
                    baseline_cloud["vortex_strength"][baseline_selected],
                    np.zeros_like(tail_cloud["vortex_strength"][tail_selected]),
                )
            ),
            "core_radius": np.concatenate(
                (
                    baseline_cloud["core_radius"][baseline_selected],
                    tail_cloud["core_radius"][tail_selected],
                )
            ),
        }
        target = centreline[target_group * 64 : (target_group + 1) * 64]
        null_centreline.append(_evaluate_field(target, source))
    null_centreline_velocity = np.vstack([item[0] for item in null_centreline])
    null_centreline_gradient = np.vstack([item[1] for item in null_centreline])
    null_gap_velocity, null_gap_gradient = _evaluate_field(gap, null_cloud)
    zero_rows = [
        {
            "n_baseline_particles": len(baseline_cloud["position"]),
            "n_appended_zero_strength_particles": len(tail_cloud["position"]),
            "centreline_velocity_max_absolute_difference": float(
                np.max(np.abs(null_centreline_velocity - baseline[0]))
            ),
            "centreline_gradient_max_absolute_difference": float(
                np.max(np.abs(null_centreline_gradient - baseline[1]))
            ),
            "gap_velocity_max_absolute_difference": float(
                np.max(np.abs(null_gap_velocity - baseline[2]))
            ),
            "gap_gradient_max_absolute_difference": float(
                np.max(np.abs(null_gap_gradient - baseline[3]))
            ),
        }
    ]

    _write_rows(output / "initial_representations.csv", representation_rows)
    _write_rows(output / "initial_field_errors.csv", error_rows)
    _write_rows(output / "zero_strength_null.csv", zero_rows)
    _plot(error_rows, output / "initial_field_sensitivity.png")


if __name__ == "__main__":
    main()
