#!/usr/bin/env python3
"""Gate the Widnall VPM challenge against a factor-two time-step refinement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage5b_dt_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage5b_dt_cache")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RING_ASSETS = ROOT / "tutorials" / "VPM" / "vortexRing" / "assets"
sys.path.insert(0, str(RING_ASSETS))

from ring_diagnostics import RingModeDiagnosticsSampler  # noqa: E402

CIRCULATION = np.pi
DEFAULT_MODES = np.arange(20, 25)
INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID = "#d8dde2"


def mode_history(
    run_directory: Path, modes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv(run_directory / "samples" / "ring_modes.csv")
    data = data[data["group_id"] == data["group_id"].min()]
    amplitude_table = data.pivot_table(
        index="flow_time",
        columns="mode",
        values="combined_amplitude",
        aggfunc="last",
    ).sort_index()
    missing = [mode for mode in modes if mode not in amplitude_table]
    if missing:
        raise ValueError(f"missing modes in {run_directory}: {missing}")
    selected = data[data["mode"].isin(modes)].copy()
    selected["radial_coefficient"] = selected["radial_amplitude"] * np.exp(
        1j * selected["radial_phase"]
    )
    selected["axial_coefficient"] = selected["axial_amplitude"] * np.exp(
        1j * selected["axial_phase"]
    )
    selected = selected.drop_duplicates(["flow_time", "mode"], keep="last")
    radial = selected.pivot(
        index="flow_time", columns="mode", values="radial_coefficient"
    ).sort_index()
    axial = selected.pivot(
        index="flow_time", columns="mode", values="axial_coefficient"
    ).sort_index()
    coefficients = np.concatenate(
        (radial[modes].to_numpy(complex), axial[modes].to_numpy(complex)), axis=1
    )
    time = amplitude_table.index.to_numpy(float) * CIRCULATION
    amplitudes = amplitude_table[modes].to_numpy(float)

    manifest = load_manifest(run_directory)
    final_path = run_directory / f"vpm_{manifest['output_label']}_final.h5"
    if final_path.is_file():
        with h5py.File(final_path, "r") as handle:
            final_time = float(handle["solver"].attrs["flow_time"]) * CIRCULATION
            position = np.asarray(handle["particles/position"], dtype=np.float64)
            circulation = np.asarray(handle["particles/circulation"], dtype=np.float64)
        if final_time > time[-1] + 1.0e-12:
            sampler = RingModeDiagnosticsSampler(
                maximum_mode=max(40, int(modes.max())),
                azimuthal_bins=128,
                reference_radius=1.0,
                transverse_origin=(0.0, 0.0),
            )
            rows = np.asarray(sampler._sample_group(position, circulation), dtype=float)
            selected_rows = rows[modes - 1]
            final_coefficients = np.concatenate(
                (
                    selected_rows[:, 1] * np.exp(1j * selected_rows[:, 4]),
                    selected_rows[:, 2] * np.exp(1j * selected_rows[:, 5]),
                )
            )
            time = np.append(time, final_time)
            amplitudes = np.vstack((amplitudes, selected_rows[:, 3]))
            coefficients = np.vstack((coefficients, final_coefficients))
    return (
        time,
        amplitudes,
        coefficients,
    )


def health_history(run_directory: Path) -> pd.DataFrame:
    data = pd.read_csv(run_directory / "samples" / "flow_integrals.csv")
    return data.sort_values("time").drop_duplicates("step", keep="last")


def ring_history(run_directory: Path) -> pd.DataFrame:
    data = pd.read_csv(run_directory / "samples" / "ring_diagnostics.csv")
    return data.sort_values("flow_time").drop_duplicates("time_step", keep="last")


def interpolate(time: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(values):
        return interpolate(time, values.real, target) + 1j * interpolate(time, values.imag, target)
    return np.column_stack(
        [np.interp(target, time, values[:, column]) for column in range(values.shape[1])]
    )


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def load_manifest(run_directory: Path) -> dict[str, object]:
    manifests = sorted(run_directory.glob("run_manifest_*.json"))
    if len(manifests) != 1:
        raise ValueError(f"expected one run manifest in {run_directory}, found {len(manifests)}")
    return json.loads(manifests[0].read_text(encoding="utf-8"))


def configured_time_step(run_directory: Path) -> float:
    return float(load_manifest(run_directory)["time_step"])


def evaluate(
    coarse_directory: Path, fine_directory: Path, modes: np.ndarray
) -> tuple[dict[str, object], dict]:
    coarse_dt = configured_time_step(coarse_directory)
    fine_dt = configured_time_step(fine_directory)
    coarse_time, coarse_modes, coarse_coefficients = mode_history(coarse_directory, modes)
    fine_time, fine_modes, fine_coefficients = mode_history(fine_directory, modes)
    common_end = min(coarse_time[-1], fine_time[-1])
    selected = coarse_time <= common_end + 1.0e-12
    time = coarse_time[selected]
    coarse_modes = coarse_modes[selected]
    fine_at_coarse = interpolate(fine_time, fine_modes, time)
    coarse_coefficients = coarse_coefficients[selected]
    fine_coefficients_at_coarse = interpolate(fine_time, fine_coefficients, time)
    coefficient_relative_error = np.linalg.norm(
        coarse_coefficients - fine_coefficients_at_coarse, axis=1
    ) / np.maximum(
        np.linalg.norm(fine_coefficients_at_coarse, axis=1),
        np.finfo(float).tiny,
    )

    coarse_health = health_history(coarse_directory)
    fine_health = health_history(fine_directory)
    coarse_ring = ring_history(coarse_directory)
    fine_ring = ring_history(fine_directory)
    coarse_ring_time = coarse_ring["flow_time"].to_numpy(float) * CIRCULATION
    fine_ring_time = fine_ring["flow_time"].to_numpy(float) * CIRCULATION
    ring_target = coarse_ring_time[coarse_ring_time <= common_end + 1.0e-12]
    coarse_radius = np.interp(
        ring_target,
        coarse_ring_time,
        coarse_ring["major_radius"].to_numpy(float),
    )
    fine_radius = np.interp(
        ring_target,
        fine_ring_time,
        fine_ring["major_radius"].to_numpy(float),
    )
    coarse_x = np.interp(
        ring_target,
        coarse_ring_time,
        coarse_ring["x_centroid"].to_numpy(float),
    )
    fine_x = np.interp(
        ring_target,
        fine_ring_time,
        fine_ring["x_centroid"].to_numpy(float),
    )

    metrics = {
        "common_end_time_star": float(common_end),
        "mode_history_relative_l2": relative_l2(coarse_modes, fine_at_coarse),
        "mode_endpoint_relative_l2": relative_l2(coarse_modes[-1], fine_at_coarse[-1]),
        "complex_mode_history_relative_l2": relative_l2(
            coarse_coefficients, fine_coefficients_at_coarse
        ),
        "complex_mode_endpoint_relative_l2": relative_l2(
            coarse_coefficients[-1], fine_coefficients_at_coarse[-1]
        ),
        "maximum_instantaneous_complex_mode_relative_l2": float(np.max(coefficient_relative_error)),
        "ring_radius_relative_l2": relative_l2(coarse_radius, fine_radius),
        "centroid_trajectory_relative_l2": relative_l2(coarse_x[1:], fine_x[1:]),
        "coarse_maximum_divergence": float(coarse_health["vorticity_divergence_error"].max()),
        "fine_maximum_divergence": float(fine_health["vorticity_divergence_error"].max()),
        "coarse_maximum_misalignment_deg": float(coarse_health["strength_misalignment_deg"].max()),
        "fine_maximum_misalignment_deg": float(fine_health["strength_misalignment_deg"].max()),
    }
    checks = {
        "mode_history_within_5_percent": metrics["mode_history_relative_l2"] < 0.05,
        "mode_endpoint_within_5_percent": metrics["mode_endpoint_relative_l2"] < 0.05,
        "complex_mode_history_within_5_percent": (
            metrics["complex_mode_history_relative_l2"] < 0.05
        ),
        "complex_mode_endpoint_within_5_percent": (
            metrics["complex_mode_endpoint_relative_l2"] < 0.05
        ),
        "ring_radius_within_0p5_percent": metrics["ring_radius_relative_l2"] < 0.005,
        "centroid_trajectory_within_0p5_percent": (
            metrics["centroid_trajectory_relative_l2"] < 0.005
        ),
        "both_divergence_below_limit": max(
            metrics["coarse_maximum_divergence"], metrics["fine_maximum_divergence"]
        )
        < 0.12,
        "both_misalignment_below_limit": max(
            metrics["coarse_maximum_misalignment_deg"],
            metrics["fine_maximum_misalignment_deg"],
        )
        < 45.0,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "comparison": f"dt={coarse_dt:g} against dt={fine_dt:g}",
        "modes": modes.tolist(),
        "metrics": metrics,
        "checks": checks,
    }
    histories = {
        "time": time,
        "coarse_modes": coarse_modes,
        "fine_modes": fine_at_coarse,
        "coefficient_relative_error": coefficient_relative_error,
        "coarse_dt": coarse_dt,
        "fine_dt": fine_dt,
        "coarse_health": coarse_health,
        "fine_health": fine_health,
        "modes": modes,
        "ring_time": ring_target,
        "coarse_radius": coarse_radius,
        "fine_radius": fine_radius,
    }
    return result, histories


def plot(histories: dict, output: Path) -> None:
    time = histories["time"]
    coarse = histories["coarse_modes"]
    fine = histories["fine_modes"]
    coarse_envelope = np.sqrt(np.mean(coarse**2, axis=1))
    fine_envelope = np.sqrt(np.mean(fine**2, axis=1))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.4))

    axis = axes[0, 0]
    axis.plot(
        time,
        coarse_envelope,
        color=BLUE,
        label=histories["coarse_label"],
    )
    axis.plot(
        time,
        fine_envelope,
        color=GOLD,
        linestyle="--",
        label=histories["fine_label"],
    )
    axis.set_xlabel(r"$t^*=t\Gamma/R^2$")
    mode_label = ",".join(str(mode) for mode in histories["modes"])
    axis.set_ylabel(rf"RMS amplitude, $m={mode_label}$")
    axis.set_title("Selected-mode envelope")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.plot(time, 100.0 * histories["coefficient_relative_error"], color=BLUE)
    axis.axhline(5.0, color=INK, linestyle="--", label="5% gate")
    axis.set_xlabel(r"$t^*=t\Gamma/R^2$")
    axis.set_ylabel("complex modal difference (%)")
    axis.set_title(histories["sensitivity_title"])
    axis.legend(frameon=False)

    axis = axes[1, 0]
    axis.plot(histories["ring_time"], histories["coarse_radius"], color=BLUE, label="coarse")
    axis.plot(
        histories["ring_time"],
        histories["fine_radius"],
        color=GOLD,
        linestyle="--",
        label="fine",
    )
    axis.set_xlabel(r"$t^*=t\Gamma/R^2$")
    axis.set_ylabel(r"major radius $R(t)$")
    axis.set_title("Resolved ring geometry")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    for health, color, label in (
        (histories["coarse_health"], BLUE, "coarse"),
        (histories["fine_health"], GOLD, "fine"),
    ):
        axis.plot(
            health["time"].to_numpy(float) * CIRCULATION,
            health["vorticity_divergence_error"],
            color=color,
            label=label,
        )
    axis.axhline(0.12, color=INK, linestyle="--", label="health limit")
    axis.set_xlabel(r"$t^*=t\Gamma/R^2$")
    axis.set_ylabel("normalized divergence error")
    axis.set_title("Particle-resolution health")
    axis.legend(frameon=False)

    for axis in axes.flat:
        axis.grid(color=GRID, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.xaxis.label.set_size(10)
        axis.yaxis.label.set_size(10)
        axis.title.set_size(11)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle(histories["figure_title"], color=INK, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), pad=2.0, h_pad=3.0, w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-directory", type=Path, required=True)
    parser.add_argument("--fine-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--coarse-label")
    parser.add_argument("--fine-label")
    parser.add_argument("--comparison-label")
    parser.add_argument("--figure-title", default="Widnall VPM time-step gate")
    parser.add_argument("--sensitivity-title", default="Phase-sensitive time-step error")
    parser.add_argument(
        "--modes",
        type=int,
        nargs="+",
        default=DEFAULT_MODES.tolist(),
        help="Azimuthal modes included in the convergence gate.",
    )
    args = parser.parse_args()
    modes = np.asarray(args.modes, dtype=int)
    if len(modes) == 0 or np.any(modes < 1) or len(np.unique(modes)) != len(modes):
        parser.error("--modes must be unique positive integers")
    result, histories = evaluate(args.coarse_directory, args.fine_directory, modes)
    histories["coarse_label"] = args.coarse_label or rf"$\Delta t={histories['coarse_dt']:g}$"
    histories["fine_label"] = args.fine_label or rf"$\Delta t={histories['fine_dt']:g}$"
    histories["figure_title"] = args.figure_title
    histories["sensitivity_title"] = args.sensitivity_title
    if args.comparison_label:
        result["comparison"] = args.comparison_label
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(histories, args.figure)
    if result["status"] != "PASS":
        raise SystemExit("WIDNALL COMPARISON GATE FAIL")
    print("WIDNALL COMPARISON GATE PASS")


if __name__ == "__main__":
    main()
