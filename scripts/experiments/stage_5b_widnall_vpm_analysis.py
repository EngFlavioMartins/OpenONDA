#!/usr/bin/env python3
"""Measure a vortex-ring Widnall challenge from restartable VPM states.

The raw particle field is converted into a strength-weighted ring centreline
and decomposed into radial and axial azimuthal Fourier modes.  The prescribed
broadband seed provides an exact amplitude reference, while

    m_peak ~= 2.26 R/a

is retained only as the published Gaussian-core estimate of the most amplified
mode.  No theoretical growth-rate curve is drawn because the current Gaussian
core-radius convention has not yet been mapped unambiguously to the radii in
the linear-stability papers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/openonda_stage5b_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/openonda_stage5b_cache")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RING_ASSETS = ROOT / "tutorials" / "VPM" / "vortexRing" / "assets"
sys.path.insert(0, str(RING_ASSETS))

from ring_diagnostics import RingModeDiagnosticsSampler  # noqa: E402

RADIUS = 1.0
CIRCULATION = np.pi
CORE_RADIUS = 0.1
INK = "#20252a"
BLUE = "#286f9b"
GOLD = "#d9973b"
GREY = "#8a99a8"
GRID = "#d8dde2"


def read_modes(run_directory: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(run_directory.glob(f"vpm_{prefix}_[0-9][0-9][0-9][0-9][0-9][0-9].h5"))
    if not files:
        raise FileNotFoundError(f"no numbered HDF5 states found for prefix {prefix!r}")

    sampler = RingModeDiagnosticsSampler(
        maximum_mode=40,
        azimuthal_bins=128,
        reference_radius=RADIUS,
        transverse_origin=(0.0, 0.0),
    )
    times: list[float] = []
    radial: list[np.ndarray] = []
    axial: list[np.ndarray] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            position = np.asarray(handle["particles/position"], dtype=np.float64)
            strength = np.asarray(handle["particles/circulation"], dtype=np.float64)
            group = np.asarray(handle["particles/group_id"], dtype=np.int32)
            time = float(handle["solver"].attrs["flow_time"])
        selected = group == np.unique(group)[0]
        rows = sampler._sample_group(position[selected], strength[selected])
        if len(rows) != sampler.maximum_mode:
            raise RuntimeError(f"mode reconstruction failed for {path}")
        rows_array = np.asarray(rows, dtype=float)
        times.append(time * CIRCULATION / RADIUS**2)
        radial.append(rows_array[:, 1])
        axial.append(rows_array[:, 2])
    return np.asarray(times), np.asarray(radial), np.asarray(axial)


def read_health(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    data = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    if "time" not in data:
        return None
    return data.dropna(subset=["time"]).drop_duplicates("step", keep="last")


def read_ring(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    data = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    if "flow_time" not in data:
        return None
    return data.dropna(subset=["flow_time"]).drop_duplicates("time_step", keep="last")


def evaluate(
    time: np.ndarray,
    radial: np.ndarray,
    axial: np.ndarray,
    epsilon: float,
    seeded_modes: int,
    health: pd.DataFrame | None,
    ring: pd.DataFrame | None,
) -> dict[str, object]:
    combined = np.hypot(radial, axial)
    seed_amplitude = epsilon / np.sqrt(seeded_modes)
    seeded = slice(0, seeded_modes)
    predicted_mode = 2.26 * RADIUS / CORE_RADIUS
    predicted_indices = np.arange(19, 24)  # modes 20--24
    first = combined[0]
    safe_first = np.maximum(first, np.finfo(float).tiny)
    growth = combined / safe_first
    result: dict[str, object] = {
        "raw_state_count": len(time),
        "earliest_time_star": float(time[0]),
        "latest_time_star": float(time[-1]),
        "widnall_amplitude": epsilon,
        "seeded_modes": seeded_modes,
        "theoretical_seed_amplitude_per_radial_mode": seed_amplitude,
        "theoretical_gaussian_peak_mode_estimate": predicted_mode,
        "earliest_radial_seed_relative_l2": float(
            np.linalg.norm(radial[0, seeded] - seed_amplitude)
            / (np.sqrt(seeded_modes) * seed_amplitude)
        ),
        "earliest_dominant_seeded_mode": int(np.argmax(combined[0, seeded]) + 1),
        "latest_dominant_seeded_mode": int(np.argmax(combined[-1, seeded]) + 1),
        "largest_growth_modes_20_to_24": float(np.max(growth[:, predicted_indices])),
        "latest_growth_modes_20_to_24": float(np.max(growth[-1, predicted_indices])),
        "largest_amplitude_modes_20_to_24_over_prescribed_seed": float(
            np.max(combined[:, predicted_indices]) / seed_amplitude
        ),
        "latest_amplitude_modes_20_to_24_over_prescribed_seed": float(
            np.max(combined[-1, predicted_indices]) / seed_amplitude
        ),
        "latest_unseeded_to_seeded_rms": float(
            np.sqrt(np.mean(combined[-1, seeded_modes:] ** 2))
            / np.sqrt(np.mean(combined[-1, seeded] ** 2))
        ),
        "theory_limit": (
            "Only the seed spectrum and m_peak estimate are overlaid. A growth-rate "
            "overlay is withheld until Gaussian core-radius definitions are mapped."
        ),
    }
    if health is not None:
        result["particle_health"] = {
            "maximum_divergence_error": float(health["vorticity_divergence_error"].max()),
            "maximum_misalignment_deg": float(health["strength_misalignment_deg"].max()),
            "maximum_peak_strength_ratio": float(
                health["max_gamma"].max() / health["max_gamma"].iloc[0]
            ),
            "divergence_limit": 0.12,
            "misalignment_limit_deg": 45.0,
        }
    if ring is not None:
        impulse = ring["impulse_norm"].to_numpy(float)
        circulation = ring["tube_circulation"].to_numpy(float)
        result["conservation"] = {
            "maximum_impulse_drift": float(np.max(np.abs(impulse / impulse[0] - 1.0))),
            "final_impulse_drift": float(abs(impulse[-1] / impulse[0] - 1.0)),
            "maximum_tube_circulation_drift": float(
                np.max(np.abs(circulation / circulation[0] - 1.0))
            ),
            "final_tube_circulation_drift": float(abs(circulation[-1] / circulation[0] - 1.0)),
        }
    return result


def plot(
    time: np.ndarray,
    radial: np.ndarray,
    axial: np.ndarray,
    epsilon: float,
    seeded_modes: int,
    health: pd.DataFrame | None,
    output: Path,
    label: str,
) -> None:
    modes = np.arange(1, radial.shape[1] + 1)
    combined = np.hypot(radial, axial)
    seed_amplitude = epsilon / np.sqrt(seeded_modes)
    predicted_mode = 2.26 * RADIUS / CORE_RADIUS
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.7), constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(modes, radial[0], color=BLUE, marker="o", ms=3, label="measured radial")
    axis.plot(modes, axial[0], color=GOLD, marker="s", ms=3, label="measured axial")
    axis.hlines(
        seed_amplitude,
        1,
        seeded_modes,
        color=INK,
        linestyle="--",
        label=r"prescribed $\epsilon_W/\sqrt{M}$",
    )
    axis.axvline(predicted_mode, color=GREY, linestyle=":", label=r"$m\simeq2.26R/a$")
    axis.set_yscale("log")
    axis.set_xlabel("azimuthal mode $m$")
    axis.set_ylabel("centreline amplitude / $R$")
    axis.set_title(f"Earliest raw state, $t^*={time[0]:.2f}$")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    image = axis.pcolormesh(
        time,
        modes,
        np.log10(np.maximum(combined.T, 1.0e-8)),
        shading="auto",
        cmap="viridis",
    )
    axis.axhline(predicted_mode, color="white", linestyle=":", linewidth=1.2)
    axis.set_xlabel(r"normalized time $t^*=t\Gamma/R^2$")
    axis.set_ylabel("azimuthal mode $m$")
    axis.set_title("Bending-mode amplitude")
    fig.colorbar(image, ax=axis, label=r"$\log_{10}(A_m/R)$")

    axis = axes[1, 0]
    for mode, color in zip(range(20, 25), plt.cm.viridis(np.linspace(0.15, 0.9, 5)), strict=True):
        axis.plot(time, combined[:, mode - 1], color=color, label=f"m={mode}")
    axis.axhline(
        seed_amplitude,
        color=INK,
        linestyle="--",
        linewidth=0.9,
        label=r"prescribed $t=0$ amplitude",
    )
    axis.set_xlabel(r"normalized time $t^*=t\Gamma/R^2$")
    axis.set_ylabel("centreline amplitude / $R$")
    axis.set_title("Modes around the Gaussian-ring prediction")
    axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes[1, 1]
    if health is not None:
        health_time = health["time"].to_numpy(float) * CIRCULATION / RADIUS**2
        axis.plot(
            health_time,
            health["vorticity_divergence_error"],
            color=BLUE,
            label="divergence error",
        )
        axis.axhline(0.12, color=BLUE, linestyle="--", linewidth=0.9, label="divergence limit")
        twin = axis.twinx()
        twin.plot(
            health_time,
            health["strength_misalignment_deg"],
            color=GOLD,
            label="misalignment",
        )
        twin.axhline(45.0, color=GOLD, linestyle="--", linewidth=0.9, label="alignment limit")
        twin.set_ylabel("misalignment (degrees)", color=GOLD)
        handles, labels = axis.get_legend_handles_labels()
        handles2, labels2 = twin.get_legend_handles_labels()
        axis.legend(handles + handles2, labels + labels2, frameon=False, fontsize=7, ncol=2)
    else:
        axis.text(0.5, 0.5, "particle-health CSV unavailable", ha="center", va="center")
    axis.set_xlabel(r"normalized time $t^*=t\Gamma/R^2$")
    axis.set_ylabel("normalized divergence error", color=BLUE)
    axis.set_title("Particle-resolution health")

    for axis in axes.flat:
        axis.grid(color=GRID, linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"VPM Widnall challenge — {label}", fontsize=14, color=INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seeded-modes", type=int, default=24)
    parser.add_argument("--health-csv", type=Path)
    parser.add_argument("--ring-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    time, radial, axial = read_modes(args.run_directory, args.prefix)
    health = read_health(args.health_csv)
    ring = read_ring(args.ring_csv)
    result = evaluate(time, radial, axial, args.epsilon, args.seeded_modes, health, ring)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(time, radial, axial, args.epsilon, args.seeded_modes, health, args.figure, args.label)


if __name__ == "__main__":
    main()
