#!/usr/bin/env python3
"""Every figure for the cylinder shedding validation experiment.

Requires the sampled histories of both cases (see analyse_von_karman.py for the
numerical report); figures only need ``samples/`` and the theme.

Usage:
    python assets/plot_von_karman.py --format png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import periodogram  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402
from _vonkarman import SHEDDING_BAND, Series, analyse_series, resample_uniform  # noqa: E402

FIGURE_DPI = 300
RADIUS = 0.5  # cylinder radius (D/2) for midspan masking


def _series(source: str) -> Series | None:
    probe = util.load_probe(source)
    forces = util.load_forces(source)
    if probe is None or forces is None:
        return None
    return Series(
        label=util.label(source),
        normalized_transverse_velocity_time=probe["t"],
        normalized_transverse_velocity=probe["normalized_transverse_velocity"],
        drag_coefficient_time=forces["t"],
        drag_coefficient=forces["drag_coefficient"],
        lift_coefficient_time=forces["t"],
        lift_coefficient=forces["lift_coefficient"],
    )


def plot_lift_history(figure_format: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), dpi=FIGURE_DPI, sharex=True)
    for source in ("reference", "fvm"):
        forces = util.load_forces(source)
        if forces is None:
            continue
        colour = util.colour(source)
        axes[0].plot(
            forces["t"],
            forces["drag_coefficient"],
            label=util.label(source),
            color=colour,
        )
        axes[1].plot(
            forces["t"],
            forces["lift_coefficient"],
            label=util.label(source),
            color=colour,
            lw=0.8,
        )
    axes[0].set(ylabel=r"$C_D$", title="Drag coefficient history")
    axes[1].set(ylabel=r"$C_L$", xlabel=r"$t\,U_\infty/D$", title="Lift coefficient history")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    util.save(fig, "lift_history", figure_format, FIGURE_DPI)
    plt.close(fig)


def plot_probe_growth(figure_format: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=FIGURE_DPI)
    for source in ("reference", "fvm"):
        probe = util.load_probe(source)
        if probe is None:
            continue
        ax.plot(
            probe["t"],
            probe["normalized_transverse_velocity"],
            label=util.label(source),
            color=util.colour(source),
            lw=0.8,
        )
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
    ax.set(
        xlabel=r"$t\,U_\infty/D$",
        ylabel=r"$q = u_y/U_\infty$ at $x/D=1.5,\,z=0$",
        title="Midspan wake probe: antisymmetric-mode growth",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    util.save(fig, "midspan_probe_growth", figure_format, FIGURE_DPI)
    plt.close(fig)


def plot_growth_fit(figure_format: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=FIGURE_DPI)
    for source in ("reference", "fvm"):
        series = _series(source)
        if series is None:
            continue
        result = analyse_series(series)
        colour = util.colour(source)
        env = np.maximum(result.amplitude, 1e-8)
        ax.semilogy(
            result.envelope_time, env, color=colour, lw=1.0, label=f"{result.label} envelope"
        )
        if result.peak_time.size:
            ax.semilogy(
                result.peak_time,
                np.maximum(result.peak_amplitude, 1e-8),
                "o",
                color=colour,
                ms=2.5,
                mfc="none",
            )
        g = result.growth
        if np.isfinite(g["growth_rate"]):
            t_fit = result.envelope_time[
                (result.envelope_time >= g["fit_start_time"])
                & (result.envelope_time <= g["fit_end_time"])
            ]
            if t_fit.size:
                y_fit = g["initial_amplitude"] * np.exp(g["growth_rate"] * t_fit)
                ax.semilogy(
                    t_fit,
                    y_fit,
                    "--",
                    color=colour,
                    lw=1.4,
                    label=f"{result.label} fit $\\sigma={g['growth_rate']:.3f}$, $A_0={g['initial_amplitude']:.2e}$",
                )
    ax.axhline(0.25, color="k", lw=0.6, ls=":", label=r"$A^*=0.25$")
    ax.set(
        xlabel=r"$t\,U_\infty/D$",
        ylabel=r"envelope amplitude $A$ (log)",
        title="Exponential growth: $\\ln A = \\ln A_0 + \\sigma t$",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    util.save(fig, "linear_growth_fit", figure_format, FIGURE_DPI)
    plt.close(fig)


def plot_onset_alignment(figure_format: str) -> None:
    reference = _series("reference")
    hybrid = _series("fvm")
    if reference is None or hybrid is None:
        raise SystemExit("onset_alignment requires both cases to have run")
    ref = analyse_series(reference)
    hyb = analyse_series(hybrid)
    mean_growth_rate = 0.5 * (ref.growth["growth_rate"] + hyb.growth["growth_rate"])
    predicted_onset_time_shift = (
        (1.0 / mean_growth_rate)
        * np.log(hyb.growth["initial_amplitude"] / ref.growth["initial_amplitude"])
        if np.isfinite(mean_growth_rate)
        and mean_growth_rate > 0
        and ref.growth["initial_amplitude"] > 0
        else np.nan
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.6), dpi=FIGURE_DPI, sharex=True)
    window = (
        np.nanmin([ref.onset_time, hyb.onset_time]) - 10.0,
        np.nanmax([ref.onset_time, hyb.onset_time]) + 10.0,
    )

    ax = axes[0]
    for series, result in ((reference, ref), (hybrid, hyb)):
        mask = (series.normalized_transverse_velocity_time >= window[0]) & (
            series.normalized_transverse_velocity_time <= window[1]
        )
        ax.plot(
            series.normalized_transverse_velocity_time[mask],
            series.normalized_transverse_velocity[mask],
            color=util.colour("reference" if result is ref else "fvm"),
            lw=0.9,
            label=result.label,
        )
        ax.axvline(
            result.onset_time,
            color=util.colour("reference" if result is ref else "fvm"),
            ls=":",
            lw=1.0,
        )
    ax.set(title="Raw onset: reference and hybrid reach $A^*$ at different times", ylabel=r"$q$")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    mask = (reference.normalized_transverse_velocity_time >= window[0]) & (
        reference.normalized_transverse_velocity_time <= window[1]
    )
    ax.plot(
        reference.normalized_transverse_velocity_time[mask],
        reference.normalized_transverse_velocity[mask],
        color=util.colour("reference"),
        lw=0.9,
        label="Reference FVM",
    )
    if np.isfinite(predicted_onset_time_shift):
        ax.plot(
            hybrid.normalized_transverse_velocity_time[mask] + predicted_onset_time_shift,
            hybrid.normalized_transverse_velocity[mask],
            color=util.colour("fvm"),
            lw=0.9,
            label=f"Coupled FVM shifted by $\\Delta t_\\mathrm{{pred}}={predicted_onset_time_shift:.2f}$ s",
        )
    ax.axvline(
        ref.onset_time
        + (predicted_onset_time_shift if np.isfinite(predicted_onset_time_shift) else 0.0),
        color=util.colour("reference"),
        ls=":",
        lw=1.0,
    )
    ax.set(
        title="Aligned by the amplitude prediction: onset collapse tests the hypothesis",
        xlabel=r"$t\,U_\infty/D$",
        ylabel=r"$q$",
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    util.save(fig, "onset_alignment", figure_format, FIGURE_DPI)
    plt.close(fig)


def plot_spectrum(figure_format: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=FIGURE_DPI)
    for source in ("reference", "fvm"):
        series = _series(source)
        if series is None:
            continue
        result = analyse_series(series)
        lo = result.saturated["start_time"]
        if not np.isfinite(lo):
            continue
        mask = series.normalized_transverse_velocity_time >= lo
        t_u, uniform_normalized_transverse_velocity = resample_uniform(
            series.normalized_transverse_velocity_time[mask],
            series.normalized_transverse_velocity[mask],
        )
        if t_u.size < 64:
            continue
        fs = 1.0 / float(np.median(np.diff(t_u)))
        freq, power = periodogram(
            uniform_normalized_transverse_velocity
            - np.mean(uniform_normalized_transverse_velocity),
            fs=fs,
        )
        band = (freq >= SHEDDING_BAND[0]) & (freq <= SHEDDING_BAND[1])
        ax.semilogy(freq[band], power[band], color=util.colour(source), lw=1.2, label=result.label)
        pk = np.argmax(power[band])
        ax.axvline(freq[band][pk], color=util.colour(source), ls=":", lw=1.0)
    ax.set(
        xlabel=r"$f\,D/U_\infty$",
        ylabel="PSD",
        title="Saturated-window spectrum (probe $q$)",
        xlim=SHEDDING_BAND,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    util.save(fig, "shedding_spectrum", figure_format, FIGURE_DPI)
    plt.close(fig)


def plot_spanwise_coherence(figure_format: str) -> None:
    times = np.asarray([t for t in util.slice_times("reference") if t > 35.0])
    if times.size == 0:
        raise SystemExit("spanwise_coherence needs reference frames past t=35 s")
    picks = times[:: max(1, times.size // 5)][:3]

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.6), dpi=FIGURE_DPI, sharex=True)
    ax = axes[0]
    for source in ("reference", "fvm"):
        frame = util.load_line(source, "spanwise_line", picks[-1])
        if frame is None:
            continue
        ax.plot(
            frame["position_z"],
            frame["velocity_y"],
            color=util.colour(source),
            lw=0.9,
            label=util.label(source),
        )
    ax.axvspan(-5.0, 5.0, color=util.COLORS["box"], alpha=0.35, label="cylinder span")
    ax.set(ylabel=r"$u_y/U_\infty$", title="Midspan coherence: $u_y(z)$ at $x/D=1.5$")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    for source in ("reference", "fvm"):
        table = util._read_line_csv(util._path(source, "spanwise_line", ".csv"))
        if table is None or table["time"].size == 0:
            continue
        t_late = table["time"] > 35.0
        uy = table["velocity_y"][t_late]
        rms = np.sqrt(np.mean(uy**2))
        for sample_time in np.unique(table["time"][t_late]):
            frame = util.load_line(source, "spanwise_line", sample_time)
            if frame is None:
                continue
            ax.plot(
                frame["position_z"],
                frame["velocity_y"] / (rms + 1e-30),
                color=util.colour(source),
                lw=0.5,
                alpha=0.35,
                label=None,
            )
    ax.set(
        xlabel=r"$z/D$",
        ylabel=r"$u_y/\mathrm{rms}(u_y)$",
        title="Spanwise profiles, saturated window (all frames overlaid)",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    util.save(fig, "spanwise_coherence", figure_format, FIGURE_DPI)
    plt.close(fig)


def plot_midspan(figure_format: str, times: tuple[float, ...]) -> None:
    for time in times:
        frames = {}
        for source in ("reference", "fvm", "vpm"):
            frames[source] = util.load_slice(source, time)
        if any(frames[s] is None for s in frames):
            print(f"  skip vorticity_midspan_t{int(round(time)):02d}: missing slice frames")
            continue
        vmax = 0.0
        for source, frame in frames.items():
            vorticity = np.abs(frame["vorticity_z"])
            vmax = max(vmax, float(np.nanmax(vorticity[np.isfinite(vorticity)])))
        levels = np.linspace(-vmax, vmax, 33)

        fig, axes = plt.subplots(
            1, 3, figsize=(12.0, 3.6), dpi=FIGURE_DPI, sharex=True, sharey=True
        )
        for ax, source in zip(axes, ("reference", "fvm", "vpm")):
            frame = frames[source]
            cy = frame["y"] ** 2 + frame["x"] ** 2 < RADIUS**2
            masked = np.ma.masked_where(cy, frame["vorticity_z"])
            cf = ax.contourf(
                frame["x"],
                frame["y"],
                masked,
                levels=levels,
                cmap=util.COLORMAPS["vorticity"],
                extend="both",
            )
            ax.add_patch(plt.Circle((0.0, 0.0), RADIUS, color="0.9", ec="k", lw=0.8, zorder=3))
            ax.set_aspect("equal")
            ax.set_title(util.label(source), fontsize=9)
            if source == "reference":
                ax.set_ylabel(r"$y/D$")
            ax.set_xlabel(r"$x/D$")
        fig.colorbar(
            cf, ax=axes, orientation="horizontal", fraction=0.03, pad=0.14, label=r"$\omega_z$"
        )
        fig.suptitle(
            rf"Midspan vorticity at $t\,U_\infty/D={time:g}$ — identical levels "
            r"($\pm$" + f"{vmax:.2f})",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        util.save(fig, f"vorticity_midspan_t{int(round(time)):02d}", figure_format, FIGURE_DPI)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="png", choices=("png", "pdf"))
    args = parser.parse_args()

    plot_lift_history(args.format)
    plot_probe_growth(args.format)
    plot_growth_fit(args.format)
    plot_onset_alignment(args.format)
    plot_spectrum(args.format)
    plot_spanwise_coherence(args.format)
    times = [t for t in util.slice_times("vpm", "slice_z0") if t > 35.0]
    if not times:
        times = [t for t in util.slice_times("reference") if t > 35.0]
    plot_midspan(args.format, tuple(times[-1:]) if times else ())
    print("All figures written.")


if __name__ == "__main__":
    main()
