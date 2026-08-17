#!/usr/bin/env python3
"""Plot coupled-run timing, particle population, and handoff diagnostics."""

from pathlib import Path
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

FIGURE_DPI = 300


def _records() -> list[dict]:
    path = util.SOLUTION / "coupler_diagnostics.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A live writer may leave one temporarily incomplete final line.
                continue
    return records


def _values(records: list[dict], section: str, key: str) -> np.ndarray:
    return np.asarray([row.get(section, {}).get(key, 0.0) for row in records], dtype=float)


def plot(figure_format: str) -> None:
    records = _records()
    if not records:
        raise SystemExit("No coupling diagnostics found in solution/.")

    time = np.asarray([row["time"] for row in records], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=FIGURE_DPI, sharex=True)

    timing = axes[0, 0]
    vpm = _values(records, "timing_seconds", "vpm")
    fvm = _values(records, "timing_seconds", "fvm")
    transfer = sum(
        (_values(records, "timing_seconds", name) for name in ("donor", "fringe", "handoff")),
        start=np.zeros_like(time),
    )
    timing.stackplot(
        time,
        vpm,
        fvm,
        transfer,
        labels=("VPM", "FVM", "transfer"),
        colors=(util.COLORS["vpm"], util.COLORS["fvm"], util.COLORS["accent"]),
        alpha=0.85,
    )
    timing.set(ylabel="wall time / step [s]", title="Coupling-step cost")
    timing.legend(loc="upper left", ncol=3, fontsize=7)

    population = axes[0, 1]
    population.plot(
        time,
        np.asarray([row.get("handoff_particle_count", 0) for row in records]) / 1e6,
        color=util.COLORS["vpm"],
        label="total",
    )
    for key, label, style in (
        ("n_remesh_out", "overlap/buffer", "-"),
        ("n_free", "free wake", "--"),
    ):
        population.plot(
            time,
            _values(records, "handoff", key) / 1e6,
            linestyle=style,
            label=label,
        )
    population.set(ylabel="particles [million]", title="Particle population")
    population.legend(loc="upper left", fontsize=7)

    # Transfer fidelity.  The in-band residual says whether the particles
    # reproduce the band they claim to carry (a bug if it is not round-off);
    # the out-of-band fraction says how much of the FVM's vorticity is finer
    # than the lattice (a resolution limit -- refine h, do not deconvolve
    # harder).  Keeping them on the same axes stops the second being mistaken
    # for the first, which is what the single scalar residual invited.
    fidelity = axes[1, 0]
    in_band = 100.0 * _values(records, "handoff", "transfer_in_band_residual")
    out_of_band = 100.0 * _values(records, "handoff", "transfer_out_of_band_fraction")
    fidelity.semilogy(
        time, np.maximum(in_band, 1e-14), color=util.COLORS["fvm"], label="in-band residual"
    )
    fidelity.semilogy(
        time, np.maximum(out_of_band, 1e-14), color=util.COLORS["vpm"], label="out-of-band (h limit)"
    )
    fidelity.axhline(1.0, color="0.6", lw=0.8, ls=":", label="1%")
    fidelity.set(xlabel="flow time [s]", ylabel="[%]", title="Transfer fidelity")
    fidelity.legend(loc="upper right", fontsize=7)

    # Per-band |omega_VPM| / |omega_FVM|.  Every curve should sit on 1.
    quality = axes[1, 1]
    band_names = sorted(
        {name for row in records for name in row.get("spectral_band_ratio", {})},
        key=lambda s: float(s.split("-")[0]),
    )
    for name in band_names:
        quality.plot(
            time,
            np.asarray(
                [row.get("spectral_band_ratio", {}).get(name, np.nan) for row in records],
                dtype=float,
            ),
            label=rf"$\lambda$ = {name}",
        )
    quality.axhline(1.0, color="0.6", lw=0.8, ls=":")
    if not band_names:
        quality.semilogy(
            time,
            np.maximum(100.0 * _values(records, "handoff", "pruned_circulation_fraction"), 1e-8),
            label=r"pruned $\Sigma|\Gamma|$ [%]",
        )
    quality.set(
        xlabel="flow time [s]",
        ylabel=r"$|\omega_{VPM}| / |\omega_{FVM}|$",
        title="Spectral agreement",
    )
    quality.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    util.save(fig, "coupling_diagnostics", figure_format, FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    plot(args.format)


if __name__ == "__main__":
    main()
