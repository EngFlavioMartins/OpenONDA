#!/usr/bin/env python3
"""Plot streamwise and cross-wake cylinder velocity profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

STYLES = {
    "reference": dict(color=util.COLORS["reference"], ls="-.", label="Reference FVM"),
    "fvm": dict(color=util.COLORS["hybrid"], ls="-", label="Hybrid FVM"),
    "vpm": dict(color=util.COLORS["vpm"], ls="--", label="Hybrid VPM"),
}


def _profile(axis, source: str, name: str, time: float, coordinate: str) -> None:
    frame = util.load_line(source, name, time)
    if frame is None:
        return
    abscissa = np.asarray(frame[coordinate], dtype=float)
    order = np.argsort(abscissa)
    axis.plot(abscissa[order], np.asarray(frame["Ux"])[order], **STYLES[source])


def plot_frame(time: float, fmt: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)
    panels = (
        ("centerline", "x", "Centerline", (-2.0, 8.0)),
        ("offaxis_y075", "x", r"Off-axis $y=0.75D$", (-2.0, 8.0)),
        ("section_x100", "y", r"Wake section $x=D$", (-2.0, 2.0)),
        ("section_x200", "y", r"Wake section $x=2D$", (-2.0, 2.0)),
    )
    for axis, (name, coordinate, title, limits) in zip(axes.flat, panels, strict=True):
        for source in ("reference", "fvm", "vpm"):
            _profile(axis, source, name, time, coordinate)
        axis.set(
            xlabel=rf"${coordinate}/D$",
            ylabel=r"$u_x/U_\infty$",
            xlim=limits,
            title=title,
        )
        axis.grid(alpha=0.18)
    axes[0, 0].add_patch(plt.Circle((0.0, 0.0), 0.5, color=util.COLORS["background_strong"]))
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.suptitle(rf"Cylinder velocity profiles, $tU_\infty/D={time:.1f}$")
    util.save(fig, f"velocity_profiles_t{time:.1f}", fmt)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    available = util.common_times(
        util.line_times("reference", "centerline"),
        util.line_times("fvm", "centerline"),
        util.line_times("vpm", "centerline"),
        util.line_times("reference", "section_x200"),
        util.line_times("fvm", "section_x200"),
        util.line_times("vpm", "section_x200"),
    )
    for time in util.plot_times(available):
        plot_frame(float(time), args.format)
    if not len(available):
        raise SystemExit("No coincident cylinder profile samples were found.")


if __name__ == "__main__":
    main()
