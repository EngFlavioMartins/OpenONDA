#!/usr/bin/env python3
"""Plot native y=0 cross sections for seeded-breakdown comparisons.

Unlike the meridional half-plane plot, this view retains both signs of z and
therefore exposes reflection-odd deformation.  It reads SurfaceSampler VTS/PVD
files only; no particle-field reconstruction or time interpolation is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pyvista as pv

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .. import setup
from .ring_metrics import _theme


ROOT = Path(__file__).resolve().parents[1]
OMEGA_ZERO = setup.RING_CIRCULATION / (np.pi * setup.CORE_RADIUS**2)


def _samples_at(run: str) -> dict[float, Path]:
    directory = ROOT / "samples" / run
    index = directory / "cross_section.pvd"
    if not index.is_file():
        raise FileNotFoundError(f"missing native cross-section index: {index}")
    return {
        float(dataset.attrib["timestep"]): directory / dataset.attrib["file"]
        for dataset in ET.parse(index).findall(".//DataSet")
    }


def _read(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = pv.read(path)
    points = np.asarray(grid.points)
    if not np.allclose(points[:, 1], 0.0, atol=1.0e-7):
        raise ValueError(f"expected native y=0 plane: {path}")
    x, ix = np.unique(points[:, 0], return_inverse=True)
    z, iz = np.unique(points[:, 2], return_inverse=True)
    if len(points) != len(x) * len(z):
        raise ValueError(f"incomplete native rectangular plane: {path}")
    omega_y = np.empty((len(x), len(z)))
    omega_y[ix, iz] = np.asarray(grid.point_data["vorticity"])[:, 1] / OMEGA_ZERO
    if not np.isfinite(omega_y).all():
        raise ValueError(f"non-finite native cross-section vorticity: {path}")
    return x, z, omega_y


def render(runs: list[str], time: float, output: Path) -> None:
    theme = _theme()
    theme.set_thesis_style()
    cmap = LinearSegmentedColormap.from_list(
        "openonda_signed_vorticity",
        [theme.PALETTE["purple"], "white", theme.PALETTE["teal"]],
    )
    records = []
    for run in runs:
        samples = _samples_at(run)
        matched = [path for timestamp, path in samples.items() if np.isclose(timestamp, time)]
        if len(matched) != 1:
            raise ValueError(f"{run} has {len(matched)} native cross sections at t={time}")
        records.append((run, *_read(matched[0])))

    maximum = max(float(np.max(np.abs(field))) for _, _, _, field in records)
    maximum = max(maximum, np.finfo(float).tiny)
    active_x = []
    for _, x, _, field in records:
        active = np.any(np.abs(field) >= 0.01 * maximum, axis=1)
        active_x.extend(x[active])
    x_limits = (float(np.min(active_x)) - 0.45, float(np.max(active_x)) + 0.45)
    levels = np.linspace(-maximum, maximum, 81)
    figure, axes = plt.subplots(
        1, len(records), figsize=(5.0 * len(records), 5.2), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    labels = {
        "cs_breakdown_baseline": "Baseline\n" + r"$h/R_0=.06$",
        "cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation": (
            "Fixed-core coverage\n" + r"$h/R_0=.05$, $\sigma_0/R_0=.06$"
        ),
        "cs_breakdown_p_moments_cpu_t6_qualification": (
            "Moment-preserving realignment\n" + r"$h/R_0=.06$"
        ),
        "cs_breakdown_filter_cs020_cpu_t6_step080": ("Control\n" + r"$C_s=.20$, $h/R_0=.06$"),
        "cs_breakdown_filter_cs000_cpu_t6_step080": ("Molecular\n" + r"$C_s=0$, $h/R_0=.06$"),
        "cs_breakdown_filter_cs020_cpu_t6_step300_continuation": (
            "Control\n" + r"$C_s=.20$, $h/R_0=.06$"
        ),
        "cs_breakdown_filter_cs000_cpu_t6_step300_continuation": (
            "Molecular\n" + r"$C_s=0$, $h/R_0=.06$"
        ),
        "cs_breakdown_filter_cs020_cpu_t6_step300_tail": ("Control\n" + r"$C_s=.20$, $h/R_0=.06$"),
        "cs_breakdown_filter_cs020_cpu_t6_step320_discriminator": (
            "Control\n" + r"$C_s=.20$, $h/R_0=.06$"
        ),
        "cs_breakdown_filter_cs000_cpu_t6_step320_discriminator": (
            "Molecular\n" + r"$C_s=0$, $h/R_0=.06$"
        ),
    }
    image = None
    for axis, (run, x, z, field) in zip(axes, records, strict=True):
        image = axis.contourf(x, z, field.T, levels=levels, cmap=cmap, extend="both")
        nonzero = np.r_[-np.array([0.01, 0.03, 0.1])[::-1], [0.01, 0.03, 0.1]]
        usable = nonzero[(nonzero > field.min()) & (nonzero < field.max())]
        if len(usable):
            axis.contour(
                x,
                z,
                field.T,
                levels=usable,
                colors=theme.PALETTE["dark"],
                linewidths=0.5,
            )
        axis.set(ylabel=r"$z/R_0$", xlim=x_limits, ylim=(-1.8, 1.8), aspect="equal")
        axis.set_title(labels.get(run, run.replace("_", " ")))
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set(xlabel=r"$x/R_0$")
    assert image is not None
    color_axis = figure.add_axes([0.88, 0.21, 0.025, 0.54])
    figure.colorbar(image, cax=color_axis, label=r"$\omega_y/\omega_0$")
    figure.suptitle(
        rf"Seeded ring interaction (breakdown scenario), Re=3415: "
        rf"$t\Gamma_0/R_0^2={np.pi * time:.2f}$"
    )
    figure.subplots_adjust(left=0.08, right=0.84, bottom=0.13, top=0.79, wspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.runs, args.time, args.output)


if __name__ == "__main__":
    main()
