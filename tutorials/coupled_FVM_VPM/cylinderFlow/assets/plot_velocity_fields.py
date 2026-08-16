#!/usr/bin/env python3
"""Plot reference, hybrid, and VPM cylinder-wake velocity fields."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402


def _on_grid(source: dict, target: dict, key: str) -> np.ndarray:
    points = np.column_stack((source["x"].ravel(), source["y"].ravel()))
    result = griddata(points, source[key].ravel(), (target["x"], target["y"]), method="linear")
    missing = ~np.isfinite(result)
    if np.any(missing):
        result[missing] = griddata(
            points,
            source[key].ravel(),
            (target["x"][missing], target["y"][missing]),
            method="nearest",
        )
    return result


def _body(axis) -> None:
    axis.add_patch(
        plt.Circle(
            (0.0, 0.0),
            0.5,
            facecolor=util.COLORS["background_strong"],
            edgecolor=util.COLORS["DarkText"],
            lw=0.5,
        )
    )


def plot_frame(time: float, fmt: str) -> None:
    reference = util.load_slice("reference", time)
    fvm = util.load_slice("fvm", time)
    vpm = util.load_slice("vpm", time)
    if reference is None or fvm is None or vpm is None:
        return
    ref = _on_grid(reference, fvm, "Ux")
    particle = _on_grid(vpm, fvm, "Ux")
    near = np.asarray(fvm["Ux"])
    body = fvm["x"] ** 2 + fvm["y"] ** 2 <= 0.55**2
    ref_error = 100.0 * np.abs(near - ref)
    representation_error = 100.0 * np.abs(near - particle)
    ref_error[body] = np.nan
    representation_error[body] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.5), constrained_layout=True)
    velocity = (ref, near, particle)
    titles = ("Reference FVM", "Hybrid FVM", "Hybrid VPM")
    vmin = min(float(np.nanmin(field)) for field in velocity)
    vmax = max(float(np.nanmax(field)) for field in velocity)
    levels = np.linspace(vmin, vmax, 41)
    image = None
    for axis, field, title in zip(axes[0], velocity, titles, strict=True):
        image = axis.contourf(
            fvm["x"], fvm["y"], field, levels=levels, cmap=util.COLORMAPS["velocity"], extend="both"
        )
        axis.set_title(title)
    error_fields = (ref_error, representation_error)
    error_titles = (r"$|u_x^{FVM}-u_x^{ref}|$ [\%]", r"$|u_x^{FVM}-u_x^{VPM}|$ [\%]")
    p95 = max(float(np.nanpercentile(field, 95)) for field in error_fields)
    error_image = None
    for axis, field, title in zip(axes[1, :2], error_fields, error_titles, strict=True):
        error_image = axis.pcolormesh(
            fvm["x"],
            fvm["y"],
            field,
            shading="auto",
            cmap=util.COLORMAPS["error"],
            vmin=0,
            vmax=p95,
        )
        axis.set_title(title)
    axes[1, 2].axis("off")
    for axis in list(axes.flat)[:5]:
        axis.set(aspect="equal", xlim=(-1.6, 3.04), ylim=(-2.0, 2.0), xlabel=r"$x/D$")
        _body(axis)
    axes[0, 0].set_ylabel(r"$y/D$")
    axes[1, 0].set_ylabel(r"$y/D$")
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label=r"$u_x/U_\infty$")
    if error_image is not None:
        fig.colorbar(error_image, ax=axes[1, :2].tolist(), shrink=0.8, label="absolute error [%]")
    fig.suptitle(rf"Cylinder wake, $tU_\infty/D={time:.1f}$")
    util.save(fig, f"velocity_fields_t{time:.1f}", fmt)
    plt.close(fig)
    print(
        f"  t={time:.1f}: FVM-reference RMS={np.sqrt(np.nanmean(ref_error**2)):.3f}%, "
        f"FVM-VPM RMS={np.sqrt(np.nanmean(representation_error**2)):.3f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    available = util.common_times(
        util.slice_times("reference"), util.slice_times("fvm"), util.slice_times("vpm")
    )
    for time in util.plot_times(available):
        plot_frame(float(time), args.format)
    if not len(available):
        raise SystemExit("No coincident cylinder field samples were found.")


if __name__ == "__main__":
    main()
