#!/usr/bin/env python3
"""Compare Coupled FVM and VPM velocity fields on the saved z=0 slice."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from . import postprocess as util


COMPARISONS = {
    "coupled_fvm_vpm_fields": ("fvm", "vpm", "(a) Coupled FVM", "(b) VPM"),
    "reference_fvm_vpm_fields": ("reference", "vpm", "(a) Reference FVM", "(b) VPM"),
    "reference_fvm_coupled_fvm_fields": (
        "reference",
        "fvm",
        "(a) Reference FVM",
        "(b) Coupled FVM",
    ),
}


def _on_grid(source: dict, target: dict) -> np.ndarray:
    """Match equivalent saved coordinates; never fill gaps or extrapolate."""
    for key in ("x", "y"):
        a, b = source[key], target[key]
        tolerance = 8 * np.finfo(np.float32).eps * max(1, float(np.max(np.abs(b))))
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=tolerance):
            raise ValueError(
                "Slice coordinates differ: resample the saved fields on one common grid"
            )
    result = np.array(source["velocity"], dtype=float, copy=True)
    result[~source["valid"]] = np.nan
    return result


def area_weights(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Vertex quadrature over rectangles whose four corners are valid.

    This matches the support used by the contour display. Report the covered
    area, because an unsampled wall strip must not count as zero error.
    """
    if not np.allclose(x, x[:1, :]) or not np.allclose(y, y[:, :1]):
        raise ValueError("Expected a rectilinear slice")
    dx, dy = np.diff(x[0]), np.diff(y[:, 0])
    if np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError("Slice axes must increase")
    cells = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    area = dy[:, None] * dx[None, :] * cells
    weights = np.zeros_like(x)
    weights[:-1, :-1] += area / 4
    weights[1:, :-1] += area / 4
    weights[:-1, 1:] += area / 4
    weights[1:, 1:] += area / 4
    return weights


def differences(x, y, left, right):
    if left.shape != (*x.shape, 3) or right.shape != left.shape:
        raise ValueError("A velocity comparison requires all three components")
    valid = np.all(np.isfinite(left), axis=-1) & np.all(np.isfinite(right), axis=-1)
    valid &= ~((np.abs(x) <= 0.5 + 1e-12) & (np.abs(y) <= 0.5 + 1e-12))
    weights = area_weights(x, y, valid)
    if not np.any(weights > 0):
        raise ValueError("No common sampled fluid area")
    delta = np.where(valid[..., None], right - left, np.nan)
    error = 100 * np.linalg.norm(delta, axis=-1)
    # Discrete area-weighted RMS; maximum is over valid sample nodes, not pixels.
    mean_square = np.sum(weights * np.nan_to_num(error) ** 2) / weights.sum()
    stats = {
        "rms_percent": float(np.sqrt(mean_square)),
        "sampled_max_percent": float(np.nanmax(error)),
        "covered_area_D2": float(weights.sum()),
        "valid_nodes": int(valid.sum()),
    }
    return error, valid, stats


def display_grid(x, y, vectors, valid, subdivisions=4):
    """Bilinearly interpolate vectors for contours, then let callers take norms.

    No smoothing or statistical calculation is performed on display pixels.
    Quads with missing corners remain masked, including the wall-adjacent strip.
    """
    nx, ny = x.shape[1], x.shape[0]
    tx = np.linspace(0, nx - 1, (nx - 1) * subdivisions + 1)
    ty = np.linspace(0, ny - 1, (ny - 1) * subdivisions + 1)
    ix = np.minimum(tx.astype(int), nx - 2)
    iy = np.minimum(ty.astype(int), ny - 2)
    wx, wy = (tx - ix)[None, :, None], (ty - iy)[:, None, None]
    a = vectors[iy[:, None], ix[None, :]]
    b = vectors[iy[:, None], ix[None, :] + 1]
    c = vectors[iy[:, None] + 1, ix[None, :]]
    d = vectors[iy[:, None] + 1, ix[None, :] + 1]
    values = (1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * c + wx * d)
    good = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    values[~good[iy[:, None], ix[None, :]]] = np.nan
    xx, yy = np.meshgrid(np.interp(tx, np.arange(nx), x[0]), np.interp(ty, np.arange(ny), y[:, 0]))
    return xx, yy, values


def _field_figure(
    time, x, y, left, right, left_title, right_title, name, fmt, dpi, *, time_scale=1
):
    util._THEME.set_thesis_style()
    error, valid, stats = differences(x, y, left, right)
    xx, yy, dense_left = display_grid(x, y, left, valid)
    _, _, dense_right = display_grid(x, y, right, valid)
    dense_error = 100 * np.linalg.norm(dense_right - dense_left, axis=-1)
    combined = np.concatenate((left[..., 0][valid], right[..., 0][valid]))
    low, high = float(combined.min()), float(combined.max())
    if high - low < 1e-8:
        low, high = low - 0.01, high + 0.01
    maximum = max(stats["sampled_max_percent"], 1e-6)

    height_cm = 10.5
    fig = plt.figure(figsize=util.figure_size(height_cm), dpi=dpi)
    axes = [
        fig.add_axes((0.14, 5.65 / height_cm, 0.30, 3.75 / height_cm)),
        fig.add_axes((0.56, 5.65 / height_cm, 0.30, 3.75 / height_cm)),
        fig.add_axes((0.14, 0.9 / height_cm, 0.30, 3.75 / height_cm)),
    ]
    velocity_bar = fig.add_axes((0.56, 3.65 / height_cm, 0.30, 0.14 / height_cm))
    error_bar = fig.add_axes((0.56, 2.25 / height_cm, 0.30, 0.14 / height_cm))
    fig.text(
        0.5,
        0.975,
        rf"$z/D=0,\quad tU_\infty/D={time * time_scale:.2f}$",
        ha="center",
        va="top",
    )
    levels = np.linspace(low, high, 41)
    for ax, values, title in zip(axes[:2], (dense_left, dense_right), (left_title, right_title)):
        velocity_plot = ax.contourf(
            xx,
            yy,
            values[..., 0],
            levels=levels,
            cmap=util.COLORMAPS["velocity"],
            corner_mask=False,
        )
        ax.set_title(title, pad=2.0)
    error_plot = axes[2].contourf(
        xx,
        yy,
        dense_error,
        levels=np.linspace(0, maximum, 41),
        cmap=util.COLORMAPS["error"],
        corner_mask=False,
    )
    axes[2].set_title("(c) Difference", pad=2.0)
    for ax in axes:
        ax.set(
            xlabel=r"$x/D$",
            xlim=(x.min(), x.max()),
            ylim=(y.min(), y.max()),
            xticks=[-1, 0, 1],
            yticks=[-1, 0, 1],
            aspect="equal",
        )
        ax.set_facecolor(util.COLORS["background_light"])
        ax.xaxis.labelpad = 1.0
        ax.add_patch(
            plt.Rectangle((-0.5, -0.5), 1, 1, facecolor="white", edgecolor="black", lw=0.5)
        )
    axes[0].set_ylabel(r"$y/D$")
    axes[2].set_ylabel(r"$y/D$")
    axes[1].tick_params(labelleft=False)
    velocity_colorbar = fig.colorbar(
        velocity_plot,
        cax=velocity_bar,
        orientation="horizontal",
        format="%.2g",
        ticks=[low, (low + high) / 2, high],
        label=r"$u_x/U_\infty$",
    )
    error_colorbar = fig.colorbar(
        error_plot,
        cax=error_bar,
        orientation="horizontal",
        format="%.2g",
        ticks=[0, maximum / 2, maximum],
        label=r"$\|\Delta\mathbf{u}\|/U_\infty$ [\%]",
    )
    velocity_colorbar.ax.xaxis.labelpad = 1.0
    error_colorbar.ax.xaxis.labelpad = 1.0
    fig.text(
        0.71,
        0.9 / height_cm,
        rf"RMS: {stats['rms_percent']:.2g}\%"
        "\n"
        rf"Max: {stats['sampled_max_percent']:.2g}\%",
        ha="center",
        va="center",
    )
    util.save(fig, f"{name}_t{time:.2f}", fmt, dpi)
    plt.close(fig)
    return {"figure": name, "time": time, **stats}


def plot_frame(
    time,
    consts,
    comparison="coupled_fvm_vpm_fields",
    figure_format="png",
    dpi=util.FIGURE_DPI,
):
    fvm, vpm, reference = (util.load_slice(source, time) for source in ("fvm", "vpm", "reference"))
    if any(item is None for item in (fvm, vpm, reference)):
        raise ValueError(f"No exactly coincident fields at t={time:g}")
    if consts["reference_length"] != 1:
        raise ValueError("Cube geometry uses D=1")
    speed = consts["freestream_speed"]
    x, y = fvm["x"], fvm["y"]
    fv = _on_grid(fvm, fvm) / speed
    vp = _on_grid(vpm, fvm) / speed
    rf = _on_grid(reference, fvm) / speed
    # Every comparison uses the same support, so its RMS is comparable with
    # the other pairings even where the fitted FVM boundary lacks a sample.
    shared_valid = (
        np.all(np.isfinite(fv), axis=-1)
        & np.all(np.isfinite(vp), axis=-1)
        & np.all(np.isfinite(rf), axis=-1)
    )
    for velocity in (fv, vp, rf):
        velocity[~shared_valid] = np.nan
    fields = {"fvm": fv, "vpm": vp, "reference": rf}
    left_source, right_source, left_title, right_title = COMPARISONS[comparison]
    return _field_figure(
        time,
        x,
        y,
        fields[left_source],
        fields[right_source],
        left_title,
        right_title,
        comparison,
        figure_format,
        dpi,
        time_scale=speed / consts["reference_length"],
    )


def main(comparison="coupled_fvm_vpm_fields"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=util.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=util.FIGURE_DPI)
    args = parser.parse_args()
    util.validate_plot_inputs()
    times = util.common_times(*(util.slice_times(s) for s in ("fvm", "vpm", "reference")))
    if not len(times):
        raise SystemExit("No coincident field samples")
    rows = []
    consts = util.run_constants()
    for time in times:
        rows.append(plot_frame(float(time), consts, comparison, args.format, args.dpi))
    util.remove_obsolete_frames(comparison, times, args.format)
    util.AUXILIARY.mkdir(parents=True, exist_ok=True)
    with (util.AUXILIARY / f"{comparison}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
