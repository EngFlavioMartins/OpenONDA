#!/usr/bin/env python3
"""Velocity field plots: Hybrid FVM vs VPM, and Reference vs stitched FVM+VPM.

Mirrors coupled_OFW_VPM/cubeFlow/assets/plot_velocity_fields.py — same
layout, titles, scaling and colorbars, fed from the native backend's VTU
snapshots and VPM particle backups.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from _plotutil import CASE_DIR, COLORMAPS, COLORS, run_constants, save
from _reference_util import (
    MATCH_TOL,
    _pvd_times,
    assert_same_time,
    load_vpm_particles,
    nearest_vpm_h5,
    nearest_vtu,
    sample_vtu,
)
from _frames_util import CM, EPS, U_INF, body_mask, hybrid_velocity, vpm_velocity, _add_body

FIG = CASE_DIR / "figures"
REF_PVD = CASE_DIR / "referenceFlow" / "solution" / "referenceFlow.pvd"
CMAP_VEL = COLORMAPS["velocity"]
CMAP_ERR = COLORMAPS["error"]
H_MESH = 0.05  # VPM/FVM spacing (cubeFlow_setup.py H_CORE)


def _add_body_patch(ax):
    """Draw the body cross-section (same patch as the OFW figures)."""
    _add_body(ax, COLORS)


def _style_axes(fig, axes, box, cf_vel, cf_err, vmax, p95):
    """Shared axis/colorbar styling for the 1x3 field figures (OFW-identical)."""
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(box["xmin"], box["xmax"])
        ax.set_ylim(box["ymin"], box["ymax"])
        _add_body_patch(ax)
        ax.set_xlabel(r"$x/D$")
    axes[0].set_ylabel(r"$y/D$")

    fig.colorbar(
        cf_vel,
        ax=axes[:2].tolist(),
        orientation="horizontal",
        shrink=0.8,
        pad=0.15,
        aspect=40,
        format="%.1f",
        label=r"$u_x / U_\infty$",
    ).set_ticks(np.linspace(-vmax, vmax, 3))
    fig.colorbar(
        cf_err,
        ax=axes[2],
        orientation="horizontal",
        shrink=0.8,
        pad=0.15,
        aspect=20,
        format="%.1f",
        label=r"$\varepsilon$ [\%]",
    ).set_ticks(np.linspace(0, p95, 3))


def fig_velocity_fields(t, hyb_vtu, particles, box, fmt, dpi):
    """Hybrid FVM vs VPM velocity field + error (OFW figure 1)."""
    n = 61
    xs = np.linspace(box["xmin"] + 0.01, box["xmax"] - 0.01, n)
    ys = np.linspace(box["ymin"] + 0.01, box["ymax"] - 0.01, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, EPS)])

    ux_h = sample_vtu(hyb_vtu, pts)["U"][:, 0].reshape(X.shape)
    ux_v = vpm_velocity(particles, pts)[:, 0].reshape(X.shape)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.5 * CM, 7.0 * CM),
        dpi=300,
        constrained_layout=True,
        sharey=True,
        sharex=True,
    )
    vmax = max(np.nanmax(np.abs(ux_h)), np.nanmax(np.abs(ux_v)))
    levels = np.linspace(-vmax, vmax, 41)

    axes[0].contourf(X, Y, ux_h, levels=levels, cmap=CMAP_VEL, extend="both")
    axes[0].set_title(r"Hybrid FVM, $u_x^\text{FVM}$")
    cf1 = axes[1].contourf(X, Y, ux_v, levels=levels, cmap=CMAP_VEL, extend="both")
    axes[1].set_title(r"VPM, $u_x^\text{VPM}$")

    # |FVM - VPM| error; masked inside the body (+margin) and where |u| ~ 0.
    error = np.abs(ux_h - ux_v) / U_INF * 100
    error[np.abs(ux_h) < U_INF / 100] = np.nan
    error[body_mask(X, Y)] = np.nan

    p95 = np.nanpercentile(error, 95)
    cf3 = axes[2].pcolormesh(X, Y, error, cmap=CMAP_ERR, vmin=0, vmax=max(p95, 1e-3))
    axes[2].set_title(r"$\varepsilon$ [\%]")

    _style_axes(fig, axes, box, cf1, cf3, vmax, max(p95, 1e-3))

    save(fig, f"velocity_fields_t{t:.2f}", fmt, dpi or 300)
    plt.close(fig)

    mean_err = np.nanmean(error)
    outlet_msg = ""
    valid = np.isfinite(error)
    outlet_band = 3.0 * H_MESH
    outlet_mask = valid & (X >= (box["xmax"] - outlet_band))
    if np.any(outlet_mask):
        outlet_err = error[outlet_mask]
        outlet_msg = (
            f", OutletMean={np.nanmean(outlet_err):.1f}%, "
            f"OutletP95={np.nanpercentile(outlet_err, 95):.1f}% "
            f"(x>={box['xmax'] - outlet_band:.2f})"
        )
    print(
        f"  Velocity field plot: velocity_fields_t{t:.2f} "
        f"(Mean={mean_err:.1f}%, P95={p95:.1f}%{outlet_msg})"
    )
    return mean_err


def fig_stitched_vs_reference(t, hyb_vtu, ref_s, particles, box, fmt, dpi):
    """Reference FVM vs stitched FVM+VPM velocity + error (OFW figure 2)."""
    n = 61
    xs = np.linspace(box["xmin"] + 0.01, box["xmax"] - 0.01, n)
    ys = np.linspace(box["ymin"] + 0.01, box["ymax"] - 0.01, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, EPS)])

    ux_ref = sample_vtu(ref_s[1], pts)["U"][:, 0].reshape(X.shape)
    ux_st = hybrid_velocity(hyb_vtu, particles, pts, box, sample_vtu)[0][:, 0].reshape(X.shape)

    err = np.abs(ux_ref - ux_st) / U_INF * 100.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.5 * CM, 7.0 * CM),
        dpi=300,
        constrained_layout=True,
        sharey=True,
        sharex=True,
    )

    vmax = max(float(np.nanmax(np.abs(ux_ref))), float(np.nanmax(np.abs(ux_st))))
    levels = np.linspace(-vmax, vmax, 41)

    axes[0].contourf(X, Y, ux_ref, levels=levels, cmap=CMAP_VEL, extend="both")
    axes[0].set_title(r"Reference FVM, $u_x^{\mathrm{ref}}$")
    cf2 = axes[1].contourf(X, Y, ux_st, levels=levels, cmap=CMAP_VEL, extend="both")
    axes[1].set_title(r"Stitched FVM+VPM, $u_x^{\mathrm{stitched}}$")

    valid = np.isfinite(err)
    p95 = float(np.nanpercentile(err[valid], 95)) if np.any(valid) else 5.0
    cf3 = axes[2].pcolormesh(X, Y, err, cmap=CMAP_ERR, vmin=0, vmax=max(p95, 1e-3))
    axes[2].set_title(r"$\varepsilon$ [\%]")

    _style_axes(fig, axes, box, cf2, cf3, vmax, max(p95, 1e-3))

    mean_err = float(np.nanmean(err[valid])) if np.any(valid) else np.nan
    save(fig, f"Stitched_RefVsHybrid_t{t:.2f}", fmt, dpi or 300)
    plt.close(fig)
    print(
        f"  Stitched vs Reference: Stitched_RefVsHybrid_t{t:.2f} "
        f"(Mean={mean_err:.1f}%, P95={p95:.1f}%)"
    )
    return mean_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", default="all", help="all | latest | comma list")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--format", default="png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    FIG.mkdir(exist_ok=True)
    C = run_constants()
    box = C["box"]

    hyb_pvd = next(iter((CASE_DIR / "solution").glob("coupled_*.pvd")), None)
    if hyb_pvd is None:
        sys.exit("no hybrid .pvd in solution/")
    entries = [(tt, f) for tt, f in _pvd_times(hyb_pvd) if tt > 1e-9]
    if args.times == "latest":
        entries = entries[-1:]
    elif args.times != "all":
        wanted = [float(v) for v in args.times.split(",")]
        entries = [min(entries, key=lambda e: abs(e[0] - w)) for w in wanted]

    for t, hyb_vtu in entries:
        out1 = FIG / f"velocity_fields_t{t:.2f}.{args.format}"
        out2 = FIG / f"Stitched_RefVsHybrid_t{t:.2f}.{args.format}"
        if not args.force and out1.exists() and out2.exists():
            continue
        h5 = nearest_vpm_h5(t)
        particles = (
            load_vpm_particles(h5)
            if h5
            else {
                "position": np.zeros((0, 3)),
                "circulation": np.zeros((0, 3)),
                "radius": np.zeros(0),
            }
        )
        ref_s = nearest_vtu(REF_PVD, t)
        if ref_s is None or abs(ref_s[0] - t) > MATCH_TOL:
            print(f"  velocity_fields t={t:.2f}: no coincident reference snapshot — skip")
            continue
        assert_same_time(t, ref_s[0])  # guard: identical physical time only
        if args.force or not out1.exists():
            fig_velocity_fields(t, hyb_vtu, particles, box, args.format, args.dpi)
        if args.force or not out2.exists():
            fig_stitched_vs_reference(t, hyb_vtu, ref_s, particles, box, args.format, args.dpi)
        print(f"  velocity_fields t={t:.2f} (ref t={ref_s[0]:.2f}) done")


if __name__ == "__main__":
    main()
