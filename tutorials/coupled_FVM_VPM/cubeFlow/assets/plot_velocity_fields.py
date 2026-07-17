#!/usr/bin/env python3
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
from _reference_util import _pvd_times, nearest_vtu, sample_vtu
from _frames_util import CM, EPS, body_mask, _add_body

FIG = CASE_DIR / "figures"
REF_PVD = CASE_DIR / "referenceFlow" / "solution" / "referenceFlow.pvd"
U_INF = 1.0
CMAP_VEL = COLORMAPS["velocity"]
CMAP_ERR = COLORMAPS["error"]


def fig_velocity_fields(t, hyb_vtu, ref_s, box, fmt, dpi):
    n = 61
    xs = np.linspace(box["xmin"] + 0.01, box["xmax"] - 0.01, n)
    ys = np.linspace(box["ymin"] + 0.01, box["ymax"] - 0.01, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, EPS)])

    ux_h = sample_vtu(hyb_vtu, pts)["U"][:, 0].reshape(X.shape)
    ux_ref = sample_vtu(ref_s[1], pts)["U"][:, 0].reshape(X.shape)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.5 * CM, 7.0 * CM),
        dpi=300,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    vmax = max(np.nanmax(np.abs(ux_h)), np.nanmax(np.abs(ux_ref)), 1.0)
    levels = np.linspace(-vmax, vmax, 41)
    axes[0].contourf(X, Y, ux_h, levels=levels, cmap=CMAP_VEL, extend="both")
    axes[0].set_title(r"Hybrid FVM, $u_x^{\mathrm{hybrid}}$")
    cf1 = axes[1].contourf(X, Y, ux_ref, levels=levels, cmap=CMAP_VEL, extend="both")
    axes[1].set_title(r"Reference FVM, $u_x^{\mathrm{ref}}$")

    err = np.abs(ux_h - ux_ref) / U_INF * 100
    err[body_mask(X, Y)] = np.nan
    p95 = np.nanpercentile(err, 95)
    cf3 = axes[2].pcolormesh(X, Y, err, cmap=CMAP_ERR, vmin=0, vmax=max(p95, 1e-3))
    axes[2].set_title(r"$\varepsilon$ [\%]")

    for ax in axes:
        ax.set_aspect("equal")
        _add_body(ax, COLORS)
        ax.set_xlabel(r"$x/D$")
    axes[0].set_ylabel(r"$y/D$")
    velocity_bar = fig.colorbar(
        cf1,
        ax=axes[:2].tolist(),
        orientation="horizontal",
        shrink=0.8,
        pad=0.12,
        aspect=40,
        format="%.1f",
        label=r"$u_x/U_\infty$",
    )
    velocity_bar.set_ticks(np.linspace(-vmax, vmax, 3))
    error_bar = fig.colorbar(
        cf3,
        ax=axes[2],
        orientation="horizontal",
        shrink=0.8,
        pad=0.12,
        aspect=20,
        format="%.1f",
        label=r"$\varepsilon$ [\%]",
    )
    error_bar.set_ticks(np.linspace(0, max(p95, 1e-3), 3))
    save(fig, f"velocity_fields_t{t:.2f}", fmt, dpi or 300)
    plt.close(fig)


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
        out = FIG / f"velocity_fields_t{t:.2f}.{args.format}"
        if not args.force and out.exists():
            continue
        ref_s = nearest_vtu(REF_PVD, t)
        fig_velocity_fields(t, hyb_vtu, ref_s, box, args.format, args.dpi)
        print(f"  velocity_fields t={t:.2f} done")


if __name__ == "__main__":
    main()
