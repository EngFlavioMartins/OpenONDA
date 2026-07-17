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
from _reference_util import _pvd_times, load_vpm_particles, nearest_vpm_h5, nearest_vtu, sample_vtu
from _frames_util import CM, D, EPS, U_INF, body_mask, hybrid_velocity, hybrid_vorticity

FIG = CASE_DIR / "figures"
REF_PVD = CASE_DIR / "referenceFlow" / "solution" / "referenceFlow.pvd"
CMAP_VEL = COLORMAPS["velocity"]
CMAP_ERR = COLORMAPS["error_diverging"]


def fig_wake_errors(t, hyb_vtu, ref_s, particles, box, fmt, dpi):
    xi = np.linspace(0.0, 9.0, 240)
    yi = np.linspace(-2.0, 2.0, 120)
    X, Y = np.meshgrid(xi, yi)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, EPS)])

    uv = hybrid_velocity(hyb_vtu, particles, pts, box, sample_vtu)[0][:, 0].reshape(X.shape) / U_INF
    wv = (
        hybrid_vorticity(hyb_vtu, particles, pts, box, sample_vtu)[0][:, 2].reshape(X.shape)
        * D
        / U_INF
    )
    ref = sample_vtu(ref_s[1], pts)
    ur = ref["U"][:, 0].reshape(X.shape) / U_INF
    wr = ref["vorticity"][:, 2].reshape(X.shape) * D / U_INF

    bm = body_mask(X, Y)
    for F in (uv, ur, wv, wr):
        F[bm] = np.nan

    fig, ax = plt.subplots(
        3,
        2,
        figsize=(12.5 * CM, 11.0 * CM),
        dpi=400,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    lev_u = np.linspace(-0.4, 1.4, 37)
    wmax = np.nanpercentile(np.abs(wr), 99) or 1.0
    lev_w = np.linspace(-wmax, wmax, 37)

    cu = ax[0, 0].contourf(X, Y, uv, levels=lev_u, cmap=CMAP_VEL, extend="both")
    ax[0, 0].set_title(r"Hybrid $u_x/U_\infty$")
    cw = ax[0, 1].contourf(X, Y, wv, levels=lev_w, cmap=COLORMAPS["vorticity"], extend="both")
    ax[0, 1].set_title(r"Hybrid $\omega_z D/U_\infty$")
    ax[1, 0].contourf(X, Y, ur, levels=lev_u, cmap=CMAP_VEL, extend="both")
    ax[1, 0].set_title(rf"Reference $u_x/U_\infty$ (t={ref_s[0]:.2f})")
    ax[1, 1].contourf(X, Y, wr, levels=lev_w, cmap=COLORMAPS["vorticity"], extend="both")
    ax[1, 1].set_title(rf"Reference $\omega_z D/U_\infty$ (t={ref_s[0]:.2f})")

    eu = (uv - ur) * 100
    ew = (wv - wr) * 100
    pu = max(np.nanpercentile(np.abs(eu), 98), 1.0)
    pw = max(np.nanpercentile(np.abs(ew), 98), 1.0)
    ceu = ax[2, 0].pcolormesh(X, Y, eu, cmap=CMAP_ERR, vmin=-pu, vmax=pu)
    ax[2, 0].set_title(r"Error $\Delta u_x/U_\infty$ [\%]")
    cew = ax[2, 1].pcolormesh(X, Y, ew, cmap=CMAP_ERR, vmin=-pw, vmax=pw)
    ax[2, 1].set_title(r"Error $\Delta\omega_z D/U_\infty$ [\%]")

    for a in ax.ravel():
        a.axvline(box["xmax"], color="k", ls="--", lw=0.6)
        a.set_xlim([0, 5])
        a.set_ylim([-1.5, 1.5])
        a.set_aspect("equal")
    for a in ax[:, 0]:
        a.set_ylabel(r"$y/D$", fontsize=7)
    for a in ax[2, :]:
        a.set_xlabel(r"$x/D$", fontsize=7)
    fig.suptitle(f"Wake fields hybrid vs reference (z=0, t={t:.2f}s)")
    fig.colorbar(cu, ax=ax[:2, 0].tolist(), shrink=0.7, label=r"$u_x/U_\infty$")
    fig.colorbar(cw, ax=ax[:2, 1].tolist(), shrink=0.7, label=r"$\omega_z D/U_\infty$")
    fig.colorbar(ceu, ax=ax[2, 0], shrink=0.9)
    fig.colorbar(cew, ax=ax[2, 1], shrink=0.9)

    save(fig, f"wake_errors_t{t:.2f}", fmt, dpi or 400)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", default="all", help="all | latest | comma list")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--format", default="png")
    ap.add_argument("--dpi", type=int, default=400)
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
        out = FIG / f"wake_errors_t{t:.2f}.{args.format}"
        if not args.force and out.exists():
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
        fig_wake_errors(t, hyb_vtu, ref_s, particles, box, args.format, args.dpi)
        print(f"  wake_errors t={t:.2f} done")


if __name__ == "__main__":
    main()
