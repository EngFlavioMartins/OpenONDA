#!/usr/bin/env python3
"""Wake error-field diagnostic: VPM vs referenceFlow on z=0, for x>0.

3 rows x 2 cols:
  row 0 = VPM solution,  row 1 = reference FVM,  row 2 = error (VPM - ref)
  col 0 = streamwise velocity u_x/Uinf,  col 1 = vorticity omega_z*D/Uinf

A dashed line marks the FVM/VPM coupling interface (the +x box face) so one can
see WHERE the error is born and how it propagates downstream into the free wake.

Mirrors coupled_OFW_VPM/cubeFlow/assets/plot_wake_error_fields.py — same
layout, scaling, titles and colorbars, fed from the native backend's VPM
particle backups and referenceFlow VTU snapshots.
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
from _frames_util import CM, D, EPS, U_INF, body_mask, vpm_velocity, vpm_vorticity

FIG = CASE_DIR / "figures"
REF_PVD = CASE_DIR / "referenceFlow" / "solution" / "referenceFlow.pvd"
CMAP_VEL = COLORMAPS["velocity"]
CMAP_ERR = COLORMAPS["error_diverging"]
VPM_XMAX = 10.0  # VPM domain +x bound (run_setup.py VPM_DOMAIN)


def fig_wake_errors(t, ref_s, particles, box, fmt, dpi):
    x_iface = box["xmax"]  # +x coupling face (interface)

    # Common wake grid: x>0 up to where the VPM exists; y over the clipped span.
    xi = np.linspace(0.0, VPM_XMAX, 320)
    yi = np.linspace(-2.0, 2.0, 160)
    Xi, Yi = np.meshgrid(xi, yi)
    pts = np.column_stack([Xi.ravel(), Yi.ravel(), np.full(Xi.size, EPS)])

    # VPM solution from the particle backup; reference from its VTU snapshot.
    uxv_g = vpm_velocity(particles, pts)[:, 0].reshape(Xi.shape) / U_INF
    wv_g = vpm_vorticity(particles, pts)[:, 2].reshape(Xi.shape) * D / U_INF
    ref = sample_vtu(ref_s[1], pts)
    uxr_g = ref["U"][:, 0].reshape(Xi.shape) / U_INF
    wr_g = ref["vorticity"][:, 2].reshape(Xi.shape) * D / U_INF

    # Mask out cube body interior + thin band so they're white and excluded.
    bm = body_mask(Xi, Yi)
    for F in (uxv_g, uxr_g, wv_g, wr_g):
        F[bm] = np.nan

    eu = (uxv_g - uxr_g) * 100  # signed velocity error
    ew = (wv_g - wr_g) * 100  # signed vorticity error

    # Create the figure:
    fig, ax = plt.subplots(
        3,
        2,
        figsize=(12.5 * CM, 11.0 * CM),
        dpi=400,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    uvmax = np.nanpercentile(np.abs(uxr_g), 99) or 1.0
    wvmax = np.nanpercentile(np.abs(wr_g), 99) if np.isfinite(wr_g).any() else np.nan

    if not np.isfinite(wvmax) or wvmax <= 0:
        wvmax = np.nanpercentile(np.abs(wv_g), 95)
    eumax = np.nanpercentile(np.abs(eu), 99)
    ewmax = np.nanpercentile(np.abs(ew), 99)

    panels = [
        (ax[0, 0], uxv_g, -uvmax, uvmax, CMAP_VEL, r"VPM  $u_x/U_\infty$"),
        (ax[0, 1], wv_g, -wvmax, wvmax, COLORMAPS["vorticity"], r"VPM  $\omega_z D/U_\infty$"),
        (ax[1, 0], uxr_g, -uvmax, uvmax, CMAP_VEL, r"Reference  $u_x/U_\infty$"),
        (
            ax[1, 1],
            wr_g,
            -wvmax,
            wvmax,
            COLORMAPS["vorticity"],
            r"Reference  $\omega_z D/U_\infty$",
        ),
        (
            ax[2, 0],
            eu,
            -eumax,
            eumax,
            CMAP_ERR,
            r"Error  $\Delta u_x/U_\infty$",
        ),
        (
            ax[2, 1],
            ew,
            -ewmax,
            ewmax,
            CMAP_ERR,
            r"Error  $\Delta\omega_z D/U_\infty$",
        ),
    ]
    pcs = {}
    for idx, (axis, F, vmn, vmx, cmap, title) in enumerate(panels):
        pc = axis.pcolormesh(Xi, Yi, F, cmap=cmap, vmin=vmn, vmax=vmx, shading="auto")
        axis.axvline(x_iface, color=COLORS["DarkText"], ls="--", lw=1.0)
        axis.set_title(title)
        axis.set_xlim([0, 5])
        axis.set_ylim([-1.5, 1.5])
        axis.set_aspect("equal")
        pcs[idx] = pc

    fig.colorbar(pcs[0], ax=[ax[0, 0], ax[1, 0]], pad=0.02, aspect=40)
    fig.colorbar(pcs[1], ax=[ax[0, 1], ax[1, 1]], pad=0.02, aspect=40)
    fig.colorbar(pcs[4], ax=[ax[2, 0]], pad=0.02, aspect=20)
    fig.colorbar(pcs[5], ax=[ax[2, 1]], pad=0.02, aspect=20)
    for axis in ax[2, :]:
        axis.set_xlabel(r"$x/D$")
    for axis in ax[:, 0]:
        axis.set_ylabel(r"$y/D$")

    fig.suptitle(f"Wake fields VPM vs reference (z=0, t={t:.2f}s)")

    save(fig, f"wake_errors_t{t:.2f}", fmt, dpi or 400)
    plt.close(fig)

    # error onset diagnostics per x-band
    for lab, m in [
        ("x<1.5", Xi < x_iface),
        ("1.5-3", (Xi >= x_iface) & (Xi < 3)),
        ("x>3", Xi >= 3),
    ]:
        print(
            f"  t={t:.1f} {lab:>6}: |Δu_x|mean={np.nanmean(np.abs(eu[m])):.3f}  "
            f"|Δω|mean={np.nanmean(np.abs(ew[m])):.3f}  "
            f"|Δω|max={np.nanmax(np.abs(ew[m])):.2f}"
        )


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
        if ref_s is None or abs(ref_s[0] - t) > MATCH_TOL:
            print(f"  wake_errors t={t:.2f}: no coincident reference snapshot — skip")
            continue
        assert_same_time(t, ref_s[0])
        fig_wake_errors(t, ref_s, particles, box, args.format, args.dpi)
        print(f"  wake_errors t={t:.2f} (ref t={ref_s[0]:.2f}) done")


if __name__ == "__main__":
    main()
