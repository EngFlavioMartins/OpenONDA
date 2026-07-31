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

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plotutil import CASE_DIR, COLORMAPS, COLORS, hybrid_pvd, run_constants, save
from _reference_util import particles_at, plot_frames, sample_vtu
from _frames_util import CM, D, EPS, U_INF, body_mask, vpm_velocity, vpm_vorticity

FIG = CASE_DIR / "figures"
REF_PVD = CASE_DIR / "referenceFlow" / "solution" / "referenceFlow.pvd"
CMAP_VEL = COLORMAPS["velocity"]
CMAP_ERR = COLORMAPS["error_diverging"]
FIGURE_FORMAT = "png"
FIGURE_DPI = 400
VPM_XMAX = 10.0


def fig_wake_errors(t, ref_vtu, particles, box):
    x_iface = box["xmax"]
    xi = np.linspace(0.0, VPM_XMAX, 320)
    yi = np.linspace(-2.0, 2.0, 160)
    Xi, Yi = np.meshgrid(xi, yi)
    pts = np.column_stack([Xi.ravel(), Yi.ravel(), np.full(Xi.size, EPS)])

    uxv_g = vpm_velocity(particles, pts)[:, 0].reshape(Xi.shape) / U_INF
    wv_g = vpm_vorticity(particles, pts)[:, 2].reshape(Xi.shape) * D / U_INF
    ref = sample_vtu(ref_vtu, pts)
    uxr_g = ref["U"][:, 0].reshape(Xi.shape) / U_INF
    wr_g = ref["vorticity"][:, 2].reshape(Xi.shape) * D / U_INF

    bm = body_mask(Xi, Yi)
    for F in (uxv_g, uxr_g, wv_g, wr_g):
        F[bm] = np.nan

    eu = (uxv_g - uxr_g) * 100
    ew = (wv_g - wr_g) * 100

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
    wvmax = np.nanpercentile(np.abs(wr_g), 99)
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
    plots = []
    for axis, field, minimum, maximum, cmap, title in panels:
        plot = axis.pcolormesh(
            Xi,
            Yi,
            field,
            cmap=cmap,
            vmin=minimum,
            vmax=maximum,
            shading="auto",
        )
        axis.axvline(x_iface, color=COLORS["DarkText"], ls="--", lw=1.0)
        axis.set_title(title)
        axis.set_xlim([0, 5])
        axis.set_ylim([-1.5, 1.5])
        axis.set_aspect("equal")
        plots.append(plot)

    fig.colorbar(plots[0], ax=[ax[0, 0], ax[1, 0]], pad=0.02, aspect=40)
    fig.colorbar(plots[1], ax=[ax[0, 1], ax[1, 1]], pad=0.02, aspect=40)
    fig.colorbar(plots[4], ax=[ax[2, 0]], pad=0.02, aspect=20)
    fig.colorbar(plots[5], ax=[ax[2, 1]], pad=0.02, aspect=20)
    for axis in ax[2, :]:
        axis.set_xlabel(r"$x/D$")
    for axis in ax[:, 0]:
        axis.set_ylabel(r"$y/D$")

    fig.suptitle(f"Wake fields VPM vs reference (z=0, t={t:.2f}s)")

    save(fig, f"wake_errors_t{t:.2f}", FIGURE_FORMAT, FIGURE_DPI)
    plt.close(fig)

    for label, mask in [
        (f"x<{x_iface:g}", Xi < x_iface),
        (f"{x_iface:g}-3", (Xi >= x_iface) & (Xi < 3)),
        ("x>3", Xi >= 3),
    ]:
        print(
            f"  t={t:.1f} {label:>6}: |Δu_x|mean={np.nanmean(np.abs(eu[mask])):.3f}  "
            f"|Δω|mean={np.nanmean(np.abs(ew[mask])):.3f}  "
            f"|Δω|max={np.nanmax(np.abs(ew[mask])):.2f}"
        )


def main() -> None:
    FIG.mkdir(exist_ok=True)
    box = run_constants()["box"]

    frames = [frame for frame in plot_frames(hybrid_pvd(), REF_PVD) if frame[2] is not None]
    if not frames:
        print("  wake_errors: no same-time reference VTUs available")
        return
    for time, _, reference_vtu in frames:
        fig_wake_errors(time, reference_vtu, particles_at(time), box)
        print(f"  wake_errors t={time:.2f} done")


if __name__ == "__main__":
    main()
