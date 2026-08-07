#!/usr/bin/env python3
"""Wake error-field diagnostic: coupled VPM vs referenceFlow, z=0, x>0.

Built entirely from sampler output under ``samples/`` - the coupled run's
``vpm_wake_slice_z0`` and referenceFlow's ``wake_slice_z0`` - so plotting needs
no particle backups and no raw VTU access.

3 rows x 2 cols:
  row 0 = VPM solution,  row 1 = reference FVM,  row 2 = error (VPM - reference)
  col 0 = streamwise velocity u_x/Uinf,  col 1 = vorticity omega_z*D/Uinf

A dashed line marks the FVM/VPM coupling interface (the +x box face) so one can
see WHERE the error is born and how it propagates downstream into the free wake.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

NAME = "wake_slice_z0"
CMAP_VEL = util.COLORMAPS["velocity"]
CMAP_VORT = util.COLORMAPS["vorticity"]
CMAP_ERR = util.COLORMAPS["error_diverging"]
FIGURE_FORMAT = "png"
FIGURE_DPI = 400
BODY_HALF = 0.5


def _body_mask(x, y, margin=0.05):
    return (np.abs(x) <= BODY_HALF + margin) & (np.abs(y) <= BODY_HALF + margin)


def _to_grid(source_slice, target):
    """Interpolate a slice onto the target slice's grid."""
    points = np.column_stack([source_slice["x"].ravel(), source_slice["y"].ravel()])
    target_points = (target["x"], target["y"])
    out = {}
    for key in ("Ux", "omega_z"):
        values = source_slice[key]
        out[key] = (
            griddata(points, values.ravel(), target_points, method="linear")
            if values is not None
            else None
        )
    return out


def fig_wake_errors(time, vpm, reference, U_inf, D, x_iface):
    x, y = vpm["x"] / D, vpm["y"] / D
    ref = _to_grid(reference, vpm)
    bm = _body_mask(x, y)

    def scaled(values, scale):
        out = values / scale
        out[bm] = np.nan
        return out

    uxv_g = scaled(vpm["Ux"], U_inf)
    wv_g = scaled(vpm["omega_z"], U_inf / D)
    uxr_g = scaled(ref["Ux"], U_inf)
    wr_g = scaled(ref["omega_z"], U_inf / D)

    eu = (uxv_g - uxr_g) * 100
    ew = (wv_g - wr_g) * 100

    fig, ax = plt.subplots(
        3,
        2,
        figsize=(12.5, 11.0),
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
        (ax[0, 1], wv_g, -wvmax, wvmax, CMAP_VORT, r"VPM  $\omega_z D/U_\infty$"),
        (ax[1, 0], uxr_g, -uvmax, uvmax, CMAP_VEL, r"Reference  $u_x/U_\infty$"),
        (ax[1, 1], wr_g, -wvmax, wvmax, CMAP_VORT, r"Reference  $\omega_z D/U_\infty$"),
        (ax[2, 0], eu, -eumax, eumax, CMAP_ERR, r"Error  $\Delta u_x/U_\infty$"),
        (ax[2, 1], ew, -ewmax, ewmax, CMAP_ERR, r"Error  $\Delta\omega_z D/U_\infty$"),
    ]
    plots = []
    for axis, field, minimum, maximum, cmap, title in panels:
        plot = axis.pcolormesh(
            x, y, field, cmap=cmap, vmin=minimum, vmax=maximum, shading="auto"
        )
        axis.axvline(x_iface, color=util.COLORS["DarkText"], ls="--", lw=1.0)
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

    fig.suptitle(f"Wake fields VPM vs reference (z=0, t={time:.2f}s)")

    util.save(fig, f"wake_errors_t{time:.2f}", FIGURE_FORMAT, FIGURE_DPI)
    plt.close(fig)

    for label, mask in [
        (f"x<{x_iface:g}", x < x_iface),
        (f"{x_iface:g}-3", (x >= x_iface) & (x < 3)),
        ("x>3", x >= 3),
    ]:
        print(
            f"  t={time:.1f} {label:>6}: |Delta u_x|mean={np.nanmean(np.abs(eu[mask])):.3f}  "
            f"|Delta w|mean={np.nanmean(np.abs(ew[mask])):.3f}  "
            f"|Delta w|max={np.nanmax(np.abs(ew[mask])):.2f}"
        )


def main() -> None:
    consts = util.run_constants()
    U_inf, D, x_iface = consts["U_inf"], consts["D"], consts["box"]["xmax"]

    times = util.comparison_times()
    if times.size == 0:
        raise SystemExit("No sampled wake data found; run the case or the resampler first.")

    plotted = 0
    for time in times:
        vpm = util.load_slice("vpm", float(time), name=NAME)
        reference = util.load_slice("reference", float(time), name=NAME)
        if vpm is None or reference is None:
            continue
        fig_wake_errors(float(time), vpm, reference, U_inf, D, x_iface)
        print(f"  wake_errors t={time:.2f} done")
        plotted += 1

    if plotted == 0:
        print("  wake_errors: no matching vpm_wake_slice_z0 / wake_slice_z0 frames available")


if __name__ == "__main__":
    main()
