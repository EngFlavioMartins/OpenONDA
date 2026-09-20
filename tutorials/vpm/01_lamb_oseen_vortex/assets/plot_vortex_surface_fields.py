#!/usr/bin/env python3
"""Cross-sectional field comparison for the single Lamb-Oseen vortex.

Deterministic schemes sample z=L/4. RWM uses the ensemble-mean
column projection, so the titles do not assign it a sampling plane.

Each of the four viscous schemes (GBD, CS, RWM, DVH) contributes **one
quadrant** of the plane.  The quadrants tile into a seamless image of
the full field, making it immediately clear which scheme over- or
under-diffuses relative to the others.

Quadrant layout
---------------
  x <= 0, y >= 0  |  x >= 0, y >= 0
  ----------------------------------
        GBD       |      CS
  ----------------------------------
        RWM      |      DVH
  ----------------------------------
  x <= 0, y <= 0  |  x >= 0, y <= 0

A single, shared colour bar per panel (velocity magnitude and z-vorticity)
enables direct quantitative comparison across schemes.

Saves: figures/vortex_surface_fields.png
"""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

if __package__:
    from .postprocess import (
        SCHEMES,
        build_arg_parser,
        figure_size,
        load_theme,
        resolve_runtime_physics,
        save_fig,
        surface_plot_tiles,
        validate_thesis_figure,
    )
else:
    from .postprocess import (
        SCHEMES,
        build_arg_parser,
        figure_size,
        load_theme,
        resolve_runtime_physics,
        save_fig,
        surface_plot_tiles,
        validate_thesis_figure,
    )

_LAYOUT = [
    ("gbd", "TL", r"$\mathrm{GBD}$", (-4.5, 4.5), "left", "top"),
    ("rwm", "BL", r"$\mathrm{RWM}$", (-4.5, -4.5), "left", "bottom"),
    ("dvh", "BR", r"$\mathrm{DVH}$", (4.5, -4.5), "right", "bottom"),
    ("cs", "TR", r"$\mathrm{CS}$", (4.5, 4.5), "right", "top"),
]

# Plot


def plot_surface_fields(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"vortex_surface_fields.{fmt}"

    colors, theme = load_theme()
    runtime = resolve_runtime_physics(
        samples_dir, args.circulation, args.kinematic_viscosity, args.b0, args.a0_over_b0
    )
    ac0 = runtime["velocity_peak_radius0"]
    run_circulation = runtime["circulation"]
    uc_ref = run_circulation / (2.0 * np.pi * ac0)
    wc_ref = run_circulation / (np.pi * ac0**2)

    tiles, _ = surface_plot_tiles(
        samples_dir, _LAYOUT, ac0, uc_ref, wc_ref, runtime["kinematic_viscosity"]
    )
    if not tiles:
        out.unlink(missing_ok=True)
        print("  [surface] no sampled fields; figure not generated")
        return 0

    # -- Shared normalisation limits ------------------------------------
    v_norm = mcolors.Normalize(vmin=0.0, vmax=0.2)
    w_norm = mcolors.Normalize(vmin=0.0, vmax=0.1)
    v_cmap = theme.COLORMAPS["vortex_speed"]
    w_cmap = theme.COLORMAPS["vortex_vorticity"]

    ax_lim = 5.0

    # -- Figure --------------------------------------------------------
    cm = 1 / 2.54
    width_cm, height_cm = 12.5, 5.4
    fig = plt.figure(figsize=(width_cm * cm, height_cm * cm))
    outer = 0.13
    bottom, top = 0.29, 0.86
    axis_height = top - bottom
    axis_width = axis_height * height_cm / width_cm
    colorbar_gap, colorbar_width = 0.015, 0.015
    ax_v = fig.add_axes([outer, bottom, axis_width, axis_height])
    cax_v = fig.add_axes([outer + axis_width + colorbar_gap, bottom, colorbar_width, axis_height])
    cax_w_x = 1.0 - outer - colorbar_width
    ax_w_x = cax_w_x - colorbar_gap - axis_width
    ax_w = fig.add_axes([ax_w_x, bottom, axis_width, axis_height])
    cax_w = fig.add_axes([cax_w_x, bottom, colorbar_width, axis_height])

    labels = {scheme: (label, ha, va) for scheme, _, label, _, ha, va in _LAYOUT}
    for tile in tiles:
        label, ha, va = labels[tile["scheme"]]
        pcm_kw = dict(shading="gouraud", rasterized=True)
        ax_v.pcolormesh(tile["x"], tile["y"], tile["velocity"], cmap=v_cmap, norm=v_norm, **pcm_kw)
        ax_w.pcolormesh(tile["x"], tile["y"], tile["vorticity"], cmap=w_cmap, norm=w_norm, **pcm_kw)

        tx = -0.85 * ax_lim if ha == "left" else 0.85 * ax_lim
        ty = 0.85 * ax_lim if va == "top" else -0.85 * ax_lim
        txt_kw = dict(
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.15", fc=colors["LightText"], alpha=0.75, lw=0),
        )
        ax_v.text(tx, ty, label, **txt_kw)
        ax_w.text(tx, ty, label, **txt_kw)

    # Contro division lines
    divider_kw = dict(color=colors["LightText"], linewidth=0.5, alpha=1.0)
    for ax in (ax_v, ax_w):
        ax.axhline(0, **divider_kw)
        ax.axvline(0, **divider_kw)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_xticks((-4.0, 0.0, 4.0))
        ax.set_yticks((-4.0, 0.0, 4.0))
        ax.set_xlabel(r"$x\,/\,a_{c,0}$")
        ax.set_ylabel(r"$y\,/\,a_{c,0}$")

    # Both panels use the same y coordinate; one labelled shared scale avoids
    # a redundant centre label colliding with the first panel's colour bar.
    ax_w.set_ylabel("")
    ax_w.tick_params(axis="y", labelleft=False)

    ax_v.set_title(r"(a) $|\mathbf{u}|/U_{c,0}$")
    ax_w.set_title(r"(b) $\omega_z/\omega_{c,0}$")

    sm_v = ScalarMappable(cmap=v_cmap, norm=v_norm)
    sm_v.set_array([])
    sm_w = ScalarMappable(cmap=w_cmap, norm=w_norm)
    sm_w.set_array([])
    cb_v = fig.colorbar(sm_v, cax=cax_v)
    cb_w = fig.colorbar(sm_w, cax=cax_w)
    for colorbar in (cb_v, cb_w):
        colorbar.ax.tick_params(pad=2)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    validate_thesis_figure(fig, (ax_v, cax_v, ax_w, cax_w))
    save_fig(fig, out, args.dpi)
    return 0


def parse_args() -> argparse.Namespace:
    p = build_arg_parser("Cross-sectional surface field tiled comparison.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return plot_surface_fields(args)


if __name__ == "__main__":
    raise SystemExit(main())
