#!/usr/bin/env python3
"""Lamb-Oseen single vortex — radial profile comparison.

Reads the last z=L/4 surface-field snapshot from each viscous scheme,
slices the grid row nearest y=0, and plots:
  - azimuthal velocity  uθ / U_{c,0}
  - z-vorticity         ωz / ω_{c,0}
  - velocity gradient   (∂uy/∂x) · a_{c,0} / U_{c,0}

Saves: figures/vortex_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .plot_style import build_arg_parser, build_style_map, figure_size, load_theme, save_fig
    from .vortex_diagnostics import (
        SCHEMES,
        TOTAL_TIME,
        pvd_time_map,
        read_surface_field,
        resolve_runtime_physics,
    )
else:
    from plot_style import build_arg_parser, build_style_map, figure_size, load_theme, save_fig
    from vortex_diagnostics import (
        SCHEMES,
        TOTAL_TIME,
        pvd_time_map,
        read_surface_field,
        resolve_runtime_physics,
    )

from matplotlib.ticker import FormatStrFormatter


# =============================================================
# Theory
# =============================================================


def lamb_oseen_profile(r: np.ndarray, t: float, gamma: float, nu: float):
    rc2 = 4.0 * nu * t
    rc = np.sqrt(rc2)
    oz = (gamma / (np.pi * rc2)) * np.exp(-(r**2) / rc2)
    vel = np.zeros_like(r)
    mask = np.abs(r) > 1e-12
    vel[mask] = gamma / (2.0 * np.pi * r[mask]) * (1.0 - np.exp(-(r[mask] ** 2) / rc2))
    return vel, oz, rc


def lamb_oseen_gradient(r: np.ndarray, t: float, gamma: float, nu: float) -> np.ndarray:
    rc2 = 4.0 * nu * t
    grad = np.zeros_like(r)
    mask = np.abs(r) > 1e-12
    exp_t = np.exp(-(r[mask] ** 2) / rc2)
    grad[mask] = (gamma / (2.0 * np.pi)) * (2.0 * exp_t / rc2 - (1.0 - exp_t) / r[mask] ** 2)
    grad[~mask] = gamma / (2.0 * np.pi * rc2)
    return grad


# =============================================================
# Data loader
# =============================================================


def load_profile(
    samples_dir: Path, scheme: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Slice the y≈0 row out of the last sampled z=L/4 field for this scheme."""

    timeline = pvd_time_map(samples_dir, "vortex", scheme)
    if not timeline:
        return None
    last_step = max(timeline, key=timeline.get)
    vts = samples_dir / f"vortex_{scheme}" / f"vortex_{scheme}_zq_{last_step:06d}.vts"
    if not vts.is_file():
        return None

    field = read_surface_field(vts)
    if np.abs(field["Uy"]).max() <= 1e-10:
        return None

    # SurfaceSampler's grid is built with np.arange, so an exact y=0 row
    # isn't guaranteed — take the row nearest to it.
    y_1d = field["y"][0, :]
    j0 = int(np.argmin(np.abs(y_1d)))
    x = field["x"][:, j0]
    uy = field["Uy"][:, j0]
    oz = field["omega_z"][:, j0]
    return x, uy, oz, timeline[last_step]


# =============================================================
# Plot
# =============================================================


def plot_vortex_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"vortex_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    colors, _ = load_theme()
    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(samples_dir, args.gamma, args.nu, args.b0, args.a0_over_b0)
    run_nu = runtime["nu"]
    run_t0 = runtime["t0"]
    ac0 = runtime["ac0"]

    uc_ref = args.gamma / (2.0 * np.pi * ac0)
    wc_ref = args.gamma / (np.pi * ac0**2)
    gc_ref = uc_ref / ac0

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("stacked_tall"))
    fig.subplots_adjust(hspace=0.12, top=0.95, bottom=0.18, left=0.12, right=0.98)

    scheme_data: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for scheme in SCHEMES:
        profile = load_profile(samples_dir, scheme)
        if profile is None:
            continue
        x, uy, oz, t = profile
        dvx = np.gradient(uy, x)
        st = style_map[scheme]
        plot_kw = {
            "color": st["color"],
            "label": st["label"],
            "marker": st["marker"],
            "markersize": 2.0,
            "markevery": 1,
            "linestyle": "None",
            "linewidth": 1.0,
        }
        axes[0].plot(x / ac0, uy / uc_ref, **plot_kw)
        axes[1].plot(x / ac0, oz / wc_ref, **plot_kw)
        axes[2].plot(x / ac0, dvx / gc_ref, **plot_kw)
        scheme_data.append((scheme, t, x, uy, oz, dvx))

    if scheme_data:
        elapsed_time = float(np.median([scheme[1] for scheme in scheme_data]))
        print(
            f"  [vortex] plotting {len(scheme_data)}/{len(SCHEMES)} methods at t={elapsed_time:.3g}s"
        )
    else:
        elapsed_time = TOTAL_TIME
        print(f"  [vortex] no sampled profiles; plotting reference only at t={elapsed_time:.3g}s")

    r_line = np.linspace(-10.0 * ac0, 10.0 * ac0, 400)
    ref_kw = {"color": colors["reference"], "lw": 1.1, "zorder": 100, "linestyle": "-"}
    theory_t = run_t0 + elapsed_time
    tv, to, _ = lamb_oseen_profile(r_line, theory_t, args.gamma, run_nu)
    tg = np.gradient(tv, r_line)
    axes[0].plot(r_line / ac0, tv / uc_ref, label="Theory", **ref_kw)
    axes[1].plot(r_line / ac0, to / wc_ref, **ref_kw)
    axes[2].plot(r_line / ac0, tg / gc_ref, **ref_kw)

    axes[0].set_title(r"Single vortex characteristics")
    axes[0].set_ylabel(r"$u_\theta / U_{c,0}$")
    axes[0].set_xlim([-5.5, 5.5])
    axes[0].set_ylim([-0.25, 0.25])

    axes[1].set_ylabel(r"$\omega_z / \omega_{c,0}$")
    axes[1].set_ylim([-0.01, 0.1])

    axes[2].set_xlabel(r"$r / a_{c,0}$")
    axes[2].set_ylabel(r"$(\partial u_y / \partial x)\,a_{c,0} / U_{c,0}$")
    axes[2].set_ylim([-0.05, 0.13])

    handles, labels = axes[0].get_legend_handles_labels()

    for ax in axes:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save_fig(fig, out, args.dpi)
    return 0


def main() -> int:
    p = build_arg_parser("Lamb-Oseen single-vortex radial profile comparison.")
    return plot_vortex_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
