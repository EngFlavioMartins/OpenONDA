#!/usr/bin/env python3
"""Lamb-Oseen single vortex — radial profile comparison.

Reads the last x-line CSV sample from each viscous scheme and plots:
  - azimuthal velocity uθ / Uc,0
  - z-vorticity  ωz / ωc
  - velocity gradient (∂uy/∂x) · rc,0 / Uc,0

Saves: figures/vortex_comparison.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    SCHEMES,
    add_physics_args,
    build_arg_parser,
    build_style_map,
    load_theme,
    read_column_half_length,
    read_flow_time,
    resolve_runtime_physics,
)


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


def finite_column_velocity(
    x: np.ndarray,
    t: float,
    gamma: float,
    nu: float,
    half_length: float,
) -> np.ndarray:
    r = np.abs(x)
    vel, _, _ = lamb_oseen_profile(r, t, gamma, nu)
    span_factor = half_length / np.sqrt(half_length**2 + r**2)
    return vel * span_factor * np.sign(x)


# =============================================================
# Data loader
# =============================================================


def last_csv(
    solution_dir: Path,
    scheme: str,
    step: int | None,
    dt: float,
    target_time: float | None = None,
):
    folder = solution_dir / f"vortex_{scheme}" / "samples"
    if step is not None:
        c = folder / f"vortex_{scheme}_x_{step:06d}.csv"
        if c.exists():
            t = read_flow_time(c)
            return c, t if t is not None else step * dt
        return None, None
    candidates = sorted(folder.glob(f"vortex_{scheme}_x_*.csv"))
    if not candidates:
        return None, None

    usable = []
    for candidate in candidates:
        df_check = pd.read_csv(candidate, comment="#")
        if df_check["Uy"].abs().max() <= 1e-10:
            continue
        t = read_flow_time(candidate)
        if t is None:
            s = int(candidate.stem.split("_")[-1])
            t = s * dt
        usable.append((candidate, t))
    if not usable:
        return None, None

    if target_time is None:
        return usable[-1]

    selected, selected_t = min(usable, key=lambda item: abs(item[1] - target_time))
    tolerance = max(0.5, 20.0 * dt)
    if abs(selected_t - target_time) > tolerance:
        print(
            f"  [skip] {scheme.upper()} final sample is t={selected_t:.3g}s, "
            f"not near requested t={target_time:.3g}s."
        )
        return None, None
    return selected, selected_t


# =============================================================
# Plot
# =============================================================


def plot_vortex_case(args) -> int:
    solution_dir = Path(args.solution_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"vortex_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    colors, _ = load_theme()
    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(solution_dir, args.gamma, args.nu, args.b0, args.a0_over_b0)
    run_nu = runtime["nu"]
    run_t0 = runtime["t0"]
    ac0 = runtime["ac0"]

    uc_ref = args.gamma / (2.0 * np.pi * ac0)
    wc_ref = args.gamma / (np.pi * ac0**2)
    gc_ref = uc_ref / ac0
    half_length = read_column_half_length(solution_dir) or 25.0 * ac0

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12.8 / 2.54, 12.8 / 2.54))
    fig.subplots_adjust(hspace=0.1, top=0.95, bottom=0.19, left=0.15, right=0.85)

    scheme_data: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for scheme in SCHEMES:
        path, t = last_csv(solution_dir, scheme, args.step, args.dt, None)
        if path is None:
            continue
        df = pd.read_csv(path, comment="#")
        x = df["x"].to_numpy()
        uy = df["Uy"].to_numpy()
        oz = df["omega_z"].to_numpy()
        dvx = np.gradient(uy, x)
        st = style_map[scheme]
        plot_kw = {
            "color": st["color"],
            "label": st["label"],
            "marker": st["marker"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
        }
        axes[0].plot(x / ac0, uy / uc_ref, **plot_kw)
        axes[1].plot(x / ac0, oz / wc_ref, **plot_kw)
        axes[2].plot(x / ac0, dvx / gc_ref, **plot_kw)
        scheme_data.append((scheme, t, x, uy, oz, dvx))

    r_line = np.linspace(-10.0 * ac0, 10.0 * ac0, 400)
    ref_kw = {"color": "gray", "lw": 1.0, "zorder": 0, "linestyle": "--"}
    theory_t = run_t0 + args.total_time
    tv = finite_column_velocity(r_line, theory_t, args.gamma, run_nu, half_length)
    to = lamb_oseen_profile(np.abs(r_line), theory_t, args.gamma, run_nu)[1]
    tg = np.gradient(tv, r_line)
    axes[0].plot(r_line / ac0, tv / uc_ref, label="Theory", **ref_kw)
    axes[1].plot(r_line / ac0, to / wc_ref, **ref_kw)
    axes[2].plot(r_line / ac0, tg / gc_ref, **ref_kw)

    axes[0].set_title(r"Single vortex characteristics")
    axes[0].set_ylabel(r"$u_\theta / U_{c,0}$")
    axes[0].set_xlim([-7.5, 7.5])

    axes[1].set_ylabel(r"$\omega_z / \omega_{c,0}$")
    axes[1].set_xlim([-7.5, 7.5])
    axes[1].set_ylim(bottom=-0.01)

    axes[2].set_xlabel(r"$r / r_{c,0}$")
    axes[2].set_ylabel(r"$(\partial u_y / \partial x)\,r_{c,0} / U_{c,0}$")
    axes[2].set_xlim([-7.5, 7.5])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01) )
    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    p = build_arg_parser("Lamb-Oseen single-vortex radial profile comparison.")
    add_physics_args(p)
    return plot_vortex_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
