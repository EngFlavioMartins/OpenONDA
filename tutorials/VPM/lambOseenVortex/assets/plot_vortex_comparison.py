#!/usr/bin/env python3
"""Lamb-Oseen single vortex — radial profile comparison.

Reads the last x-line CSV sample from each viscous scheme and plots:
  - azimuthal velocity  uθ / U_{c,0}
  - z-vorticity         ωz / ω_{c,0}
  - velocity gradient   (∂uy/∂x) · a_{c,0} / U_{c,0}

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
    build_arg_parser,
    build_style_map,
    load_theme,
    publication_size,
    read_column_half_length,
    resolve_runtime_physics,
    save_publication_figure,
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
    samples_dir: Path,
    scheme: str,
    step: int | None,
    dt: float,
    target_time: float | None = None,
):
    folder = samples_dir / f"vortex_{scheme}"
    path = folder / f"vortex_{scheme}_x.csv"
    if not path.exists():
        return None, None

    data = pd.read_csv(path)
    if data.empty or data["Uy"].abs().max() <= 1e-10:
        return None, None
    if step is not None:
        available_steps = data["time_step"].dropna().astype(int).unique()
        if step not in available_steps:
            return None, None
        selected_t = float(data.loc[data["time_step"] == step, "flow_time"].iloc[0])
        return path, selected_t

    available_times = np.sort(data["flow_time"].unique())
    selected_t = float(available_times[-1])
    if target_time is not None:
        selected_t = float(available_times[np.argmin(np.abs(available_times - target_time))])
    tolerance = max(0.5, 20.0 * dt)
    if target_time is not None and abs(selected_t - target_time) > tolerance:
        print(
            f"  [skip] {scheme.upper()} final sample is t={selected_t:.3g}s, "
            f"not near requested t={target_time:.3g}s."
        )
        return None, None
    return path, selected_t


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
    half_length = read_column_half_length(samples_dir) or 25.0 * ac0

    target_time = None if args.step is not None else args.total_time

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=publication_size(13.5))
    fig.subplots_adjust(hspace=0.12, top=0.94, bottom=0.18, left=0.17, right=0.97)

    scheme_data: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for scheme in SCHEMES:
        path, t = last_csv(samples_dir, scheme, args.step, args.dt, target_time)
        if path is None:
            continue
        df = pd.read_csv(path)
        df = df[np.isclose(df["flow_time"], t)]
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

    if len(scheme_data) != len(SCHEMES):
        plt.close(fig)
        out.unlink(missing_ok=True)
        print(
            f"  [vortex] complete profiles available for {len(scheme_data)}/{len(SCHEMES)} "
            "methods; figure not generated"
        )
        return 1

    elapsed_time = (
        args.total_time
        if target_time is not None
        else (float(np.median([s[1] for s in scheme_data])) if scheme_data else args.total_time)
    )

    r_line = np.linspace(-10.0 * ac0, 10.0 * ac0, 400)
    ref_kw = {"color": colors["reference"], "lw": 1.0, "zorder": 0, "linestyle": "--"}
    theory_t = run_t0 + elapsed_time
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

    axes[2].set_xlabel(r"$r / a_{c,0}$")
    axes[2].set_ylabel(r"$(\partial u_y / \partial x)\,a_{c,0} / U_{c,0}$")
    axes[2].set_xlim([-7.5, 7.5])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    save_publication_figure(fig, out, args.dpi)
    return 0


def main() -> int:
    p = build_arg_parser("Lamb-Oseen single-vortex radial profile comparison.")
    return plot_vortex_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
