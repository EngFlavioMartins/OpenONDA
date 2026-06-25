#!/usr/bin/env python3
"""Lamb-Oseen single vortex — radial profile comparison.

Reads the last x-line CSV sample from each viscous scheme and plots:
  - azimuthal velocity u_θ / U_{c,0}
  - z-vorticity  ω_z / ω_c
  - velocity gradient (∂u_y/∂x) · r_{c,0} / U_{c,0}

Saves: figures/lamb_oseen_comparison.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    SCHEMES,
    add_physics_args,
    build_arg_parser,
    build_style_map,
    load_theme,
    read_flow_time,
    resolve_runtime_physics,
)


# ── Theory ───────────────────────────────────────────────────────────────────


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
    """Velocity for the finite straight vortex column used in the tutorial setup.

    The solver initializes a vortex column of finite span (z in [-4 rc0, 4 rc0]),
    not an infinite 2D vortex. The center-plane velocity is therefore weaker than
    the infinite Lamb-Oseen expression by the usual straight-segment factor.
    """
    r = np.abs(x)
    vel, _, _ = lamb_oseen_profile(r, t, gamma, nu)
    span_factor = half_length / np.sqrt(half_length**2 + r**2)
    return vel * span_factor * np.sign(x)


# ── Data loader ───────────────────────────────────────────────────────────────
def last_csv(solution_dir: Path, scheme: str, step: int | None, dt: float):
    """Return (path, flow_time) for the last stable x-profile CSV.

    Reads flow_time from the CSV header comment if available,
    falling back to step * dt reconstruction otherwise.
    For schemes that go unstable, picks the last file whose velocity
    data is non-trivial (max |Uy| > 1e-10).
    """
    folder = solution_dir / f"vortex_{scheme}" / "samples"
    if step is not None:
        c = folder / f"vortex_{scheme}_x_{step:06d}.csv"
        if c.exists():
            t = read_flow_time(c)
            return c, t if t is not None else step * dt
    candidates = sorted(folder.glob(f"vortex_{scheme}_x_*.csv"))
    if not candidates:
        return None, None
    # Walk backwards to find the last file with non-trivial data
    for last in reversed(candidates):
        df_check = pd.read_csv(last, comment="#")
        if df_check["Uy"].abs().max() > 1e-10:
            break
    else:
        last = candidates[-1]
    t = read_flow_time(last)
    if t is not None:
        return last, t
    s = int(last.stem.split("_")[-1])
    return last, s * dt


# ── Plot ──────────────────────────────────────────────────────────────────────


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
    rc_ref = runtime["rc0"]

    uc_ref = args.gamma / (2.0 * np.pi * rc_ref)
    wc_ref = args.gamma / (np.pi * rc_ref**2)
    gc_ref = uc_ref / rc_ref
    half_length = 4.0 * rc_ref

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12.8 / 2.54, 12.8 / 2.54))
    fig.subplots_adjust(hspace=0.25, top=0.95, bottom=0.19, left=0.15, right=0.85)

    scheme_data: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for scheme in SCHEMES:
        path, t = last_csv(solution_dir, scheme, args.step, args.dt)
        if path is None:
            continue
        df = pd.read_csv(path, comment="#")
        x = df["x"].to_numpy()
        uy = df["Uy"].to_numpy()
        oz = df["omega_z"].to_numpy()
        dvx = 2.0 * df["Sxy"].to_numpy()  # Sxy = 0.5*(dvdx + dudy); for 2D vortex dudy = -dvdx
        st = style_map[scheme]
        plot_kw = {
            "color": st["color"],
            "marker": st["marker"],
            "markersize": 2.2,
            "linewidth": 1.0,
            "linestyle": "None",
            "label": st["label"],
        }
        axes[0].plot(x / rc_ref, uy / uc_ref, **plot_kw)
        axes[1].plot(x / rc_ref, oz / wc_ref, **plot_kw)
        axes[2].plot(x / rc_ref, dvx / gc_ref, **plot_kw)
        scheme_data.append((scheme, t, x, uy, oz, dvx))

    r_line = np.linspace(-10.0 * rc_ref, 10.0 * rc_ref, 400)
    kw_theory = {"color": "black", "lw": 1.0, "zorder": 0, "linestyle": "--"}
    theory_t = run_t0 + args.total_time
    tv = finite_column_velocity(r_line, theory_t, args.gamma, run_nu, half_length)
    to = lamb_oseen_profile(np.abs(r_line), theory_t, args.gamma, run_nu)[1]
    tg = np.gradient(tv, r_line)
    axes[0].plot(r_line / rc_ref, tv / uc_ref, label="Theory", **kw_theory)
    axes[1].plot(r_line / rc_ref, to / wc_ref, **kw_theory)
    axes[2].plot(r_line / rc_ref, tg / gc_ref, **kw_theory)

    axes[0].set_title(r"Azimuthal velocity, $u_\theta / U_{c,0}$")
    axes[0].set_ylabel(r"$u_\theta / U_{c,0}$")
    axes[0].set_xlim([-7.5, 7.5])

    axes[1].set_title(r"Vorticity, $\omega_z / \omega_c$")
    axes[1].set_ylabel(r"$\omega_z / \omega_c$")
    axes[1].set_xlim([-7.5, 7.5])
    axes[1].set_ylim(bottom=-0.01)

    axes[2].set_title(r"Velocity gradient, $(\partial u_y / \partial x)\,r_{c,0} / U_{c,0}$")
    axes[2].set_xlabel(r"Normalized radius, $r / r_{c,0}$")
    axes[2].set_ylabel(r"$(\partial u_y / \partial x)\,r_{c,0} / U_{c,0}$")
    axes[2].set_xlim([-7.5, 7.5])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=10)
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
