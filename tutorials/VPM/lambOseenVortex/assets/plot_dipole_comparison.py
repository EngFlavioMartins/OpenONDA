#!/usr/bin/env python3
"""Counter-rotating vortex dipole — core trajectory and radius comparison.

Reads VTS z=0 samples from each viscous scheme and plots:
  - core x-position  xc / b0  vs  ν t / b0²
  - core radius       a_c / a_{c,0}  vs  ν t / b0²

Saves: figures/dipole_comparison.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    SCHEMES,
    build_arg_parser,
    build_style_map,
    centroid,
    load_theme,
    pvd_time_map,
    publication_size,
    resolve_runtime_physics,
    save_publication_figure,
)


def _weighted_core_radius(points: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    wt = float(w.sum())
    if wt <= 1e-30:
        return np.nan
    r2 = ((points - center) ** 2).sum(axis=1)
    return float(np.sqrt((w * r2).sum() / wt))


# =============================================================
# Data extraction
# =============================================================


def extract_dipole_timeseries(
    samples_dir: Path,
    scheme: str,
    b0: float,
    target_time: float,
    tolerance: float,
) -> dict | None:
    import pyvista as pv

    case_samples_dir = samples_dir / f"dipole_{scheme}"
    vts_list = (
        sorted(
            [
                (int(m.group(1)), p)
                for p in case_samples_dir.glob(f"dipole_{scheme}_z0_*.vts")
                if (m := re.search(r"_(\d+)\.vts$", p.name))
            ],
            key=lambda x: x[0],
        )
        if case_samples_dir.exists()
        else []
    )

    time_map = pvd_time_map(samples_dir, "dipole", scheme)
    if not time_map or max(time_map.values()) < target_time - tolerance:
        return None
    rows = []
    for step, vts_path in vts_list:
        if step not in time_map:
            continue
        t = time_map[step]
        try:
            grid = pv.read(str(vts_path))
            xy = grid.points[:, :2].astype(np.float64)
            omega_z = grid.point_data["Vorticity"][:, 2].astype(np.float64)
        except Exception:
            continue
        mask = omega_z > 0.0
        if np.count_nonzero(mask) < 2:
            continue
        w = np.abs(omega_z[mask])
        if w.sum() < 1e-30:
            continue
        c = centroid(xy[mask], omega_z[mask])
        if np.any(np.isnan(c)):
            continue
        a_c = _weighted_core_radius(xy[mask], w, c)
        rows.append((t, float(c[0]), float(c[1]), a_c, float(w.sum())))

    if not rows:
        return None
    d = np.array(rows, dtype=float)
    return {"t": d[:, 0], "x_core": d[:, 1], "a_c": d[:, 3], "total_gamma": d[:, 4]}


# =============================================================
# Plot
# =============================================================


def plot_dipole_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"dipole_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    runtime = resolve_runtime_physics(samples_dir, args.gamma, args.nu, args.b0, args.a0_over_b0)
    run_nu = runtime["nu"]
    a0 = runtime["ac0"]
    colors, _ = load_theme()
    style_map = build_style_map(colors)

    fig, axes = plt.subplots(1, 2, figsize=publication_size(7.5))
    fig.subplots_adjust(wspace=0.43, bottom=0.34, top=0.88, left=0.13, right=0.97)

    plotted_schemes = []
    tolerance = max(0.5, 20.0 * args.dt)
    for scheme in SCHEMES:
        ts = extract_dipole_timeseries(
            samples_dir,
            scheme,
            args.b0,
            args.total_time,
            tolerance,
        )
        if ts is None:
            print(f"  [dipole] skipping {scheme!r} — no data")
            continue
        t = ts["t"]
        xc = ts["x_core"]
        a_c = ts["a_c"]
        mask = np.isfinite(xc) & np.isfinite(a_c)
        t, xc, a_c = t[mask], xc[mask], a_c[mask]
        if len(t) == 0:
            continue
        tau = run_nu * t / (a0**2)
        st = style_map[scheme]
        plot_kw = {
            "color": st["color"],
            "label": st["label"],
            "marker": st["marker"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
        }
        a_c_norm = a_c / a_c[0] if a_c[0] > 0 else a_c
        axes[0].plot(tau, xc / args.b0, **plot_kw)
        axes[1].plot(tau, a_c_norm, **plot_kw)
        plotted_schemes.append(scheme)

    if len(plotted_schemes) != len(SCHEMES):
        plt.close(fig)
        out.unlink(missing_ok=True)
        print(
            f"  [dipole] complete trajectories available for {len(plotted_schemes)}/"
            f"{len(SCHEMES)} methods; figure not generated"
        )
        return 1

    axes[0].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[0].set_ylabel(r"$x_c / b_0$")
    axes[0].set_title("Core trajectory over time")
    axes[0].set_ylim([0.0, 2.0])
    axes[1].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[1].set_ylabel(r"$a_c / a_{c,0}$")
    axes[1].set_title(r"Core radius over time")
    axes[1].set_ylim([0.9, 4.0])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.05),
        )
    save_publication_figure(fig, out, args.dpi)
    return 0


def main() -> int:
    p = build_arg_parser("Counter-rotating dipole trajectory and core-radius comparison.")
    return plot_dipole_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
