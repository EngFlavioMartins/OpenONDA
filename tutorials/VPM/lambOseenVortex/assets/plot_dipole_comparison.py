#!/usr/bin/env python3
"""Counter-rotating vortex dipole — core trajectory and radius comparison.

Reads HDF5 backup snapshots from each viscous scheme and plots:
  - core x-position  x_c / d  vs  nu t / d²
  - core radius       r_c / r_{c,0}  vs  nu t / d²

Saves: figures/dipole_comparison.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    SCHEMES,
    add_physics_args,
    build_arg_parser,
    build_style_map,
    load_theme,
    pvd_time_map,
    resolve_runtime_physics,
)


# ── HDF5 helpers ──────────────────────────────────────────────────────────────


def h5_files(solution_dir: Path, prefix: str, scheme: str) -> list[Path]:
    folder = solution_dir / f"{prefix}_{scheme}"
    return sorted(
        folder.glob(f"vpm_{prefix}_{scheme}_*.h5"),
        key=lambda p: int(re.search(r"_(\d+)\.h5$", p.name).group(1)),
    )


def read_h5(path: Path):
    # Guard against truncated/corrupt backups (e.g. from an interrupted run):
    # a bad file raises OSError ("file signature not found") — skip it rather
    # than killing the whole figure. A zero-byte / too-small file (the run died
    # mid-write) is detected up front so we emit a clear message instead of
    # h5py's cryptic "file signature not found".
    if not path.is_file() or path.stat().st_size < 4096:
        print(f"  [warn] skipping truncated backup {path.name} "
              f"({path.stat().st_size if path.is_file() else 0} bytes)")
        return None, None, None, None
    try:
        with h5py.File(path, "r") as f:
            t = float(f["solver"].attrs["flow_time"])
            n = int(f["solver"].attrs["number_of_particles"])
            if n == 0:
                return t, None, None, None
            pos = f["particles"]["position"][:n].astype(np.float64)
            circ = f["particles"]["circulation"][:n].astype(np.float64)
            radius = f["particles"]["radius"][:n].astype(np.float64)
        return t, pos, circ[:, 2], radius
    except (OSError, KeyError) as exc:
        print(f"  [warn] skipping unreadable backup {path.name}: {exc}")
        return None, None, None, None


def core_properties(pos: np.ndarray, gz: np.ndarray, sign: float = 1.0):
    mask = gz > 0.0 if sign > 0.0 else gz < 0.0
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan, np.nan, 0.0
    w = np.abs(gz[mask])
    wt = float(w.sum())
    if wt < 1e-30:
        return np.nan, np.nan, np.nan, 0.0
    xc = float(np.sum(w * pos[mask, 0]) / wt)
    yc = float(np.sum(w * pos[mask, 1]) / wt)
    r2 = (pos[mask, 0] - xc) ** 2 + (pos[mask, 1] - yc) ** 2
    rc = float(np.sqrt(np.sum(w * r2) / wt))
    return xc, yc, rc, wt


def core_properties_cs(pos: np.ndarray, gz: np.ndarray, radius: np.ndarray, sign: float = 1.0):
    mask = gz > 0.0 if sign > 0.0 else gz < 0.0
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan, np.nan, 0.0
    w = np.abs(gz[mask])
    wt = float(w.sum())
    if wt < 1e-30:
        return np.nan, np.nan, np.nan, 0.0
    xc = float(np.sum(w * pos[mask, 0]) / wt)
    yc = float(np.sum(w * pos[mask, 1]) / wt)
    rc = float(np.sqrt(np.sum(w * radius[mask] ** 2) / wt))
    return xc, yc, rc, wt


def extract_dipole_timeseries(solution_dir: Path, scheme: str) -> dict | None:
    files = h5_files(solution_dir, "dipole", scheme)
    if files:
        rows = []
        for p in files:
            t, pos, gz, radius = read_h5(p)
            if pos is None:
                continue
            if scheme == "cs":
                xc, yc, rc, gam = core_properties_cs(pos, gz, radius)
            else:
                xc, yc, rc, gam = core_properties(pos, gz)
            rows.append((t, xc, yc, rc, gam, len(gz)))
        if rows:
            d = np.array(rows, dtype=float)
            return {"t": d[:, 0], "x_core": d[:, 1], "r_core": d[:, 3], "total_gamma": d[:, 4]}

    # --- VTS fallback (no HDF5 backup files available) ---
    import pyvista as pv

    samples_dir = solution_dir / f"dipole_{scheme}" / "samples"
    if not samples_dir.exists():
        return None
    vts_list = sorted(
        [
            (int(m.group(1)), p)
            for p in samples_dir.glob(f"dipole_{scheme}_z0_*.vts")
            if (m := re.search(r"_(\d+)\.vts$", p.name))
        ],
        key=lambda x: x[0],
    )
    if not vts_list:
        return None

    time_map = pvd_time_map(solution_dir, "dipole", scheme)
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
        # Positive-circulation vortex core
        mask = omega_z > 0.0
        if np.count_nonzero(mask) < 2:
            continue
        w = omega_z[mask]
        wt = float(w.sum())
        if wt < 1e-30:
            continue
        xc = float(np.dot(w, xy[mask, 0]) / wt)
        yc = float(np.dot(w, xy[mask, 1]) / wt)
        r2 = (xy[mask, 0] - xc) ** 2 + (xy[mask, 1] - yc) ** 2
        rc = float(np.sqrt(np.dot(w, r2) / wt))
        rows.append((t, xc, yc, rc, wt))
    if not rows:
        return None
    d = np.array(rows, dtype=float)
    return {"t": d[:, 0], "x_core": d[:, 1], "r_core": d[:, 3], "total_gamma": d[:, 4]}


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_dipole_case(args) -> int:
    solution_dir = Path(args.solution_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"dipole_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    runtime = resolve_runtime_physics(solution_dir, args.gamma, args.nu, args.b0, args.a0_over_b0)
    run_nu = runtime["nu"]
    a0 = runtime["rc0"]
    colors, _ = load_theme()
    style_map = build_style_map(colors)

    fig, axes = plt.subplots(1, 2, figsize=(12.8 / 2.54, 7.5 / 2.54))
    fig.subplots_adjust(wspace=0.25, bottom=0.30, top=0.92, left=0.08, right=0.92)

    for scheme in SCHEMES:
        ts = extract_dipole_timeseries(solution_dir, scheme)
        if ts is None:
            print(f"  [dipole] skipping {scheme!r} — no data")
            continue
        t = ts["t"]
        xc = ts["x_core"]
        rc = ts["r_core"]
        mask = np.isfinite(xc) & np.isfinite(rc)
        t, xc, rc = t[mask], xc[mask], rc[mask]
        if len(t) == 0:
            continue
        tau = run_nu * t / (args.separation**2)
        st = style_map[scheme]
        kw = {
            "color": st["color"],
            "label": st["label"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
            "marker": st["marker"],
        }
        axes[0].plot(tau, xc / args.separation, **kw)
        axes[1].plot(tau, rc / a0, **kw)

    axes[0].set_xlabel(r"Normalized time, $\nu t / d_0^2$")
    axes[0].set_ylabel(r"Normalized core trajectory, $x_c / d$")
    axes[0].set_title("Core trajectory over time")
    axes[0].set_ylim([0.0, 2.0])
    axes[1].set_xlabel(r"Normalized time, $\nu t / d_0^2$")
    axes[1].set_ylabel(r"Normalized core radius, $r_c / r_{c,0}$")
    axes[1].set_title(r"Core radius over time")
    axes[1].set_ylim([0.7, 3.5])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.05),
            fontsize=10,
        )
    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    p = build_arg_parser("Counter-rotating dipole trajectory and core-radius comparison.")
    add_physics_args(p)
    return plot_dipole_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
