#!/usr/bin/env python3
"""Counter-rotating vortex dipole — core trajectory and radius comparison.

Reads HDF5 backup snapshots from each viscous scheme and plots:
  - core x-position  $x_c / b_0$  vs  $\nu t / b_0^2$
  - core radius       $r_c / r_{c,0}$  vs  $\nu t / b_0^2$

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
    centroid,
    load_theme,
    pvd_time_map,
    resolve_runtime_physics,
)


# ── Particle helpers ──────────────────────────────────────────────────────────


def h5_files(solution_dir: Path, prefix: str, scheme: str) -> list[Path]:
    folder = solution_dir / f"{prefix}_{scheme}"
    return sorted(
        folder.glob(f"vpm_{prefix}_{scheme}_*.h5"),
        key=lambda p: int(re.search(r"_(\d+)\.h5$", p.name).group(1)),
    )


def read_h5(path: Path):
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
        return t, pos, circ[:, 2]
    except (OSError, KeyError) as exc:
        print(f"  [warn] skipping unreadable backup {path.name}: {exc}")
        return None, None, None, None


def _weighted_core_radius(points: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    r"""Kernel-consistent core radius from the vorticity 2nd moment.

    Returns ``a = sqrt(Σ ω r² / Σ ω)`` — the exact core radius of a Lamb-Oseen
    profile ``ω = ω₀·exp(-r²/a²)`` (for which ⟨r²⟩ = a²), measured directly from
    the ``ω``-weighted spatial spread about *center*.

    Why this and not a Gaussian fit of the reconstructed field?  Each viscous
    scheme stores its diffusion differently: CS widens the *reconstruction
    kernel* (particle radius grows to ~4.4 r_c0) while keeping particles
    frozen, whereas DVH/GBD keep the kernel pinned at 2.5h and carry the
    spread in the *particle positions*.  A Gaussian fit of the reconstructed
    field therefore measures kernel width for CS but particle spread for
    DVH/GBD — an apples-to-oranges comparison, and the (truncated) fit further
    clips the wide CS kernel.  The raw 2nd moment applies the *same* operator
    to every scheme's field; self-normalising each curve by its t=0 value (in
    the plot) divides out the residual reconstruction offset, leaving the
    physical relative growth r_c(t)/r_c(0).
    """
    w = np.asarray(weights, dtype=np.float64)
    wt = float(w.sum())
    if wt <= 1e-30:
        return np.nan
    r2 = ((points - center) ** 2).sum(axis=1)
    return float(np.sqrt((w * r2).sum() / wt))


def extract_dipole_timeseries(solution_dir: Path, scheme: str, b0: float) -> dict | None:
    """Extract core trajectory and kernel-consistent core radius for the dipole.

    Prefers VTS z=0 grid data (actual reconstructed vorticity field) and
    measures the core radius via the ω-weighted 2nd moment
    (see ``_weighted_core_radius``); falls back to HDF5 particle data.
    """
    import pyvista as pv

    # ── Primary: VTS grid data (z=0 plane) ──
    samples_dir = solution_dir / f"dipole_{scheme}" / "samples"
    vts_list = sorted(
        [
            (int(m.group(1)), p)
            for p in samples_dir.glob(f"dipole_{scheme}_z0_*.vts")
            if (m := re.search(r"_(\d+)\.vts$", p.name))
        ],
        key=lambda x: x[0],
    ) if samples_dir.exists() else []

    time_map = pvd_time_map(solution_dir, "dipole", scheme)
    rows = []
    if vts_list and time_map:
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
            w = np.abs(omega_z[mask])
            if w.sum() < 1e-30:
                continue
            c = centroid(xy[mask], omega_z[mask])
            if np.any(np.isnan(c)):
                continue
            # Kernel-consistent 2nd moment over the full positive-ω region
            # (no r_max truncation — that would clip CS's wide kernel).
            rc = _weighted_core_radius(xy[mask], w, c)
            rows.append((t, float(c[0]), float(c[1]), rc, float(w.sum())))
    else:
        # ── Fallback: HDF5 particle data ──
        files = h5_files(solution_dir, "dipole", scheme)
        for p in files:
            t, pos, gz = read_h5(p)
            if pos is None:
                continue
            # Select particles near z=0 (central third of the column)
            z = pos[:, 2]
            z_lo, z_hi = z.min(), z.max()
            z_rng = z_hi - z_lo
            z_mask = (z >= z_lo + z_rng / 3.0) & (z <= z_hi - z_rng / 3.0)
            pos2d = pos[z_mask, :2]
            gz2d = gz[z_mask]

            mask_pos = gz2d > 0.0
            if mask_pos.sum() < 2:
                continue
            c = centroid(pos2d[mask_pos], gz2d[mask_pos])
            if np.any(np.isnan(c)):
                continue
            # Kernel-consistent 2nd moment of the positive-Γ particle cloud.
            # NOTE: from particle data this measures the *position* spread only
            # (it excludes the reconstruction-kernel width that the VTS-field
            # path captures), so it is self-consistent for relative growth but
            # not directly comparable to the VTS numbers across schemes.  VTS
            # data exists for every scheme here, so this branch is a safety net.
            rc = _weighted_core_radius(pos2d[mask_pos], gz2d[mask_pos], c)
            rows.append((t, float(c[0]), float(c[1]), rc, float(np.abs(gz2d[mask_pos]).sum())))

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

    fig, axes = plt.subplots(1, 2, figsize=(12.8 / 2.54, 8.5 / 2.54))
    fig.subplots_adjust(wspace=0.25, bottom=0.30, top=0.92, left=0.08, right=0.92)

    for scheme in SCHEMES:
        ts = extract_dipole_timeseries(solution_dir, scheme, args.b0)
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
        tau = run_nu * t / (args.b0**2)
        st = style_map[scheme]
        kw = {
            "color": st["color"],
            "label": st["label"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
            "marker": st["marker"],
        }
        # Self-normalise each scheme by its own t=0 core radius so the y-axis
        # is the physical relative growth r_c(t)/r_c(0).  This divides out the
        # per-scheme reconstruction-kernel offset (see _weighted_core_radius),
        # making the four schemes' core-growth directly comparable.
        rc_norm = rc / rc[0] if rc[0] > 0 else rc
        axes[0].plot(tau, xc / args.b0, **kw)
        axes[1].plot(tau, rc_norm, **kw)

    axes[0].set_xlabel(r"Normalized time, $\nu t / b_0^2$")
    axes[0].set_ylabel(r"Normalized core trajectory, $x_c / b_0$")
    axes[0].set_title("Core trajectory over time")
    axes[0].set_ylim([0.0, 4.5])
    axes[1].set_xlabel(r"Normalized time, $\nu t / b_0^2$")
    axes[1].set_ylabel(r"Normalized core radius, $r_c / r_{c,0}$")
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
