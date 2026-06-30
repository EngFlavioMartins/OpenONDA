#!/usr/bin/env python3
"""Co-rotating vortex merger — rotation angle, core radius, and separation.

Reads VTS z=0 sample files (or HDF5 snapshots as fallback) and plots
three diagnostics against nu t / b₀²:
  - rotation angle θ  [deg]
  - normalised core area  a² / b₀²
  - normalised separation  b / b₀

Core detection uses peak |ω_z| with a bimodality (valley) check to
distinguish two separate vortices from single merged core.

Reference data from Cerretelli & Williamson (2003) is overlaid when available.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    REF_DIR,
    SCHEMES,
    add_physics_args,
    build_arg_parser,
    build_style_map,
    load_theme,
    pvd_time_map,
    resolve_runtime_physics,
)

THETA_REF = REF_DIR / "theta_vs_tau.csv"
A2_REF = REF_DIR / "a2_over_b02.csv"
B_REF = REF_DIR / "b_over_b0_tau.csv"


def h5_files(solution_dir: Path, prefix: str, scheme: str) -> list[Path]:
    folder = solution_dir / f"{prefix}_{scheme}"
    return sorted(
        folder.glob(f"vpm_{prefix}_{scheme}_*.h5"),
        key=lambda p: int(re.search(r"_(\d+)\.h5$", p.name).group(1)),
    )


def h5_time(path: Path) -> float:
    with h5py.File(path, "r") as f:
        return float(f["solver"].attrs["flow_time"])


def read_h5(path: Path):
    with h5py.File(path, "r") as f:
        t = float(f["solver"].attrs["flow_time"])
        n = int(f["solver"].attrs["number_of_particles"])
        if n == 0:
            return t, None, None
        pos = f["particles"]["position"][:n].astype(np.float64)
        circ = f["particles"]["circulation"][:n].astype(np.float64)

        z = pos[:, 2]
        z_lo, z_hi = z.min(), z.max()
        z_rng = z_hi - z_lo
        mask = (z >= z_lo + z_rng / 3.0) & (z <= z_hi - z_rng / 3.0)
        pos = pos[mask]
        circ = circ[mask]

    return t, pos, circ[:, 2]


def vts_files(solution_dir: Path, prefix: str, scheme: str) -> list:
    folder = solution_dir / f"{prefix}_{scheme}" / "samples"
    if not folder.exists():
        return []
    result = []
    for p in folder.glob(f"{prefix}_{scheme}_z0_*.vts"):
        m = re.search(r"_(\d+)\.vts$", p.name)
        if m:
            result.append((int(m.group(1)), p))
    return sorted(result, key=lambda x: x[0])


def read_vts(path: Path):
    import pyvista as pv

    grid = pv.read(str(path))
    xy = grid.points[:, :2].astype(np.float64)
    omega_z = grid.point_data["Vorticity"][:, 2].astype(np.float64)
    return xy, omega_z


def _kmeans2(pts, w, init, max_iter=50, tol=1e-6):
    centers = init.copy().astype(np.float64)
    labels = np.empty(len(pts), dtype=np.int32)
    for _ in range(max_iter):
        d0 = np.sum((pts - centers[0]) ** 2, axis=1)
        d1 = np.sum((pts - centers[1]) ** 2, axis=1)
        labels[:] = (d1 < d0).astype(np.int32)
        new = np.empty_like(centers)
        conv = True
        for k in range(2):
            wk = w[labels == k]
            s = wk.sum()
            new[k] = (wk[:, None] * pts[labels == k]).sum(axis=0) / s if s > 1e-30 else centers[k]
            if np.linalg.norm(new[k] - centers[k]) > tol:
                conv = False
        centers = new
        if conv:
            break
    return labels, centers


def _neighborhood_centroid(xy, w, seed, radius, max_iter=20, tol=1e-8):
    c = seed.copy().astype(np.float64)
    for _ in range(max_iter):
        dist = np.linalg.norm(xy - c, axis=1)
        mask = dist < radius
        if not mask.any():
            break
        wm = w[mask]
        s = wm.sum()
        if s < 1e-30:
            break
        c_new = (wm[:, None] * xy[mask]).sum(axis=0) / s
        if np.linalg.norm(c_new - c) < tol:
            c = c_new
            break
        c = c_new
    return c


def step_diagnostics(xy, gz, b0, prev_c, prev_c1, use_peaks=False):
    if xy is None or gz is None:
        return np.nan, np.nan, np.nan, 0.0, None, None
    xy = xy.astype(np.float64)
    w = np.abs(gz).astype(np.float64)
    if w.sum() < 1e-30:
        return np.nan, np.nan, np.nan, 0.0, None, None

    R = 0.45 * b0

    # Always use centroid-based logic for smoother tracking,
    # but use peaks for initial guesses if not provided.
    if prev_c is not None:
        init = prev_c.copy()
    else:
        # Robust peak finding
        i1 = np.argmax(w)
        p1 = xy[i1]

        # Mask neighborhood to find second peak
        d1 = np.linalg.norm(xy - p1, axis=1)
        w2 = w.copy()
        w2[d1 < 0.25 * b0] = 0.0
        if w2.max() < 0.1 * w[i1]:  # Likely merged
            init = np.array([p1, p1])
        else:
            i2 = np.argmax(w2)
            init = np.array([p1, xy[i2]])

    _, kmeans_centers = _kmeans2(xy, w, init)
    c0 = _neighborhood_centroid(xy, w, kmeans_centers[0], R)
    c1 = _neighborhood_centroid(xy, w, kmeans_centers[1], R)

    # Bimodality/Valley check between centroids
    mid_test = 0.5 * (c0 + c1)
    d_mid = np.linalg.norm(xy - mid_test, axis=1)
    w_mid = w[np.argmin(d_mid)]

    # Values at centroids
    d_c0 = np.linalg.norm(xy - c0, axis=1)
    d_c1 = np.linalg.norm(xy - c1, axis=1)
    w_c0 = w[np.argmin(d_c0)]
    w_c1 = w[np.argmin(d_c1)]

    sep_now = np.linalg.norm(c0 - c1)
    if sep_now < 0.15 * b0 or w_mid > 0.96 * min(w_c0, w_c1):
        cores = np.array([mid_test, mid_test])
    else:
        cores = np.array([c0, c1])

    # Consistency with previous frame or symmetry
    ref = prev_c1 if prev_c1 is not None else np.array([0.0, 0.5 * b0])
    if np.linalg.norm(cores[1] - ref) < np.linalg.norm(cores[0] - ref):
        cores = cores[::-1]

    sep = float(np.linalg.norm(cores[0] - cores[1]))
    mid = 0.5 * (cores[0] + cores[1])
    ang = float(np.arctan2(cores[0, 1] - mid[1], cores[0, 0] - mid[0]))

    # --- Core area σ² via system-moment decomposition (kernel-consistent) ---
    core_mask = w > 0.05 * w.max()
    if core_mask.sum() >= 5:
        pts_c, w_c = xy[core_mask], w[core_mask]
        wt_c = w_c.sum()
        c_sys = np.array([(w_c * pts_c[:, 0]).sum() / wt_c, (w_c * pts_c[:, 1]).sum() / wt_c])
        r2_sys = (w_c * ((pts_c - c_sys) ** 2).sum(axis=1)).sum() / wt_c
        a2 = 0.5 * max(r2_sys - (sep / 2.0) ** 2, 0.0)
    else:
        a2 = np.nan

    return sep, a2, ang, float(w.sum()), cores.copy(), cores[0].copy()


def extract_merging_timeseries(
    solution_dir: Path, scheme: str, nu: float, b0: float
) -> dict | None:
    rows = []
    prev_c = None
    prev_c1 = None

    vts_list = vts_files(solution_dir, "merging", scheme)
    if vts_list:
        h5_map = {
            int(p.stem.rsplit("_", 1)[-1]): p for p in h5_files(solution_dir, "merging", scheme)
        }
        time_map = pvd_time_map(solution_dir, "merging", scheme)
        initial_circ = None
        for step, vts_path in vts_list:
            h5_path = h5_map.get(step)
            if h5_path is not None:
                t = h5_time(h5_path)
            elif step in time_map:
                t = time_map[step]
            else:
                continue
            xy, omega_z = read_vts(vts_path)

            x_uniq = np.unique(np.round(xy[:, 0], 6))
            dA = (x_uniq[1] - x_uniq[0]) ** 2 if len(x_uniq) > 1 else 0.0025
            circ = omega_z.sum() * dA
            if initial_circ is None:
                initial_circ = max(circ, 1e-10)
            elif abs(circ - initial_circ) / initial_circ > 0.50:
                print(
                    f"  [{scheme}] circulation changed by "
                    f"{(circ - initial_circ) / initial_circ:+.0%} "
                    f"at step {step} — truncating."
                )
                break

            sep, a2, ang, gam, prev_c, prev_c1 = step_diagnostics(
                xy, omega_z, b0, prev_c, prev_c1, use_peaks=True
            )
            rows.append((t, sep, a2, ang, gam))
    else:
        files = h5_files(solution_dir, "merging", scheme)
        if not files:
            return None
        for p in files:
            t, pos, gz = read_h5(p)
            xy = pos[:, :2] if pos is not None else None
            sep, a2, ang, gam, prev_c, prev_c1 = step_diagnostics(
                xy, gz, b0, prev_c, prev_c1, use_peaks=False
            )
            rows.append((t, sep, a2, ang, gam))

    if not rows:
        return None
    d = np.array(rows, dtype=float)
    tau = nu * d[:, 0] / (b0**2)
    raw_ang = d[:, 3]
    finite = np.isfinite(raw_ang)
    if finite.sum() > 1:
        unwrapped = np.full_like(raw_ang, np.nan)
        unwrapped[finite] = np.unwrap(raw_ang[finite])
        th = np.degrees(unwrapped - unwrapped[finite][0])
    else:
        th = np.full_like(raw_ang, np.nan)
    b_over_b0 = d[:, 1] / b0
    b_over_b0[d[:, 1] == 0.0] = np.nan
    return {
        "tau": tau,
        "theta_deg": th,
        "a2_over_b02": d[:, 2] / b0**2,
        "b_over_b0": b_over_b0,
        "total_gamma": d[:, 4],
    }


def plot_merging_case(args) -> int:
    solution_dir = Path(args.solution_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"merging_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    colors, theme = load_theme()

    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(solution_dir, args.gamma, args.nu, args.b0, args.a0_over_b0)
    run_nu = runtime["nu"]
    tau_max = 0.05 #run_nu * args.total_time / (args.b0**2)

    fig, axes = plt.subplots(1, 3, figsize=(12.8 / 2.54, 7.68 / 2.54))
    fig.subplots_adjust(wspace=0.6, top=0.93, bottom=0.37, left=0.09, right=0.91)

    for scheme in SCHEMES:
        ts = extract_merging_timeseries(solution_dir, scheme, run_nu, args.b0)
        if ts is None:
            print(f"  [merging] skipping {scheme!r} — no data")
            continue
        st = style_map[scheme]
        plot_kw = {
            "color": st["color"],
            "marker": st["marker"],
            "markersize": 2.2,
            "linewidth": 1.0,
            "linestyle": "None",
            "label": st["label"],
        }
        theta_mask = np.isfinite(ts["theta_deg"])
        a2_mask = np.isfinite(ts["a2_over_b02"])
        sep_mask = np.isfinite(ts["b_over_b0"])
        axes[0].plot(ts["tau"][theta_mask], ts["theta_deg"][theta_mask], **plot_kw)
        axes[1].plot(ts["tau"][a2_mask], ts["a2_over_b02"][a2_mask], **plot_kw)
        axes[2].plot(ts["tau"][sep_mask], ts["b_over_b0"][sep_mask], **plot_kw)

    ref_kw = dict(
        color="black",
        linestyle="-",
        linewidth=1.0,
        zorder=0,
        label=r"Cerretelli \& Williamson (2003)",
    )
    if THETA_REF.exists():
        r = np.loadtxt(THETA_REF, delimiter=",")
        axes[0].plot(r[:, 0], r[:, 1], **ref_kw)
    if A2_REF.exists():
        r = np.loadtxt(A2_REF, delimiter=",")
        axes[1].plot(r[:, 0], r[:, 1], **ref_kw)
    if B_REF.exists():
        r = np.loadtxt(B_REF, delimiter=",")
        axes[2].plot(r[:, 0], r[:, 1], **ref_kw)

    # ── Reference curves ─────────────────────────────────────────────────────
    # Theta: finite-core-corrected rotation rate (Moore-Saffman, initial a0/b0).
    # dθ/dt = (Γ/π b²) * [1 + 2*(a/b)² * (1 - 2*ln(a/b))]
    # In normalised form: dθ/dτ = (Re/π) * correction  [rad/τ]
    run_b0 = args.b0
    run_a0_over_b0 = args.a0_over_b0
    # Use ML convention: a0_ML = r_c0/sqrt(2), sigma²=2*nu*t
    a0_sigma = run_a0_over_b0 / np.sqrt(2.0)  # a0_ML/b0

    tau_fc = np.linspace(0.0, tau_max, 300)
    eps2 = a0_sigma**2 + 2.0 * tau_fc  # sigma^2/b0^2  (grows as 2*tau)
    eps = np.sqrt(eps2)
    corr = 1.0 + 2.0 * eps2 * (1.0 - 2.0 * np.log(np.maximum(eps, 1e-12)))
    dtheta_dtau_rad = (runtime["nu"] / (np.pi * 1.0)) * corr  # placeholder
    # Full expression: dθ/dτ [rad/τ] = Re/π * correction
    Re_run = args.gamma / runtime["nu"]
    dtheta_dtau_deg = Re_run / np.pi * corr * (180.0 / np.pi)
    theta_fc = np.cumsum(dtheta_dtau_deg * np.gradient(tau_fc))
    theta_fc -= theta_fc[0]

    axes[0].set_xlabel(r"$\nu t / b_0^2$")
    axes[0].set_ylabel(r"$\theta$ [deg]")
    axes[0].set_title(r"Rotation angle, $\theta$")
    axes[0].set_ylim([-10, 520])
    axes[0].set_xlim([0, tau_max])

    axes[1].set_xlabel(r"$\nu t / b_0^2$")
    axes[1].set_ylabel(r"$\sigma^2 / b_0^2$")
    axes[1].set_title(r"Core radius, $\sigma^2 / b_0^2$")
    axes[1].set_ylim([0, 0.35])
    axes[1].set_xlim([0, tau_max])

    axes[2].set_xlabel(r"$\nu t / b_0^2$")
    axes[2].set_ylabel(r"$b / b_0$")
    axes[2].set_title(r"Separation, $b / b_0$")
    axes[2].set_ylim([0, 1.5])
    axes[2].set_xlim([0, tau_max])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=10)
    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    p = build_arg_parser("Co-rotating vortex merger diagnostics comparison.")
    add_physics_args(p)
    return plot_merging_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
