#!/usr/bin/env python3
"""z = 0 symmetry-plane field comparison for the single Lamb-Oseen vortex.

Each of the four viscous schemes (GBD, CS, RWM, DVH-R) contributes **one
quadrant** of the plane.  The quadrants tile into a seamless image of
the full field, making it immediately clear which scheme over- or
under-diffuses relative to the others.

Quadrant layout
---------------
  x ≤ 0, y ≥ 0  │  x ≥ 0, y ≥ 0
  ───────────────┼───────────────
        GBD      │      CS
  ───────────────┼───────────────
        RWM     │     DVH-R
  ───────────────┼───────────────
  x ≤ 0, y ≤ 0  │  x ≥ 0, y ≤ 0

A single, shared colour bar per panel (velocity magnitude and z-vorticity)
enables direct quantitative comparison across schemes.

Saves: figures/vortex_surface_fields.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import vtk
import vtk.util.numpy_support as ns
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_theme, FIGURES_DIR, SOLUTION_DIR

# ── Physical reference constants (matching vortex_setup.py) ─────────────────
_NU = 1.0 / 530.0
_RC0 = 0.125  # m  — initial core radius
_T0 = _RC0**2 / (4.0 * _NU)  # s  ≈ 2.071 s
_UC0 = 1.0 / (2.0 * np.pi * _RC0)  # m/s ≈ 1.273
_WC0 = 1.0 / (np.pi * _RC0**2)  # /s  ≈ 20.37
_DT = 0.03  # s — default time-step

# Quadrant layout: (folder_suffix, quadrant_id, LaTeX label, text xy, ha, va)
# quadrant_id: 'TL', 'TR', 'BL', 'BR'
_LAYOUT = [
    ("gbd", "TL", r"$\mathrm{GBD}$", (-4.5, 4.5), "left", "top"),
    ("cs", "TR", r"$\mathrm{CS}$", (4.5, 4.5), "right", "top"),
    ("rwm", "BL", r"$\mathrm{RWM}$", (-4.5, -4.5), "left", "bottom"),
    ("dvh", "BR", r"$\mathrm{DVH}$", (4.5, -4.5), "right", "bottom"),
]

# Grid mid-index — derived dynamically from each VTS grid in _read_vts().


def _quad(arr: np.ndarray, qid: str, mid: int) -> np.ndarray:
    """Slice a (ny, nx) array to the requested quadrant.

    VTK structured-grid ordering: i (x) varies fastest → arr[j, i].
    """
    if qid == "TL":
        return arr[mid:, : mid + 1]  # x ≤ 0, y ≥ 0
    if qid == "TR":
        return arr[mid:, mid:]  # x ≥ 0, y ≥ 0
    if qid == "BL":
        return arr[: mid + 1, : mid + 1]  # x ≤ 0, y ≤ 0
    if qid == "BR":
        return arr[: mid + 1, mid:]  # x ≥ 0, y ≤ 0
    raise ValueError(f"Unknown quadrant id: {qid!r}")


def _read_vts(path: Path):
    """Return (X, Y, vel_mag, vort_z, mid) as (ny, nx) float64 numpy arrays.

    Uses the VTK structured-grid reader.  Expects fields 'VelocityMagnitude'
    and 'Vorticity' to be present in the point-data.
    ``mid`` is the index of the grid centre (x=0, y=0).
    """

    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()

    dims = [0, 0, 0]
    grid.GetDimensions(dims)
    nx_d, ny_d = dims[0], dims[1]
    mid = nx_d // 2
    n = grid.GetNumberOfPoints()
    pts = np.array([grid.GetPoint(i) for i in range(n)]).reshape(ny_d, nx_d, 3)
    X = pts[:, :, 0]
    Y = pts[:, :, 1]

    pd = grid.GetPointData()
    vel_mag = ns.vtk_to_numpy(pd.GetArray("VelocityMagnitude")).reshape(ny_d, nx_d)
    vort = ns.vtk_to_numpy(pd.GetArray("Vorticity")).reshape(ny_d, nx_d, 3)
    vort_z = vort[:, :, 2]

    return X, Y, np.clip(vel_mag, 0.0, None), np.clip(vort_z, 0.0, None), mid


def _find_last_vts(solution_dir: Path, scheme: str) -> tuple[Path | None, float | None]:
    """Return (path, flow_time) for the last stable z0 VTS file.

    Walks backwards from the last file to find one with non-trivial velocity
    data (max > 1e-10), so schemes that went unstable are clipped.
    flow_time is obtained from the matching H5 snapshot (same step number).
    """
    folder = solution_dir / f"vortex_{scheme}" / "samples"
    files = sorted(folder.glob(f"vortex_{scheme}_z0_*.vts"))
    if not files:
        return None, None
    import re as _re

    # Walk backwards to find last file with non-trivial data
    last = files[-1]
    for candidate in reversed(files):
        try:
            _, _, vm, _, _ = _read_vts(candidate)
            if vm.max() > 1e-10:
                last = candidate
                break
        except Exception:
            continue
    m = _re.search(r"_(\d+)\.vts$", last.name)
    if m is None:
        return last, None
    step = int(m.group(1))
    h5_path = solution_dir / f"vortex_{scheme}" / f"vpm_vortex_{scheme}_{step:06d}.h5"
    flow_time = None
    if h5_path.exists():
        import h5py

        with h5py.File(h5_path, "r") as f:
            flow_time = float(f["solver"].attrs["flow_time"])
    return last, flow_time


def plot_surface_fields(
    solution_dir: Path, figures_dir: Path, dpi: int = 300, fmt: str = "png", dt: float = _DT
) -> int:
    load_theme()  # applies rcParams (LaTeX, fonts, etc.)

    # ── Load each scheme's surface data ──────────────────────────────────────
    datasets: dict[str, dict] = {}
    for scheme, qid, *_ in _LAYOUT:
        vts, flow_time = _find_last_vts(solution_dir, scheme)
        if vts is None:
            print(f"  [surface] no VTS for {scheme!r} — skipping quadrant")
            continue
        try:
            X, Y, vm, wz, mid = _read_vts(vts)
        except Exception as exc:
            print(f"  [surface] read error {vts.name}: {exc}")
            continue
        step = int(vts.stem.split("_")[-1])
        datasets[scheme] = dict(
            X=X, Y=Y, vel_mag=vm, vort_z=wz, step=step, flow_time=flow_time, mid=mid
        )

    if not datasets:
        print("  [surface] no data found — nothing to plot.")
        return 1

    # ── Shared normalisation limits ───────────────────────────────────────────
    v_max = max(d["vel_mag"].max() for d in datasets.values()) / _UC0
    w_max = max(d["vort_z"].max() for d in datasets.values()) / _WC0
    v_norm = mcolors.Normalize(vmin=0.0, vmax=v_max)
    w_norm = mcolors.Normalize(vmin=0.0, vmax=w_max)
    v_cmap = "plasma"
    w_cmap = "inferno"

    # Derive axis extent from data (normalised by rc0)
    x_ext = max(abs(d["X"]).max() for d in datasets.values()) / _RC0
    y_ext = max(abs(d["Y"]).max() for d in datasets.values()) / _RC0
    ext = max(x_ext, y_ext)
    ax_lim = ext

    # Representative time — prefer actual flow_time from H5, fall back to step*dt
    first_data = next(iter(datasets.values()))
    ft = first_data.get("flow_time")
    if ft is not None:
        t_total = _T0 + ft
    else:
        t_total = _T0 + first_data["step"] * dt

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_v, ax_w) = plt.subplots(
        1, 2, figsize=(12.8 / 2.54, 5.97 / 2.54), constrained_layout=True
    )

    for scheme, qid, label, (_tx_frac, _ty_frac), ha, va in _LAYOUT:
        if scheme not in datasets:
            continue
        d = datasets[scheme]
        mid = d["mid"]
        Xn = d["X"] / _RC0
        Yn = d["Y"] / _RC0
        Xs = _quad(Xn, qid, mid)
        Ys = _quad(Yn, qid, mid)
        vms = _quad(d["vel_mag"] / _UC0, qid, mid)
        wzs = _quad(d["vort_z"] / _WC0, qid, mid)

        pcm_kw = dict(shading="gouraud", rasterized=True)
        ax_v.pcolormesh(Xs, Ys, vms, cmap=v_cmap, norm=v_norm, **pcm_kw)
        ax_w.pcolormesh(Xs, Ys, wzs, cmap=w_cmap, norm=w_norm, **pcm_kw)

        # Place labels at 85 % of axis extent in the appropriate corner
        tx = -0.85 * ax_lim if ha == "left" else 0.85 * ax_lim
        ty = 0.85 * ax_lim if va == "top" else -0.85 * ax_lim
        txt_kw = dict(
            fontsize=10,
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65, lw=0),
        )
        ax_v.text(tx, ty, label, **txt_kw)
        ax_w.text(tx, ty, label, **txt_kw)

    # Quadrant dividers
    divider_kw = dict(color="white", linewidth=1.0, alpha=0.9)
    for ax in (ax_v, ax_w):
        ax.axhline(0, **divider_kw)
        ax.axvline(0, **divider_kw)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_xlabel(r"$x\,/\,r_{c,0}$")
        ax.set_ylabel(r"$y\,/\,r_{c,0}$")

    ax_v.set_title(r"Velocity magnitude, $|\mathbf{u}|\,/\,U_{c,0}$")
    ax_w.set_title(r"Vorticity, $\omega_z\,/\,\omega_{c,0}$")

    # Shared colorbars (one per panel)
    sm_v = ScalarMappable(cmap=v_cmap, norm=v_norm)
    sm_v.set_array([])
    sm_w = ScalarMappable(cmap=w_cmap, norm=w_norm)
    sm_w.set_array([])
    cb_v = fig.colorbar(sm_v, ax=ax_v, fraction=0.05, pad=0.04)
    cb_w = fig.colorbar(sm_w, ax=ax_w, fraction=0.05, pad=0.04)
    cb_v.set_label(r"$|\mathbf{u}|\,/\,U_{c,0}$", fontsize=10, loc="top")
    cb_w.set_label(r"$\omega_z\,/\,\omega_{c,0}$", fontsize=10, loc="top")

    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"vortex_surface_fields.{fmt}"
    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="z=0 surface field tiled comparison.")
    p.add_argument("--solution-dir", default=str(SOLUTION_DIR))
    p.add_argument("--figures-dir", default=str(FIGURES_DIR))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--format",
        choices=["png", "svg"],
        default="png",
        help="Output figure format (default: png).",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=_DT,
        help="Time-step size [s] (used to compute physical time from step number).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return plot_surface_fields(
        solution_dir=Path(args.solution_dir),
        figures_dir=Path(args.figures_dir),
        dpi=args.dpi,
        fmt=args.format,
        dt=args.dt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
