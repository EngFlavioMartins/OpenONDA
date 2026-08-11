#!/usr/bin/env python3
"""z = 0 symmetry-plane field comparison for the single Lamb-Oseen vortex.

Each of the four viscous schemes (GBD, CS, RWM, DVH-R) contributes **one
quadrant** of the plane.  The quadrants tile into a seamless image of
the full field, making it immediately clear which scheme over- or
under-diffuses relative to the others.

Quadrant layout
---------------
  x <= 0, y >= 0  |  x >= 0, y >= 0
  ----------------------------------
        GBD       |      CS
  ----------------------------------
        RWM      |     DVH-R
  ----------------------------------
  x <= 0, y <= 0  |  x >= 0, y <= 0

A single, shared colour bar per panel (velocity magnitude and z-vorticity)
enables direct quantitative comparison across schemes.

Saves: figures/vortex_surface_fields.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    SCHEMES,
    build_arg_parser,
    load_theme,
    publication_size,
    pvd_time_map,
    resolve_runtime_physics,
    save_publication_figure,
)

_LAYOUT = [
    ("gbd", "TL", r"$\mathrm{GBD}$", (-4.5, 4.5), "left", "top"),
    ("cs", "TR", r"$\mathrm{CS}$", (4.5, 4.5), "right", "top"),
    ("rwm", "BL", r"$\mathrm{RWM}$", (-4.5, -4.5), "left", "bottom"),
    ("dvh", "BR", r"$\mathrm{DVH}$", (4.5, -4.5), "right", "bottom"),
]


# =============================================================
# Grid helpers
# =============================================================


def _quad(arr: np.ndarray, qid: str, mid: int) -> np.ndarray:
    if qid == "TL":
        return arr[mid:, : mid + 1]
    if qid == "TR":
        return arr[mid:, mid:]
    if qid == "BL":
        return arr[: mid + 1, : mid + 1]
    if qid == "BR":
        return arr[: mid + 1, mid:]
    raise ValueError(f"Unknown quadrant id: {qid!r}")


def _read_vts(path: Path):
    import vtk
    import vtk.util.numpy_support as ns

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


# =============================================================
# Data loading
# =============================================================


def _find_surface_vts(
    samples_dir: Path,
    scheme: str,
    step: int | None,
    target_time: float,
    tolerance: float,
) -> tuple[Path | None, float | None]:
    folder = samples_dir / f"vortex_{scheme}"
    time_by_step = pvd_time_map(samples_dir, "vortex", scheme)
    if not time_by_step:
        return None, None

    if step is not None:
        selected_step = step
        selected_time = time_by_step.get(step)
        if selected_time is None:
            return None, None
    else:
        selected_step = min(time_by_step, key=lambda item: abs(time_by_step[item] - target_time))
        selected_time = time_by_step[selected_step]
        if abs(selected_time - target_time) > tolerance:
            print(
                f"  [skip] {scheme.upper()} final surface is t={selected_time:.3g}s, "
                f"not near requested t={target_time:.3g}s."
            )
            return None, None

    path = folder / f"vortex_{scheme}_z0_{selected_step:06d}.vts"
    return (path, selected_time) if path.is_file() else (None, None)


# =============================================================
# Plot
# =============================================================


def plot_surface_fields(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"vortex_surface_fields.{fmt}"

    colors, theme = load_theme()
    runtime = resolve_runtime_physics(samples_dir, args.gamma, args.nu, args.b0, args.a0_over_b0)
    run_nu = runtime["nu"]
    ac0 = runtime["ac0"]
    t0 = runtime["t0"]
    uc_ref = args.gamma / (2.0 * np.pi * ac0)
    wc_ref = args.gamma / (np.pi * ac0**2)

    # -- Load each scheme's surface data ----------------------------------
    datasets: dict[str, dict] = {}
    tolerance = max(0.5, 20.0 * args.dt)
    for scheme, qid, *_ in _LAYOUT:
        vts, sample_time = _find_surface_vts(
            samples_dir,
            scheme,
            args.step,
            args.total_time,
            tolerance,
        )
        if vts is None:
            print(f"  [surface] no requested VTS for {scheme!r} — skipping quadrant")
            continue
        try:
            X, Y, vm, wz, mid = _read_vts(vts)
        except Exception as exc:
            print(f"  [surface] read error {vts.name}: {exc}")
            continue
        step = int(vts.stem.split("_")[-1])
        datasets[scheme] = dict(
            X=X,
            Y=Y,
            vel_mag=vm,
            vort_z=wz,
            step=step,
            time=sample_time,
            mid=mid,
        )

    if len(datasets) != len(SCHEMES):
        out.unlink(missing_ok=True)
        print(
            f"  [surface] complete fields available for {len(datasets)}/{len(SCHEMES)} "
            "methods; figure not generated"
        )
        return 1

    # -- Shared normalisation limits ------------------------------------
    v_max = max(d["vel_mag"].max() for d in datasets.values()) / uc_ref
    w_max = max(d["vort_z"].max() for d in datasets.values()) / wc_ref
    v_norm = mcolors.Normalize(vmin=0.0, vmax=v_max)
    w_norm = mcolors.Normalize(vmin=0.0, vmax=w_max)
    v_cmap = theme.COLORMAPS["vortex_speed"]
    w_cmap = theme.COLORMAPS["vortex_vorticity"]

    x_ext = max(abs(d["X"]).max() for d in datasets.values()) / ac0
    y_ext = max(abs(d["Y"]).max() for d in datasets.values()) / ac0
    ext = max(x_ext, y_ext)
    ax_lim = ext

    # -- Figure --------------------------------------------------------
    fig, (ax_v, ax_w) = plt.subplots(
        1,
        2,
        figsize=publication_size(6.2),
        constrained_layout=True,
    )

    for scheme, qid, label, (_tx_frac, _ty_frac), ha, va in _LAYOUT:
        if scheme not in datasets:
            continue
        d = datasets[scheme]
        mid = d["mid"]
        Xn = d["X"] / ac0
        Yn = d["Y"] / ac0
        Xs = _quad(Xn, qid, mid)
        Ys = _quad(Yn, qid, mid)
        vms = _quad(d["vel_mag"] / uc_ref, qid, mid)
        wzs = _quad(d["vort_z"] / wc_ref, qid, mid)

        pcm_kw = dict(shading="gouraud", rasterized=True)
        ax_v.pcolormesh(Xs, Ys, vms, cmap=v_cmap, norm=v_norm, **pcm_kw)
        ax_w.pcolormesh(Xs, Ys, wzs, cmap=w_cmap, norm=w_norm, **pcm_kw)

        tx = -0.85 * ax_lim if ha == "left" else 0.85 * ax_lim
        ty = 0.85 * ax_lim if va == "top" else -0.85 * ax_lim
        txt_kw = dict(
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.15", fc=colors["LightText"], alpha=0.65, lw=0),
        )
        ax_v.text(tx, ty, label, **txt_kw)
        ax_w.text(tx, ty, label, **txt_kw)

    divider_kw = dict(color=colors["LightText"], linewidth=1.0, alpha=0.9)
    for ax in (ax_v, ax_w):
        ax.axhline(0, **divider_kw)
        ax.axvline(0, **divider_kw)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_xlabel(r"$x\,/\,a_{c,0}$")
        ax.set_ylabel(r"$y\,/\,a_{c,0}$")

    ax_v.set_title(r"Velocity magnitude, $|\mathbf{u}|\,/\,U_{c,0}$")
    ax_w.set_title(r"Vorticity, $\omega_z\,/\,\omega_{c,0}$")

    sm_v = ScalarMappable(cmap=v_cmap, norm=v_norm)
    sm_v.set_array([])
    sm_w = ScalarMappable(cmap=w_cmap, norm=w_norm)
    sm_w.set_array([])
    cb_v = fig.colorbar(sm_v, ax=ax_v, fraction=0.05, pad=0.04)
    cb_w = fig.colorbar(sm_w, ax=ax_w, fraction=0.05, pad=0.04)
    cb_v.set_label(r"$|\mathbf{u}|\,/\,U_{c,0}$", loc="top")
    cb_w.set_label(r"$\omega_z\,/\,\omega_{c,0}$", loc="top")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, out, args.dpi)
    return 0


def parse_args() -> argparse.Namespace:
    p = build_arg_parser("z=0 surface field tiled comparison.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return plot_surface_fields(args)


if __name__ == "__main__":
    raise SystemExit(main())
