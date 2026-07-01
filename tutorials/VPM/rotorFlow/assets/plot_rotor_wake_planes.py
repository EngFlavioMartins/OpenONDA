#!/usr/bin/env python3
"""Rotor wake validation — axial velocity vs momentum theory.

Reads the streamwise-normal crossflow sampler planes and overlays
time- and azimuthally-averaged axial-velocity profiles against
actuator-disk momentum theory:

    rotor disk:   u/U∞ = 1 − a
    far wake:     u/U∞ = 1 − 2a          (a = 1/3 Betz optimum)

As the planes move downstream the annular rotor wake should show a clear
deficit relative to the freestream.

Saves: figures/rotor_wake_planes.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_arg_parser, build_rotor_style_map, load_theme


# ==============================================================================
# File helpers
# ==============================================================================


def _step_number(path: Path) -> int:
    match = re.search(r"_(\d+)\.vts$", path.name)
    return int(match.group(1)) if match else -1


def _plane_files(samples_dir: Path, tag: str) -> list[Path]:
    return sorted(samples_dir.glob(f"slice_{tag}_*.vts"), key=_step_number)


def _discover_planes(samples_dir: Path, rotor_radius: float) -> list[tuple[str, str]]:
    tags: list[tuple[float, str]] = []
    for pvd in sorted(samples_dir.glob("slice_x*m.pvd")):
        match = re.fullmatch(r"slice_x(?P<x>[-+]?\d+(?:\.\d+)?)m\.pvd", pvd.name)
        if match:
            tags.append((float(match.group("x")), f"x{match.group('x')}m"))

    if not tags:
        tags = [(1.5 * rotor_radius, f"x{int(round(1.5 * rotor_radius))}m")]
        tags += [(3.0 * rotor_radius, f"x{int(round(3.0 * rotor_radius))}m")]
        tags += [(4.5 * rotor_radius, f"x{int(round(4.5 * rotor_radius))}m")]

    tags.sort(key=lambda item: item[0])
    return [(tag, rf"${x_loc / rotor_radius:g}R$") for x_loc, tag in tags]


def _select_time_window(
    files: list[Path],
    dt: float,
    omega: float,
    averaging_rotations: float,
    tail_fraction: float,
) -> list[Path]:
    if not files:
        return []
    if averaging_rotations > 0.0 and dt > 0.0 and omega > 0.0:
        t_max = _step_number(files[-1]) * dt
        t_cut = t_max - averaging_rotations * 2.0 * np.pi / omega
        selected = [f for f in files if _step_number(f) * dt >= t_cut]
    else:
        n_tail = max(1, int(np.ceil(len(files) * tail_fraction)))
        selected = files[-n_tail:]
    return selected or [files[-1]]


def _azimuthal_profile(grid, U_inf: float, rotor_radius: float, radial_edges: np.ndarray):
    """Return one plane's azimuthally averaged u_x/U∞ in radial bins."""
    pts = np.asarray(grid.points)
    vel = np.asarray(grid.point_data["Velocity"])
    r_over_R = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2) / rotor_radius
    ux_over_U = vel[:, 0] / U_inf
    bin_idx = np.digitize(r_over_R, radial_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(radial_edges) - 1) & np.isfinite(ux_over_U)

    sums = np.zeros(len(radial_edges) - 1)
    counts = np.zeros(len(radial_edges) - 1)
    np.add.at(sums, bin_idx[valid], ux_over_U[valid])
    np.add.at(counts, bin_idx[valid], 1.0)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0.0)


def _time_azimuthal_profile(
    files: list[Path],
    U_inf: float,
    rotor_radius: float,
    radial_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    profiles = []
    for f in files:
        profiles.append(_azimuthal_profile(pv.read(f), U_inf, rotor_radius, radial_edges))
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    return radial_centers, np.nanmean(np.vstack(profiles), axis=0)


# ==============================================================================
# Plot
# ==============================================================================


def plot_wake_planes(args) -> int:
    if pv is None:
        print("  [WARNING] pyvista unavailable — skipping rotor wake-plane plot.")
        return 1

    samples = Path(args.solution_dir) / "samples"
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"rotor_wake_planes.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    R, U = args.rotor_radius, args.u_inf
    omega = args.tip_speed_ratio * U / R

    planes = _discover_planes(samples, R)
    colors, _ = load_theme()
    styles = build_rotor_style_map(colors)
    s_ref = styles["reference"]
    s_vpm = styles["vpm"]
    fig, ax = plt.subplots(figsize=(12.8 / 2.54, 7.4 / 2.54))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.14, top=0.90)

    markers = ['s', 'o', '^']
    n_planes = len(planes)
    alphas = np.linspace(0.5, 1.0, n_planes) if n_planes > 1 else [0.8]
    radial_edges = np.linspace(0.0, args.r_max, args.radial_bins + 1)

    found = False
    ux_min = np.inf
    ux_max = -np.inf
    for i, (tag, label) in enumerate(planes):
        files = _select_time_window(
            _plane_files(samples, tag),
            dt=args.dt,
            omega=omega,
            averaging_rotations=args.averaging_rotations,
            tail_fraction=args.tail_fraction,
        )
        if not files:
            continue
        try:
            r, ux = _time_azimuthal_profile(files, U, R, radial_edges)
        except Exception as exc:
            print(f"  [WARNING] could not read wake samples for {tag}: {exc}")
            continue
        ax.plot(
            r, ux,
            color=s_vpm["color"],
            alpha=alphas[i],
            marker=markers[i % len(markers)],
            markersize=s_vpm["markersize"],
            lw=s_vpm["linewidth"],
            label=rf"{label}, $\langle u_x\rangle_{{t,\theta}}$",
        )
        ux_min = min(ux_min, float(np.nanmin(ux)))
        ux_max = max(ux_max, float(np.nanmax(ux)))
        found = True

    if not found:
        print("  [WARNING] no rotor wake-plane samples found — run rotorFlow first.")
        return 0

    # Momentum-theory references (text-annotated, not in legend)
    ax.axhline(1.0, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"])
    ax.axhline(1.0 - args.a, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"])
    ax.axhline(1.0 - 2.0 * args.a, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"])

    x_text = 1.03
    ax.text(x_text, 1.03, r"$U_\infty$", color="dimgray", va="center", ha="left", fontsize=10)
    ax.text(x_text, 1.03 - args.a, r"$(1-a)U_\infty$", color="dimgray", va="center", ha="left", fontsize=10)
    ax.text(x_text, 1.03 - 2.0 * args.a, r"$(1-2a)U_\infty$",
            color="dimgray", va="center", ha="left", fontsize=10)

    ax.axvspan(args.hub_radius / R, 1.0, color=s_ref["color"], alpha=0.5, linewidth=0.0)
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\langle u_x/U_\infty\rangle_{t,\theta}$")
    ax.set_xlim([0.0, args.r_max])
    ax.set_ylim([min(0.25, ux_min - 0.05), max(1.1, ux_max + 0.05)])
    ax.set_title(r"Azimuthally averaged wake deficit")
    ax.legend(loc="best")

    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    p = build_arg_parser("Rotor wake-plane validation vs momentum theory.")
    p.add_argument("--rotor-radius", type=float, default=6.0)
    p.add_argument("--hub-radius", type=float, default=1.0)
    p.add_argument("--u-inf", type=float, default=7.0)
    p.add_argument("--tip-speed-ratio", type=float, default=7.0)
    p.add_argument("--dt", type=float, default=0.005, help="Simulation time step used in rotor_setup.py.")
    p.add_argument("--averaging-rotations", type=float, default=3.0)
    p.add_argument("--tail-fraction", type=float, default=0.25)
    p.add_argument("--radial-bins", type=int, default=32)
    p.add_argument("--r-max", type=float, default=1.25)
    p.add_argument("--a", type=float, default=1.0 / 3.0, help="Axial induction (Betz=1/3).")
    return plot_wake_planes(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
