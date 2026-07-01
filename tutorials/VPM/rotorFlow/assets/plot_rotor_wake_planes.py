#!/usr/bin/env python3
"""
Rotor wake validation — axial velocity vs momentum theory.
==========================================================
Reads the streamwise-normal crossflow sampler planes and overlays the axial
velocity profile along the horizontal (z = 0) line against actuator-disk
momentum theory:

    rotor disk:   u/U∞ = 1 − a
    far wake:     u/U∞ = 1 − 2a          (a = 1/3 Betz optimum)

As the planes move downstream the measured deficit should deepen from ~a toward
the fully-developed far-wake value 2a inside the wake tube.

Saves: figures/rotor_wake_planes.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None

ASSETS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASSETS_DIR))
from _common import CM, build_arg_parser, load_theme, save_figure


def _latest(samples_dir: Path, tag: str) -> Path | None:
    files = sorted(
        samples_dir.glob(f"slice_{tag}_*.vts"),
        key=lambda p: int(m.group(1)) if (m := re.search(r"_(\d+)\.vts$", p.name)) else -1,
    )
    return files[-1] if files else None


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


def _line_profile(grid, U_inf: float):
    """Return (y, u_x/U∞) along the z≈0 horizontal line of the plane."""
    pts = np.asarray(grid.points)
    vel = np.asarray(grid.point_data["Velocity"])
    z = pts[:, 2]
    mask = np.abs(z) <= (0.5 * (np.unique(np.round(z, 6))[1] - np.unique(np.round(z, 6))[0])
                         if np.unique(z).size > 1 else 1e-6)
    if mask.sum() < 3:  # fall back to nearest z-row
        z0 = z[np.argmin(np.abs(z))]
        mask = np.isclose(z, z0)
    y = pts[mask, 1]
    ux = vel[mask, 0] / U_inf
    order = np.argsort(y)
    return y[order], ux[order]


def main() -> int:
    ap = build_arg_parser("Rotor wake-plane validation vs momentum theory.")
    ap.add_argument("--rotor-radius", type=float, default=6.0)
    ap.add_argument("--u-inf", type=float, default=7.0)
    ap.add_argument("--a", type=float, default=1.0 / 3.0, help="Axial induction (Betz=1/3).")
    args = ap.parse_args()

    if pv is None:
        print("  [WARNING] pyvista unavailable — skipping rotor wake-plane plot.")
        return 1

    samples = Path(args.solution_dir) / "samples"
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)
    R, U = args.rotor_radius, args.u_inf

    planes = _discover_planes(samples, R)
    colors, _ = load_theme()
    plane_colors = [
        colors.get("vpm", "#1b9e77"),
        colors.get("hybrid", "#d95f02"),
        colors.get("dvhr", "#7570b3"),
        colors.get("AccentGreen", "#2B7A4E"),
        colors.get("AccentRed", "#9C2F50"),
    ]
    fig, ax = plt.subplots(figsize=(12.8 * CM, 7.4 * CM))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.34, top=0.90)
    found = False
    ux_min = np.inf
    ux_max = -np.inf
    for (tag, label), color in zip(planes, plane_colors):
        f = _latest(samples, tag)
        if f is None:
            continue
        try:
            y, ux = _line_profile(pv.read(f), U)
        except Exception as exc:
            print(f"  [WARNING] could not read {f.name}: {exc}")
            continue
        ax.plot(y / R, ux, color=color, lw=1.0, label=rf"VLM-VPM, {label}")
        ux_min = min(ux_min, float(np.nanmin(ux)))
        ux_max = max(ux_max, float(np.nanmax(ux)))
        found = True

    if not found:
        print("  [WARNING] no rotor wake-plane samples found — run rotorFlow first.")
        return 0

    # Momentum-theory references
    ref_color = colors.get("RefGray", "#6E8898")
    theory_color = colors.get("DarkText", "#2E3D46")
    ax.axhline(1.0, color=ref_color, ls=":", lw=0.8, label=r"$U_\infty$")
    ax.axhline(1.0 - args.a, color=theory_color, ls="--", lw=0.8, label=r"disk $1-a$")
    ax.axhline(
        1.0 - 2.0 * args.a,
        color=theory_color,
        ls="-.",
        lw=0.8,
        label=r"far wake $1-2a$",
    )
    ax.axvspan(-1.0, 1.0, color=ref_color, alpha=0.10, linewidth=0.0)
    ax.set_xlabel(r"$y/R$")
    ax.set_ylabel(r"$u_x/U_\infty$")
    ax.set_xlim([-1.25, 1.25])
    ax.set_ylim([min(0.25, ux_min - 0.05), max(1.1, ux_max + 0.05)])
    ax.set_title(r"Wake-plane axial velocity")
    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.46),
        columnspacing=0.9,
        handlelength=2.0,
    )

    out = figs / f"rotor_wake_planes.{args.format}"
    save_figure(fig, out, args.dpi, args.format)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
