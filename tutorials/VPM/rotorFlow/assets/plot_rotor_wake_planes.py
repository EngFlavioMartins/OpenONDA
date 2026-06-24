#!/usr/bin/env python3
"""
Rotor wake validation at 3/6/9 D — axial velocity vs momentum theory.
=====================================================================
Reads the streamwise-normal crossflow sampler planes written at 3 D, 6 D and 9 D
downstream (x = 36, 72, 108 m for R = 6 m) and overlays the axial-velocity
profile along the horizontal (z = 0) line against actuator-disk momentum theory:

    rotor disk:   u/U∞ = 1 − a
    far wake:     u/U∞ = 1 − 2a          (a = 1/3 Betz optimum)

As the planes move downstream the measured deficit should deepen from ~a toward
the fully-developed far-wake value 2a inside the wake tube (|y| ≲ R√2).

Saves: figures/rotor_wake_planes.png
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None

CASE_DIR = Path(__file__).resolve().parents[1]
THEME_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"
FONT_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "DejaVuSerif.ttf"


def _load_theme():
    import matplotlib.font_manager as fm

    theme = None
    if THEME_PATH.exists():
        spec = importlib.util.spec_from_file_location("mpl_setup", THEME_PATH)
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        try:
            theme.set_style()
        except Exception:
            pass

    if FONT_PATH.exists():
        fm.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "DejaVu Serif"

    if theme is not None and hasattr(theme, "COLORS"):
        return dict(theme.COLORS)
    return {}


def _latest(samples_dir: Path, tag: str) -> Path | None:
    files = sorted(
        samples_dir.glob(f"slice_{tag}_*.vts"),
        key=lambda p: int(m.group(1)) if (m := re.search(r"_(\d+)\.vts$", p.name)) else -1,
    )
    return files[-1] if files else None


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
    ap = argparse.ArgumentParser(description="Rotor wake-plane validation vs momentum theory.")
    ap.add_argument("--solution-dir", default=str(CASE_DIR / "solution" / "rotor"))
    ap.add_argument("--figures-dir", default=str(CASE_DIR / "figures"))
    ap.add_argument("--rotor-radius", type=float, default=6.0)
    ap.add_argument("--u-inf", type=float, default=7.0)
    ap.add_argument("--a", type=float, default=1.0 / 3.0, help="Axial induction (Betz=1/3).")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if pv is None:
        print("  [WARNING] pyvista unavailable — skipping rotor wake-plane plot.")
        return 1

    samples = Path(args.solution_dir) / "samples"
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)
    R, U = args.rotor_radius, args.u_inf

    planes = [("x36m", "3 D"), ("x72m", "6 D"), ("x108m", "9 D")]
    colors = _load_theme()
    plane_colors = [
        colors.get("vpm", "#1b9e77"),
        colors.get("hybrid", "#d95f02"),
        colors.get("dvhr", "#7570b3"),
    ]
    fig, ax = plt.subplots(figsize=(12.8 / 2.54, 7.0 / 2.54))
    found = False
    for (tag, label), color in zip(planes, plane_colors):
        f = _latest(samples, tag)
        if f is None:
            continue
        try:
            y, ux = _line_profile(pv.read(f), U)
        except Exception as exc:
            print(f"  [WARNING] could not read {f.name}: {exc}")
            continue
        ax.plot(y / R, ux, "-", color=color, lw=1.4, label=f"VPM, {label} downstream")
        found = True

    if not found:
        print("  [WARNING] no slice_x{36,72,108}m planes found — run the extended rotor case first.")
        return 0

    # Momentum-theory references
    ax.axhline(1.0, color="grey", ls=":", lw=0.8, label=r"$U_\infty$")
    ax.axhline(1.0 - args.a, color="k", ls="--", lw=0.8, label=r"disk $1-a$")
    ax.axhline(1.0 - 2.0 * args.a, color="k", ls="-.", lw=0.8, label=r"far wake $1-2a$")
    ax.axvspan(-1.0, 1.0, color="grey", alpha=0.08)  # rotor radius band
    ax.set_xlabel(r"$y/R$")
    ax.set_ylabel(r"Axial velocity $u_x/U_\infty$")
    ax.set_title("Rotor wake axial velocity vs momentum theory")
    ax.legend(ncol=2)
    fig.tight_layout()
    out = figs / "rotor_wake_planes.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
