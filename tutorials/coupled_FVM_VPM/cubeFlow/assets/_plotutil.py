"""Load native FVM–VPM cube-flow results for plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION = CASE_DIR / "solution"
FIGURES = CASE_DIR / "figures"

# Palette shared across the three plots (color-blind-safe, light background).
COLORS = {
    "fvm": "#0072B2",
    "vpm": "#D55E00",
    "cd": "#009E73",
    "cl": "#CC79A7",
    "accent": "#333333",
    "box": "#B0B0B0",
}


def metadata() -> dict:
    """Return ``run_metadata.json`` as a dict (empty if the run has not written it)."""
    path = SOLUTION / "run_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_constants() -> dict:
    """Return plot scales, falling back to the tutorial defaults."""
    meta = metadata()
    phys = meta.get("physics", {})
    u_inf = np.asarray(phys.get("u_inf", [1.0, 0.0, 0.0]), dtype=float)
    box = meta.get("fvm_solver", {}).get("fvm_domain", {})
    return {
        "U_inf": float(np.linalg.norm(u_inf)) or 1.0,
        "u_inf_vec": u_inf,
        "D": 1.0,  # cube side length (CUBE_SIDE in cube_setup.py)
        "nu": float(phys.get("nu", 1e-3)),
        "box": box
        or {
            "xmin": -1.5,
            "xmax": 1.5,
            "ymin": -1.5,
            "ymax": 1.5,
            "zmin": -1.5,
            "zmax": 1.5,
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="png", help="figure format (png, pdf, svg)")
    ap.add_argument("--dpi", type=int, default=160)
    return ap.parse_args()


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def save(fig, name: str, fmt: str, dpi: int) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  wrote {out.relative_to(CASE_DIR)}")
    return out


def load_forces() -> dict:
    """Load ``ibm_forces_history.csv`` for the cube body.

    The IBM force log is appended to, so a file that survived several runs may
    hold repeats; keep only the last monotonic-in-step block (the latest run).
    Returns a dict of 1-D arrays, empty if the file is missing/empty.
    """
    path = SOLUTION / "ibm_forces_history.csv"
    if not path.exists():
        return {}
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = np.atleast_1d(rows)
    if rows.size == 0:
        return {}
    steps = rows["step"].astype(int)
    # Last run = the final maximal run of strictly increasing step numbers.
    start = 0
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:
            start = i
    rows = rows[start:]
    return {name: rows[name] for name in rows.dtype.names}


def load_flow_integrals() -> dict:
    """Load ``samples/flow_integrals.csv`` (VPM global diagnostics per step)."""
    path = SOLUTION / "samples" / "flow_integrals.csv"
    if not path.exists():
        return {}
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if rows.size == 0:
        return {}
    return {name: rows[name] for name in rows.dtype.names}


def latest_vtu() -> Path | None:
    files = sorted(SOLUTION.glob("coupled_*_*.vtu"))
    return files[-1] if files else None


def latest_vpm_h5() -> Path | None:
    files = sorted(SOLUTION.glob("vpm_vpm_solution_*.h5"))
    return files[-1] if files else None


def load_vpm_particles(h5_path: Path) -> dict:
    """Read particle position/vorticity/circulation from a VPM backup."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        p = f["particles"]
        data = {
            "position": p["position"][:],
            "vorticity": p["vorticity"][:],
            "circulation": p["circulation"][:],
            "time": float(f["solver"].attrs["flow_time"]),
            "n": int(f["solver"].attrs["number_of_particles"]),
        }
    return data
