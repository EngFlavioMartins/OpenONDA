"""Load native FVM–VPM cube-flow results for plotting."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION = CASE_DIR / "solution"
FIGURES = CASE_DIR / "figures"

# Use the same publication theme as coupled_OFW_VPM/cubeFlow.  Keeping one
# source of truth matters here: the figures are intended to be visually
# interchangeable across the native and OpenFOAM backends.
_THEME_PATH = Path(__file__).resolve().parents[4] / "docs" / "themes" / "matplotlib_setup.py"
_THEME_SPEC = importlib.util.spec_from_file_location("openonda_matplotlib_setup", _THEME_PATH)
if _THEME_SPEC is None or _THEME_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load plotting theme from {_THEME_PATH}")
_THEME = importlib.util.module_from_spec(_THEME_SPEC)
_THEME_SPEC.loader.exec_module(_THEME)
_THEME.set_style()

COLORS = dict(_THEME.COLORS)
COLORS.update(
    {
        # Native names retained as aliases for the existing plotting helpers.
        "fvm": COLORS["hybrid"],
        "vpm": COLORS["vpm"],
        "cd": COLORS["hybrid"],
        "cl": COLORS["vpm"],
        "accent": COLORS["DarkText"],
        "box": COLORS["background_strong"],
    }
)
COLORMAPS = dict(_THEME.COLORMAPS)


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
        "D": 1.0,  # cube side length (CUBE_SIDE in cubeFlow_setup.py)
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
    ap.add_argument("--times", default="all")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def apply_style() -> None:
    _THEME.set_style()


def save(fig, name: str, fmt: str, dpi: int) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.{fmt}"
    _THEME.save_fig(fig, out, figure_format=fmt, dpi=dpi)
    print(f"  wrote {out.relative_to(CASE_DIR)}")
    return out


def load_forces() -> dict:
    """Load the cube force history.

    Prefers the wall-patch integration ``forces_history.csv`` (body-fitted
    cube — the current tutorial), falling back to ``ibm_forces_history.csv``
    from older immersed-boundary runs.  Both logs are appended to, so keep
    only the last monotonic-in-step block (the latest run).  Returns a dict
    of 1-D arrays with at least ``time``, ``Cd``, ``Cl``; wall data also has
    the pressure/viscous split (``Fpx``, ``Fvx``), IBM data has ``slip``.
    """
    for fname in ("forces_history.csv", "ibm_forces_history.csv"):
        path = SOLUTION / fname
        if not path.exists():
            continue
        rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        rows = np.atleast_1d(rows)
        if rows.size == 0:
            continue
        steps = rows["step"].astype(int)
        # Last run = the final maximal run of strictly increasing step numbers.
        start = 0
        for i in range(1, len(steps)):
            if steps[i] <= steps[i - 1]:
                start = i
        rows = rows[start:]
        return {name: rows[name] for name in rows.dtype.names}
    return {}


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
    # Parallel FVM output is a PVTU collection plus one VTU piece per rank.
    # Always sample the collection; selecting the lexicographically last piece
    # would silently plot only one partition of the domain.
    collections = sorted(SOLUTION.glob("coupled_*_*.pvtu"))
    if collections:
        return collections[-1]
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
            "radius": p["radius"][:],
            "time": float(f["solver"].attrs["flow_time"]),
            "n": int(f["solver"].attrs["number_of_particles"]),
        }
    return data
