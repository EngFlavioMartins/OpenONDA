"""Load native FVM–VPM cube-flow results for plotting."""

from __future__ import annotations

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
REFERENCE_FORCE_FILES = (
    CASE_DIR / "referenceFlow" / "solution" / "forces_history.csv",
    CASE_DIR / "postprocessed" / "reference" / "forces_history.csv",
    CASE_DIR / "postprocessed" / "reference" / "forces_history.csv.gz",
)


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


def save(fig, name: str, fmt: str, dpi: int) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.{fmt}"
    _THEME.save_fig(fig, out, figure_format=fmt, dpi=dpi)
    print(f"  wrote {out.relative_to(CASE_DIR)}")
    return out


def load_forces(path: Path | None = None) -> dict[str, np.ndarray]:
    """Load one body-fitted wall-force history."""
    source = path or SOLUTION / "forces_history.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    rows = np.atleast_1d(
        np.genfromtxt(source, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if rows.size == 0:
        raise ValueError(f"empty force history: {source}")
    if "step" in rows.dtype.names:
        resets = np.flatnonzero(np.diff(rows["step"].astype(int)) <= 0)
        rows = rows[resets[-1] + 1 :] if resets.size else rows
    return {name: np.asarray(rows[name]) for name in rows.dtype.names}


def load_reference_forces() -> dict[str, np.ndarray]:
    return load_forces(next(path for path in REFERENCE_FORCE_FILES if path.exists()))


def hybrid_pvd() -> Path:
    files = sorted(SOLUTION.glob("coupled_*.pvd"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one hybrid PVD in {SOLUTION}, found {len(files)}")
    return files[0]


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
