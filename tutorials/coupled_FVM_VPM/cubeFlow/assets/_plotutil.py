"""Load sampled FVM-VPM cube-flow results for plotting.

Every figure is built from sampler output alone — the line CSVs and slice VTS
files under ``samples/`` — so plotting needs no solver, no GPU and no raw
field dumps. Three solutions are compared:

``reference``  fully meshed FVM (referenceFlow/samples/)
``fvm``        the coupled run's FVM near field (samples/fvm_*)
``vpm``        the coupled run's VPM far field  (samples/vpm_*)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION = CASE_DIR / "solution"
SAMPLES = CASE_DIR / "samples"
REFERENCE_SAMPLES = CASE_DIR / "referenceFlow" / "samples"
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
        "fvm": COLORS["hybrid"],
        "vpm": COLORS["vpm"],
        "cd": COLORS["hybrid"],
        "cl": COLORS["vpm"],
        "accent": COLORS["DarkText"],
        "box": COLORS["background_strong"],
    }
)
COLORMAPS = dict(_THEME.COLORMAPS)

SOURCES = {
    "reference": {"dir": REFERENCE_SAMPLES, "prefix": "", "label": "Reference FVM"},
    "fvm": {"dir": SAMPLES, "prefix": "fvm_", "label": "Coupled FVM"},
    "vpm": {"dir": SAMPLES, "prefix": "vpm_", "label": "Coupled VPM"},
}


def label(source: str) -> str:
    return SOURCES[source]["label"]


def colour(source: str) -> str:
    return COLORS["reference"] if source == "reference" else COLORS[source]


def _path(source: str, name: str, suffix: str) -> Path:
    entry = SOURCES[source]
    return entry["dir"] / f"{entry['prefix']}{name}{suffix}"


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


# ---- Line samplers -------------------------------------------------------


def _read_line_csv(path: Path) -> dict[str, np.ndarray]:
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    return {name: np.asarray(rows[name]) for name in rows.dtype.names}


def line_times(source: str, name: str) -> np.ndarray:
    """Sampling times available for one line sampler."""
    path = _path(source, name, ".csv")
    if not path.exists():
        return np.empty(0)
    return np.unique(_read_line_csv(path)["flow_time"])


def load_line(source: str, name: str, time: float) -> dict[str, np.ndarray] | None:
    """Return the frame of one line sampler nearest ``time``, sorted by x."""
    path = _path(source, name, ".csv")
    if not path.exists():
        return None
    table = _read_line_csv(path)
    times = np.unique(table["flow_time"])
    if times.size == 0:
        return None
    picked = times[np.argmin(np.abs(times - time))]
    mask = table["flow_time"] == picked
    frame = {key: values[mask] for key, values in table.items()}
    order = np.argsort(frame["x"])
    frame = {key: values[order] for key, values in frame.items()}
    frame["time"] = picked
    return frame


# ---- Surface samplers ----------------------------------------------------


def slice_frames(source: str, name: str = "slice_z0") -> list[tuple[float, Path]]:
    """Return (time, path) for every slice snapshot listed in the PVD index."""
    pvd = _path(source, name, ".pvd")
    if not pvd.exists():
        return []
    matches = re.finditer(r'timestep="([^"]+)"\s+[^>]*file="([^"]+)"', pvd.read_text())
    return sorted((float(m.group(1)), pvd.parent / m.group(2)) for m in matches)


def slice_times(source: str, name: str = "slice_z0") -> np.ndarray:
    return np.array([t for t, _ in slice_frames(source, name)])


def load_slice(source: str, time: float, name: str = "slice_z0") -> dict | None:
    """Return the slice snapshot nearest ``time`` as 2-D arrays on its grid."""
    import pyvista as pv

    frames = slice_frames(source, name)
    if not frames:
        return None
    picked_time, path = min(frames, key=lambda item: abs(item[0] - time))
    grid = pv.read(path)
    ni, nj, _ = grid.dimensions
    shape = (ni, nj)

    def field(key, component=None):
        if key not in grid.point_data:
            return None
        values = np.asarray(grid.point_data[key], dtype=float)
        values = values if component is None else values[:, component]
        return values.reshape(shape, order="F")

    points = np.asarray(grid.points, dtype=float)
    return {
        "time": picked_time,
        "x": points[:, 0].reshape(shape, order="F"),
        "y": points[:, 1].reshape(shape, order="F"),
        "Ux": field("Velocity", 0),
        "Uy": field("Velocity", 1),
        "omega_z": field("Vorticity", 2),
    }


# ---- Forces --------------------------------------------------------------


def load_forces(source: str) -> dict[str, np.ndarray] | None:
    """Load one body-fitted wall-force history, trimming restarted runs."""
    path = SOURCES[source]["dir"] / "forces_history.csv"
    if not path.exists():
        return None
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if rows.size == 0:
        return None
    if "step" in rows.dtype.names:
        resets = np.flatnonzero(np.diff(rows["step"].astype(int)) <= 0)
        rows = rows[resets[-1] + 1 :] if resets.size else rows
    return {name: np.asarray(rows[name]) for name in rows.dtype.names}


def load_vpm_forces() -> dict[str, np.ndarray] | None:
    """Load the panel-body force history written by the coupled VPM solver."""
    path = SAMPLES / "vpm_forces.csv"
    if not path.exists():
        return None
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    return {name: np.asarray(rows[name]) for name in rows.dtype.names} if rows.size else None


def comparison_times() -> np.ndarray:
    """Times sampled by the coupled run, which drive the per-frame figures."""
    times = line_times("fvm", "centerline")
    return times[times > 1e-9]
