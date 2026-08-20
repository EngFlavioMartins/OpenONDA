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

# Keep one publication theme: reference and coupled figures are intended to be
# visually interchangeable.
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

# Hybrid line and surface diagnostics use a coarser cadence than the dense
# reference history. PLOT_DT is their default comparison cadence;
# common_times() still derives exact intersections from the files themselves.
PLOT_TIME_STEP_SIZE = 0.60
TIME_TOL = 1e-3


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


def load_vpm_particles(path: Path) -> dict[str, np.ndarray]:
    """Load the active particle arrays from one native VPM checkpoint."""
    import h5py

    with h5py.File(path, "r") as handle:
        count = int(handle["solver"].attrs["number_of_particles"])
        return {
            name: np.asarray(handle[f"particles/{name}"][:count])
            for name in ("position", "circulation", "radius")
        }


def run_constants() -> dict:
    """Return plot scales, falling back to the tutorial defaults."""
    meta = metadata()
    phys = meta.get("physics", {})
    freestream_velocity = np.asarray(phys.get("freestream_velocity", [1.0, 0.0, 0.0]), dtype=float)
    box = meta.get("fvm_solver", {}).get("fvm_domain", {})
    frames = slice_frames("fvm")
    if frames:
        import pyvista as pv

        bounds = pv.read(frames[0][1]).bounds
        box = {
            "xmin": float(bounds.x_min),
            "xmax": float(bounds.x_max),
            "ymin": float(bounds.y_min),
            "ymax": float(bounds.y_max),
            "zmin": box.get("zmin", -1.5),
            "zmax": box.get("zmax", 1.5),
        }
    return {
        "freestream_speed": float(np.linalg.norm(freestream_velocity)) or 1.0,
        "freestream_velocity": freestream_velocity,
        "D": 1.0,  # cube side length (CUBE_SIDE in cube_flow_setup.py)
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
    """Load a line history or a single-frame VPM line sample."""
    with path.open(encoding="utf-8") as stream:
        first_line = stream.readline()
    match = re.fullmatch(r"#\s*flow_time\s*=\s*([^\s]+)\s*", first_line)
    rows = np.atleast_1d(
        np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
            skip_header=1 if match else 0,
        )
    )
    table = {name: np.asarray(rows[name]) for name in rows.dtype.names}
    if "flow_time" in table:
        return table
    if match is None:
        raise ValueError(f"{path} has no flow_time column or metadata")
    table["flow_time"] = np.full(rows.size, float(match.group(1)))
    return table


def line_times(source: str, name: str) -> np.ndarray:
    """Sampling times available for one line sampler."""
    path = _path(source, name, ".csv")
    if not path.exists():
        return np.empty(0)
    return np.unique(_read_line_csv(path)["flow_time"])


def load_line(
    source: str, name: str, time: float, tol: float = TIME_TOL
) -> dict[str, np.ndarray] | None:
    """Return the frame of one line sampler at ``time``, sorted by x.

    ``None`` if this source has no sample within ``tol`` of ``time`` - callers
    must treat that as "no data for this panel", never substitute the nearest
    available frame regardless of how far away it is.
    """
    path = _path(source, name, ".csv")
    if not path.exists():
        return None
    table = _read_line_csv(path)
    times = np.unique(table["flow_time"])
    if times.size == 0:
        return None
    picked = times[np.argmin(np.abs(times - time))]
    if abs(picked - time) > tol:
        return None
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


def load_slice(
    source: str, time: float, name: str = "slice_z0", tol: float = TIME_TOL
) -> dict | None:
    """Return the slice snapshot at ``time`` as 2-D arrays on its grid.

    ``None`` if this source has no snapshot within ``tol`` of ``time`` - see
    :func:`load_line` for why a distant snapshot is never substituted.
    """
    import pyvista as pv

    frames = slice_frames(source, name)
    if not frames:
        return None
    picked_time, path = min(frames, key=lambda item: abs(item[0] - time))
    if abs(picked_time - time) > tol:
        return None
    grid = pv.read(path)
    ni, nj, _ = grid.dimensions
    shape = (nj, ni)
    legacy_fvm = source != "vpm" and "OpenONDASurfaceOrdering" not in grid.field_data

    def field(key, component=None):
        if key not in grid.point_data:
            return None
        values = np.asarray(grid.point_data[key], dtype=float)
        values = values if component is None else values[:, component]
        return values.reshape((ni, nj)).T if legacy_fvm else values.reshape(shape)

    points = np.asarray(grid.points, dtype=float)
    return {
        "time": picked_time,
        "x": points[:, 0].reshape(shape),
        "y": points[:, 1].reshape(shape),
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


def common_times(*series: np.ndarray, tol: float = TIME_TOL) -> np.ndarray:
    """Times present in every supplied sampler series."""
    if not series or any(np.asarray(values).size == 0 for values in series):
        return np.empty(0)
    base = np.unique(np.asarray(series[0], dtype=float))
    matched = [
        time
        for time in base
        if time > tol
        and all(
            np.min(np.abs(np.asarray(values, dtype=float) - time)) <= tol for values in series[1:]
        )
    ]
    return np.asarray(matched)


def comparison_times(time_step_size: float = PLOT_TIME_STEP_SIZE) -> np.ndarray:
    """Canonical ``dt``-spaced grid that drives every per-frame figure.

    Built from the union of every source's native sample times rather than
    one file, so the grid still reaches its full extent if a single sampler
    (e.g. an interrupted run) is short. Each requested time is independently
    resolved against the file that actually claims to have it - see
    :func:`load_line` / :func:`load_slice` - so a source missing a given frame
    just leaves that panel empty rather than showing a mistimed one.
    """
    sources = [
        line_times("reference", "centerline"),
        line_times("fvm", "centerline"),
        line_times("vpm", "centerline"),
        slice_times("reference"),
        slice_times("fvm"),
        slice_times("vpm"),
    ]
    available = [t for t in sources if t.size]
    if not available:
        return np.empty(0)
    t_max = max(t.max() for t in available)
    n_steps = int(round(t_max / time_step_size))
    return np.arange(1, n_steps + 1) * time_step_size
