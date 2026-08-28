"""Load sampled cylinder FVM-VPM results for analysis and plotting.

Every figure is built from sampler output alone — the line CSVs, slice VTS
files under ``samples/``, and the run metadata — so post-processing needs no
solver, no GPU and no raw field dumps. Three solutions are compared:

``reference``  fully meshed FVM (reference_flow/samples/)
``fvm``        the coupled run's FVM near field (samples/fvm_*)
``vpm``        the coupled run's VPM far field  (samples/vpm_*)
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(CASE_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CASE_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

COUPLED_CASE = CASE_DIR
REFERENCE_CASE = CASE_DIR / "reference_flow"
SOLUTION = COUPLED_CASE / "solution"
SAMPLES = COUPLED_CASE / "samples"
REFERENCE_SAMPLES = REFERENCE_CASE / "samples"
FIGURES = CASE_DIR / "figures"

# Physical problem (must match cylinder_shedding_flow_setup.py).
DIAMETER = 1.0
REYNOLDS = 150.0

# Keep one publication theme: reference and coupled figures are intended to be
# visually interchangeable.
_THEME_PATH = Path(__file__).resolve().parents[4] / "docs" / "themes" / "matplotlib_setup.py"
_THEME_SPEC = importlib.util.spec_from_file_location("openonda_matplotlib_setup", _THEME_PATH)
if _THEME_SPEC is None or _THEME_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load plotting theme from {_THEME_PATH}")
_THEME = importlib.util.module_from_spec(_THEME_SPEC)
_THEME_SPEC.loader.exec_module(_THEME)
_THEME.set_style()

CM = 1.0 / 2.54
FIGURE_WIDTH_CM = 12.5
FIGURE_WIDTH = FIGURE_WIDTH_CM * CM
FIGURE_DPI = _THEME.DEFAULT_DPI
EXPORT_FORMATS = _THEME.EXPORT_FORMATS
FONT_SIZE_PT = _THEME.FONT_SIZE_PT

matplotlib.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT,
        "figure.titlesize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

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

TIME_TOL = 1e-3
PROBE_NAME = (
    "midspan_probe"  # normalized_transverse_velocity(t) = velocity_y(1.5, 0, 0)/freestream_speed
)


def label(source: str) -> str:
    return SOURCES[source]["label"]


def colour(source: str) -> str:
    return COLORS["reference"] if source == "reference" else COLORS[source]


def _path(source: str, name: str, suffix: str) -> Path:
    entry = SOURCES[source]
    return entry["dir"] / f"{entry['prefix']}{name}{suffix}"


def metadata() -> dict:
    """Return coupled benchmark metadata, falling back to coupler metadata."""
    for name in ("benchmark_metadata.json", "run_metadata.json"):
        path = SOLUTION / name
        if path.exists():
            return json.loads(path.read_text())
    return {}


def run_constants() -> dict:
    """Return flow scales, falling back to the tutorial defaults."""
    meta = metadata()
    phys = meta.get("physics", {})
    freestream_velocity = np.asarray(phys.get("freestream_velocity", [1.0, 0.0, 0.0]), dtype=float)
    kinematic_viscosity = float(
        phys.get("kinematic_viscosity", float(np.linalg.norm(freestream_velocity)) / REYNOLDS)
    )
    end_time = float(phys.get("end_time", np.nan))
    return {
        "freestream_speed": float(np.linalg.norm(freestream_velocity)) or 1.0,
        "freestream_velocity": freestream_velocity,
        "diameter": DIAMETER,
        "reynolds": float(phys.get("reynolds_number", REYNOLDS)),
        "kinematic_viscosity": kinematic_viscosity,
        "end_time": end_time,
        "seed_amplitude": float(phys.get("seed_amplitude", meta.get("seed_amplitude", 0.0))),
    }


def figure_size(height_cm: float) -> tuple[float, float]:
    """Fixed 12.5 cm publication width and requested height in inches."""
    return FIGURE_WIDTH, height_cm * CM


def save(fig, name: str, fmt: str, dpi: int = FIGURE_DPI) -> Path:
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported figure format: {fmt!r}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.{fmt}"
    fig.savefig(out, format=fmt, dpi=dpi, bbox_inches=None, facecolor="white")
    print(f"  wrote {out.relative_to(CASE_DIR)}")
    return out


# ---- Time-series samplers -------------------------------------------------


def _read_csv(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if rows.size == 0:
        return None
    table = {name: np.asarray(rows[name]) for name in rows.dtype.names}
    return table


def load_probe(source: str, name: str = PROBE_NAME) -> dict[str, np.ndarray] | None:
    """Load the point-probe history ``normalized_transverse_velocity(t) = velocity_y/freestream_speed`` at x/D=1.5, z=0."""
    table = _read_csv(_path(source, name, ".csv"))
    if table is None:
        return None
    order = np.argsort(table["time"])
    freestream_speed = run_constants()["freestream_speed"]
    return {
        "t": table["time"][order],
        "normalized_transverse_velocity": table["velocity_y"][order] / freestream_speed,
        "step": table["step"][order],
    }


def load_forces(source: str) -> dict[str, np.ndarray] | None:
    """Load physical wall-patch forces from the reference or coupled FVM."""
    path = (
        REFERENCE_SAMPLES / "forces_history.csv"
        if source == "reference"
        else SAMPLES / "forces_history.csv"
    )
    table = _read_csv(path)
    if table is None:
        return None
    order = np.argsort(table["time"])
    return {
        "t": table["time"][order],
        "drag_coefficient": table["drag_coefficient"][order],
        "lift_coefficient": table["lift_coefficient"][order],
    }


def load_vpm_forces() -> dict[str, np.ndarray] | None:
    """Load the panel-body force history written by the coupled VPM solver."""
    table = _read_csv(SAMPLES / "vpm_forces.csv")
    if table is None:
        return None
    order = np.argsort(table["time"])
    return {name: table[name][order] for name in table}


# ---- Line and surface samplers --------------------------------------------


def _read_line_csv(path: Path) -> dict[str, np.ndarray]:
    """Load a line sample into the canonical in-memory field schema."""
    with path.open(encoding="utf-8") as stream:
        first_line = stream.readline()
    time_comment = re.fullmatch(r"#\s*time\s*=\s*([^\s]+)\s*", first_line)
    rows = np.atleast_1d(
        np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
            skip_header=1 if time_comment else 0,
        )
    )
    table = {name: np.asarray(rows[name]) for name in rows.dtype.names}
    if "time" in table:
        return table
    if time_comment is None:
        raise ValueError(f"{path} has no time column or metadata")
    table["time"] = np.full(rows.size, float(time_comment.group(1)))
    return table


def line_times(source: str, name: str) -> np.ndarray:
    """Sampling times available for one line sampler."""
    path = _path(source, name, ".csv")
    if not path.exists():
        return np.empty(0)
    return np.unique(_read_line_csv(path)["time"])


def load_line(
    source: str, name: str, time: float, tol: float = TIME_TOL
) -> dict[str, np.ndarray] | None:
    """Return one line-sampler frame sorted along its varying coordinate."""
    path = _path(source, name, ".csv")
    if not path.exists():
        return None
    table = _read_line_csv(path)
    times = np.unique(table["time"])
    if times.size == 0:
        return None
    picked = times[np.argmin(np.abs(times - time))]
    if abs(picked - time) > tol:
        return None
    mask = table["time"] == picked
    frame = {key: values[mask] for key, values in table.items()}
    coordinate = max(
        ("position_x", "position_y", "position_z"),
        key=lambda key: float(np.ptp(frame[key])),
    )
    order = np.argsort(frame[coordinate])
    frame = {key: values[order] for key, values in frame.items()}
    frame["time"] = picked
    return frame


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
    """Return the slice snapshot at ``time`` as 2-D arrays on its grid."""
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
    has_surface_ordering = "surface_ordering" in grid.field_data
    if source != "vpm" and not has_surface_ordering:
        raise ValueError(f"{path} has no canonical surface-ordering marker")

    def field(key, component=None):
        if key not in grid.point_data:
            return None
        values = np.asarray(grid.point_data[key], dtype=float)
        values = values if component is None else values[:, component]
        return values.reshape(shape)

    points = np.asarray(grid.points, dtype=float)
    return {
        "time": picked_time,
        "x": points[:, 0].reshape(shape),
        "y": points[:, 1].reshape(shape),
        "velocity_x": field("velocity", 0),
        "velocity_y": field("velocity", 1),
        "vorticity_z": field("vorticity", 2),
    }


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


def comparison_times(
    time_step_size: float, names: tuple[str, ...] = ("midspan_probe",)
) -> np.ndarray:
    """Canonical time-step-size-spaced grid that drives every per-frame figure."""
    sources = []
    for source in ("reference", "fvm", "vpm"):
        for name in names:
            sources.append(line_times(source, name))
        sources.append(slice_times(source))
    available = [t for t in sources if t.size]
    if not available:
        return np.empty(0)
    t_max = max(t.max() for t in available)
    n_steps = int(round(t_max / time_step_size))
    return np.arange(1, n_steps + 1) * time_step_size
