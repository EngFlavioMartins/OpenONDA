"""Load sampled cylinder FVM-VPM results for analysis and plotting.

Every figure is built from sampler output alone — the line CSVs, slice VTS
files under ``samples/``, and the run metadata — so post-processing needs no
solver, no GPU and no raw field dumps. Three solutions are compared:

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

# Physical problem (must match cylinderSheddingFlow_setup.py).
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
PROBE_NAME = "midspan_probe"  # q(t) = Uy(1.5, 0, 0)/Uinf


def label(source: str) -> str:
    return SOURCES[source]["label"]


def colour(source: str) -> str:
    return COLORS["reference"] if source == "reference" else COLORS[source]


def _path(source: str, name: str, suffix: str) -> Path:
    entry = SOURCES[source]
    return entry["dir"] / f"{entry['prefix']}{name}{suffix}"


def metadata() -> dict:
    """Return ``run_metadata.json`` as a dict (empty if not written)."""
    path = SOLUTION / "run_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_constants() -> dict:
    """Return flow scales, falling back to the tutorial defaults."""
    meta = metadata()
    phys = meta.get("physics", {})
    freestream_velocity = np.asarray(phys.get("freestream_velocity", [1.0, 0.0, 0.0]), dtype=float)
    kinematic_viscosity = float(
        phys.get("kinematic_viscosity", float(np.linalg.norm(freestream_velocity)) / REYNOLDS)
    )
    t_end = float(phys.get("end_time", np.nan))
    return {
        "freestream_speed": float(np.linalg.norm(freestream_velocity)) or 1.0,
        "freestream_velocity": freestream_velocity,
        "D": DIAMETER,
        "reynolds": REYNOLDS,
        "nu": kinematic_viscosity,
        "t_end": t_end,
        "seed_amplitude": float(meta.get("seed_amplitude", 0.0)),
    }


def save(fig, name: str, fmt: str, dpi: int) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.{fmt}"
    _THEME.save_fig(fig, out, figure_format=fmt, dpi=dpi)
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
    return {name: np.asarray(rows[name]) for name in rows.dtype.names}


def load_probe(source: str, name: str = PROBE_NAME) -> dict[str, np.ndarray] | None:
    """Load the point-probe history ``q(t) = Uy/Uinf`` at x/D=1.5, z=0."""
    table = _read_csv(_path(source, name, ".csv"))
    if table is None:
        return None
    order = np.argsort(table["flow_time"])
    u_inf = run_constants()["freestream_speed"]
    return {
        "t": table["flow_time"][order],
        "q": table["Uy"][order] / u_inf,
        "step": table["time_step"][order],
    }


def load_forces(source: str) -> dict[str, np.ndarray] | None:
    """Load the IBM force history for one source (``Cd``, ``Cl`` columns)."""
    table = _read_csv(_path(source, "ibm_forces_history", ".csv"))
    if table is None:
        return None
    order = np.argsort(table["time"])
    return {
        "t": table["time"][order],
        "Cd": table["Cd"][order],
        "Cl": table["Cl"][order],
        "slip": table["slip"][order],
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
    """Return the frame of one line sampler at ``time``, sorted by x."""
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
    """Canonical ``dt``-spaced grid that drives every per-frame figure."""
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
