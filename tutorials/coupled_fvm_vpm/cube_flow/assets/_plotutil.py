"""Load sampled FVM-VPM cube-flow results for plotting.

Every figure is built from sampler output alone — the line CSVs and slice VTS
files under ``samples/`` — so plotting needs no solver, no GPU and no raw
field dumps. Three solutions are compared:

``reference``  fully meshed FVM (reference_flow/samples/)
``fvm``        the coupled run's FVM near field (samples/fvm_*)
``vpm``        the coupled run's VPM far field  (samples/vpm_*)
"""

from __future__ import annotations

import importlib.util
import json
import matplotlib
from pathlib import Path
import re

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION = CASE_DIR / "solution"
SAMPLES = CASE_DIR / "samples"
REFERENCE_SAMPLES = CASE_DIR / "reference_flow" / "samples"
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

# Publication canvas and export settings shared by every cube-flow figure.
# Keep this width local: the cube-flow figures are intended for a 12.5 cm
# manuscript column, while the general OpenONDA theme also offers wider sizes.
CM = 1.0 / 2.54
FIGURE_WIDTH_CM = 12.5
FIGURE_WIDTH = FIGURE_WIDTH_CM * CM
FIGURE_DPI = _THEME.DEFAULT_DPI
EXPORT_FORMATS = _THEME.EXPORT_FORMATS

# Use the bundled DejaVu Serif face for both normal text and MathText.  This
# preserves the OpenONDA serif typography without launching an external LaTeX
# process once per frame (a complete run can contain hundreds of figures).
matplotlib.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": _THEME.FONT_SIZE_PT,
        "axes.labelsize": _THEME.FONT_SIZE_PT,
        "axes.titlesize": _THEME.FONT_SIZE_PT,
        "figure.titlesize": _THEME.FONT_SIZE_PT,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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

# Cross-solver comparisons require the same accepted physical time. This
# tolerance covers floating-point accumulation only; it must never admit a
# neighbouring VPM state.
TIME_ATOL = 1.0e-9


def label(source: str) -> str:
    return SOURCES[source]["label"]


def colour(source: str) -> str:
    return COLORS["reference"] if source == "reference" else COLORS[source]


def _path(source: str, name: str, suffix: str) -> Path:
    entry = SOURCES[source]
    return entry["dir"] / f"{entry['prefix']}{name}{suffix}"


def metadata() -> dict:
    """Return the metadata belonging to the selected coupled samples."""
    path = SOLUTION / "run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing run metadata for {SAMPLES}: expected {path}. "
            "Refusing to infer plotting scales or timestep provenance."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Run metadata must contain a JSON object: {path}")
    return data


def load_vpm_particles(path: Path) -> dict[str, np.ndarray]:
    """Load the active particle arrays from one native VPM checkpoint."""
    import h5py

    with h5py.File(path, "r") as handle:
        attrs = handle["solver"].attrs
        count = int(attrs["n_particles_total"])
        if count == 0:
            return {
                "position": np.empty((0, 3)),
                "vortex_strength": np.empty((0, 3)),
                "core_radius": np.empty(0),
            }

        particles = handle["particles"]
        return {
            "position": np.asarray(particles["position"][:count]),
            "vortex_strength": np.asarray(particles["vortex_strength"][:count]),
            "core_radius": np.asarray(particles["core_radius"][:count]),
        }


def run_constants() -> dict:
    """Return plot scales from the selected run metadata."""
    meta = metadata()
    phys = meta["physics"]
    freestream_velocity = np.asarray(phys["freestream_velocity"], dtype=float)
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
        "reference_length": 1.0,  # cube side length (CUBE_SIDE in cube_flow_setup.py)
        "kinematic_viscosity": float(phys["kinematic_viscosity"]),
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


def figure_size(height_cm: float) -> tuple[float, float]:
    """Return the fixed 12.5 cm publication width and ``height_cm`` in inches."""
    return FIGURE_WIDTH, height_cm * CM


def save(fig, name: str, fmt: str, dpi: int = FIGURE_DPI) -> Path:
    """Save without auto-cropping so the exported width stays exactly 12.5 cm.

    Every caller sets ``left``, ``right``, ``bottom``, ``top``, ``wspace`` and
    ``hspace`` explicitly with :meth:`matplotlib.figure.Figure.subplots_adjust`.
    Applying ``tight_layout`` or ``bbox_inches='tight'`` here would override
    those controls and change the physical canvas size.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported figure format: {fmt!r}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.{fmt}"
    fig.savefig(out, format=fmt, dpi=dpi, bbox_inches=None, facecolor="white")
    try:
        display_path = out.relative_to(CASE_DIR)
    except ValueError:
        display_path = out
    print(f"  wrote {display_path}")
    return out


# ---- Line samplers -------------------------------------------------------


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
    source: str, name: str, time: float, tol: float = TIME_ATOL
) -> dict[str, np.ndarray] | None:
    """Return the frame of one line sampler at ``time``, sorted by position_x.

    ``None`` if this source has no sample within ``tol`` of ``time`` - callers
    must treat that as "no data for this panel", never substitute the nearest
    available frame regardless of how far away it is.
    """
    path = _path(source, name, ".csv")
    if not path.exists():
        return None
    table = _read_line_csv(path)
    times = np.unique(table["time"])
    if times.size == 0:
        return None
    picked = times[np.argmin(np.abs(times - time))]
    if not np.isclose(picked, time, rtol=0.0, atol=tol):
        return None
    mask = table["time"] == picked
    frame = {key: values[mask] for key, values in table.items()}
    order = np.argsort(frame["position_x"])
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
    source: str, time: float, name: str = "slice_z0", tol: float = TIME_ATOL
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
    if not np.isclose(picked_time, time, rtol=0.0, atol=tol):
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


def common_times(*series: np.ndarray, tol: float = TIME_ATOL) -> np.ndarray:
    """Accepted physical times present in every supplied sampler series."""
    if not series or any(np.asarray(values).size == 0 for values in series):
        return np.empty(0)
    base = np.unique(np.asarray(series[0], dtype=float))
    matched = [
        time
        for time in base
        if time > tol
        and all(
            np.any(np.isclose(np.asarray(values, dtype=float), time, rtol=0.0, atol=tol))
            for values in series[1:]
        )
    ]
    return np.asarray(matched)


def _require_coincident_overlap(label: str, base: np.ndarray, *series: np.ndarray) -> np.ndarray:
    """Return exact common sample times after every source has begun sampling."""
    values = [np.unique(np.asarray(sample_times, dtype=float)) for sample_times in (base, *series)]
    positive = [sample_times[sample_times > TIME_ATOL] for sample_times in values]
    if any(sample_times.size == 0 for sample_times in positive):
        raise ValueError(f"{label} sources must contain positive sample times")

    overlap_start = max(sample_times[0] for sample_times in positive)
    overlap_end = min(sample_times[-1] for sample_times in positive)
    if overlap_end < overlap_start - TIME_ATOL:
        raise ValueError(f"{label} sources have no overlapping sample interval")

    base_overlap = positive[0][
        (positive[0] >= overlap_start - TIME_ATOL) & (positive[0] <= overlap_end + TIME_ATOL)
    ]
    matched = common_times(base_overlap, *positive[1:])
    if matched.size == 0:
        raise ValueError(f"{label} sources have no exact common sample times")
    return matched


def validate_plot_inputs() -> dict[str, float]:
    """Validate source provenance and exact cross-solver sample times."""
    meta = metadata()
    required_metadata = (
        ("physics", "freestream_velocity"),
        ("physics", "kinematic_viscosity"),
        ("physics", "end_time"),
    )
    for section, key in required_metadata:
        if key not in meta.get(section, {}):
            raise ValueError(f"Run metadata is missing {section}.{key}")

    required_files = [
        SOURCES["fvm"]["dir"] / "forces_history.csv",
        SOURCES["reference"]["dir"] / "forces_history.csv",
        SOLUTION / "coupler_diagnostics.jsonl",
    ]
    for source in ("reference", "fvm", "vpm"):
        required_files.extend(
            [_path(source, name, ".csv") for name in ("centreline", "offaxis_y075")]
        )
    for source in ("reference", "fvm", "vpm"):
        required_files.append(_path(source, "slice_z0", ".pvd"))
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing plotting input: {missing[0]}")

    profile_times = _require_coincident_overlap(
        "Profile",
        line_times("fvm", "centreline"),
        line_times("vpm", "centreline"),
        line_times("reference", "centreline"),
        line_times("fvm", "offaxis_y075"),
        line_times("vpm", "offaxis_y075"),
        line_times("reference", "offaxis_y075"),
    )
    field_times = _require_coincident_overlap(
        "Field",
        slice_times("fvm"),
        slice_times("vpm"),
        slice_times("reference"),
    )
    fvm_forces = load_forces("fvm")
    reference_forces = load_forces("reference")
    if fvm_forces is None or reference_forces is None:
        raise ValueError("Force histories must contain at least one row")
    force_times = _require_coincident_overlap("Force", fvm_forces["time"], reference_forces["time"])

    for source in ("reference", "fvm", "vpm"):
        for _, path in slice_frames(source):
            if not path.is_file():
                raise FileNotFoundError(f"PVD index references a missing field sample: {path}")

    return {
        "latest_profile_time": float(profile_times[-1]),
        "latest_field_time": float(field_times[-1]),
        "latest_force_time": float(force_times[-1]),
    }
