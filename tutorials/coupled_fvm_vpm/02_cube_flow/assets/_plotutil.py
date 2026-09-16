"""Cube comparison data and the fixed-width thesis figure style.

The two FVM fields are sampled offline with the same 3D reconstruction;
forces and VPM velocities come directly from the original samplers.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
import re

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION = CASE_DIR / "solution"
SAMPLES = CASE_DIR / "samples"
REFERENCE_SAMPLES = Path(
    os.environ.get(
        "OPENONDA_CUBE_REFERENCE_SAMPLES",
        str(CASE_DIR / "reference_flow" / "samples" / "fine"),
    )
)
FIGURES = CASE_DIR / "figures"

# Keep one publication theme: reference and coupled figures are intended to be
# visually interchangeable.
from openonda import plotting as _THEME

# Publication canvas and export settings shared by every cube-flow figure.
# Keep this width local: the cube-flow figures are intended for a 12.5 cm
# manuscript column, while the general OpenONDA theme also offers wider sizes.
CM = 1.0 / 2.54
FIGURE_WIDTH_CM = 12.5
FIGURE_WIDTH = FIGURE_WIDTH_CM * CM
FIGURE_DPI = _THEME.DEFAULT_DPI
EXPORT_FORMATS = _THEME.EXPORT_FORMATS
FONT_SIZE_PT = _THEME.THESIS_FONT_SIZE_PT
COMPARISON = SAMPLES / "comparison"

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
    """Load the active particle arrays from one native VPM backup."""
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
    freestream_speed = float(np.linalg.norm(freestream_velocity))
    if not np.isfinite(freestream_speed) or freestream_speed <= 0:
        raise ValueError("A positive finite freestream speed is required for normalization")
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
        "freestream_speed": freestream_speed,
        "freestream_velocity": freestream_velocity,
        "reference_length": 1.0,  # cube side length (CUBE_SIDE in setup.py)
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
    _THEME.validate_thesis_figure(fig, fig.axes)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.{fmt}"
    fig.savefig(out, format=fmt, dpi=dpi, bbox_inches=None, facecolor="white")
    try:
        display_path = out.relative_to(CASE_DIR)
    except ValueError:
        display_path = out
    print(f"  wrote {display_path}")
    return out


def remove_obsolete_frames(prefix: str, times: np.ndarray, fmt: str) -> None:
    """Retire generated frames that no longer have a matched source state."""
    expected = {f"{prefix}_t{time:.2f}.{fmt}" for time in times}
    for path in FIGURES.glob(f"{prefix}_t*.{fmt}"):
        generated_name = re.fullmatch(rf"{re.escape(prefix)}_t[0-9]+\.[0-9]{{2}}\.{fmt}", path.name)
        if generated_name and path.name not in expected:
            path.unlink()


def comparison_manifest() -> dict:
    path = COMPARISON / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run assets/prepare_fine_reference.py first")
    return json.loads(path.read_text())


def comparison_frame(time: float) -> dict | None:
    for row in comparison_manifest()["frames"]:
        if np.isclose(row["time"], time, rtol=0, atol=TIME_ATOL):
            with np.load(COMPARISON / row["file"], allow_pickle=False) as archive:
                return {key: archive[key] for key in archive.files}
    return None


# ---- Line samplers -------------------------------------------------------


@lru_cache(maxsize=None)
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
    names = rows.dtype.names
    if names is None:
        raise ValueError(f"{path} does not contain a named CSV table")
    table = {name: np.asarray(rows[name]) for name in names}
    if "step" in table and table["step"].size:
        reset = np.flatnonzero(np.diff(table["step"].astype(int)) < 0)
        if reset.size:
            raise ValueError(f"Line samples contain a restarted history: {path}")
    if "time" in table:
        return table
    if time_comment is None:
        raise ValueError(f"{path} has no time column or metadata")
    table["time"] = np.full(rows.size, float(time_comment.group(1)))
    return table


def line_times(source: str, name: str) -> np.ndarray:
    """Sampling times available for one line sampler."""
    if source != "vpm":
        return np.array([row["time"] for row in comparison_manifest()["frames"]])
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
    if source != "vpm":
        frame = comparison_frame(time)
        if frame is None:
            return None
        points = frame[f"{name}_points"]
        velocity = frame[f"{name}_{source}"]
        return {
            "time": time,
            **{f"position_{axis}": points[:, i] for i, axis in enumerate("xyz")},
            **{f"velocity_{axis}": velocity[:, i] for i, axis in enumerate("xyz")},
        }
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


@lru_cache(maxsize=None)
def slice_frames(source: str, name: str = "slice_z0") -> list[tuple[float, Path]]:
    """Return (time, path) for every slice snapshot listed in the PVD index."""
    pvd = _path(source, name, ".pvd")
    if not pvd.exists():
        return []
    matches = re.finditer(r'timestep="([^"]+)"\s+[^>]*file="([^"]+)"', pvd.read_text())
    return sorted((float(m.group(1)), pvd.parent / m.group(2)) for m in matches)


def slice_times(source: str, name: str = "slice_z0") -> np.ndarray:
    if source != "vpm":
        return np.array([row["time"] for row in comparison_manifest()["frames"]])
    return np.array([t for t, _ in slice_frames(source, name)])


def load_slice(
    source: str, time: float, name: str = "slice_z0", tol: float = TIME_ATOL
) -> dict | None:
    """Return the slice snapshot at ``time`` as 2-D arrays on its grid.

    ``None`` if this source has no snapshot within ``tol`` of ``time`` - see
    :func:`load_line` for why a distant snapshot is never substituted.
    """
    import pyvista as pv

    if source != "vpm":
        frame = comparison_frame(time)
        if frame is None:
            return None
        velocity = frame[f"slice_{source}"]
        return {
            "time": time,
            "x": frame["slice_x"],
            "y": frame["slice_y"],
            "velocity": velocity,
            "valid": np.all(np.isfinite(velocity), axis=-1),
            **{f"velocity_{axis}": velocity[..., i] for i, axis in enumerate("xyz")},
        }

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
        "velocity_z": field("velocity", 2),
        "velocity": np.asarray(grid.point_data["velocity"], dtype=float).reshape(*shape, 3),
        "valid": np.asarray(
            grid.point_data.get("vtkValidPointMask", np.ones(ni * nj)), dtype=bool
        ).reshape(shape),
        "vorticity_z": field("vorticity", 2),
    }


# ---- Forces --------------------------------------------------------------


def load_forces(source: str) -> dict[str, np.ndarray] | None:
    """Load raw cube forces, rejecting ambiguous or nonmonotonic histories."""
    path = SOURCES[source]["dir"] / "forces_history.csv"
    if not path.exists():
        return None
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if rows.size == 0:
        return None
    names = rows.dtype.names
    if names is None:
        raise ValueError(f"{path} does not contain a named CSV table")
    if "patch" in names and np.any(rows["patch"] != "cube"):
        raise ValueError(f"Unexpected force patch in {path}")
    if np.any(~np.isfinite(rows["time"])) or np.any(np.diff(rows["time"]) <= 0):
        raise ValueError(f"Force times are duplicate or nonmonotonic in {path}")
    if np.any(~np.isfinite(rows["drag_coefficient"])):
        raise ValueError(f"Non-finite raw drag coefficient in {path}")
    return {name: np.asarray(rows[name]) for name in names}


def load_vpm_forces() -> dict[str, np.ndarray] | None:
    """Load the panel-body force history written by the coupled VPM solver."""
    path = SAMPLES / "vpm_forces.csv"
    if not path.exists():
        return None
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    names = rows.dtype.names
    if names is None:
        raise ValueError(f"{path} does not contain a named CSV table")
    return {name: np.asarray(rows[name]) for name in names} if rows.size else None


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


def _validate_metadata_provenance(meta: dict) -> None:
    """Accept current metadata while keeping existing schema-2 runs readable."""
    provenance = (meta.get("schema_version"), meta.get("coupling_method"))
    supported_provenance = {
        (2, "absolute_common_m4_lattice_blend"),
        (3, "buffered_m4_renewal"),
    }
    if provenance not in supported_provenance:
        raise ValueError(
            "Samples do not belong to a supported cube-flow coupling metadata schema: "
            f"schema={provenance[0]!r}, method={provenance[1]!r}"
        )


def _is_lfs_pointer(path: Path) -> bool:
    """Return whether ``path`` is an unhydrated Git-LFS pointer file."""
    with path.open("rb") as stream:
        return stream.readline().startswith(b"version https://git-lfs.github.com/spec/")


_PIMPLE_COMPARISON_FIELDS = (
    "type",
    "algorithm",
    "n_correctors",
    "n_outer_correctors",
    "n_nonorthogonal_correctors",
    "min_outer_correctors",
    "outer_residual_tolerance",
    "outer_continuity_tolerance",
    "max_iterations",
    "tolerance",
    "velocity_relaxation",
    "pressure_relaxation",
    "ddt_corr",
    "ibm_forcing_loops",
    "ibm_second_solve",
)


def _pimple_configurations_match(coupled: dict, reference: dict) -> bool:
    """Compare every current executable PIMPLE control recorded by both runs."""
    try:
        return all(coupled[key] == reference[key] for key in _PIMPLE_COMPARISON_FIELDS)
    except KeyError:
        return False


def comparison_configurations() -> tuple[dict, dict]:
    reference = json.loads(
        (CASE_DIR / "reference_flow" / "solution" / "fine" / "fvm_metadata.json").read_text()
    )["configuration"]
    coupled = json.loads((SOLUTION / "fvm_metadata.json").read_text())["configuration"]
    for key in (
        "transport",
        "initial_velocity",
        "initial_kinematic_pressure",
        "schemes",
        "turbulence",
    ):
        if coupled[key] != reference[key]:
            raise ValueError(f"Coupled/fine configurations differ in {key}; review the comparison")
    if not _pimple_configurations_match(coupled["pimple"], reference["pimple"]):
        raise ValueError("Coupled/fine configurations differ in pimple; review the comparison")
    for key in (
        "momentum_tolerance",
        "pressure_tolerance",
        "momentum_relative_tolerance",
        "pressure_relative_tolerance",
        "momentum_final_relative_tolerance",
        "pressure_final_relative_tolerance",
    ):
        if coupled["linear"][key] != reference["linear"][key]:
            raise ValueError(f"Coupled/fine linear tolerances differ: {key}")

    def force_definition(config):
        samplers = [s for s in config["samplers"] if s["type"] == "ForceSampler"]
        if len(samplers) != 1:
            raise ValueError("Expected one cube force sampler")
        return {
            key: value for key, value in samplers[0].items() if key not in ("schedule", "file_name")
        }

    if force_definition(coupled) != force_definition(reference):
        raise ValueError("Force patches or normalization differ")
    force = force_definition(reference)
    if (
        force["patch_names"] != ["cube"]
        or force["reference_length"] != 1
        or force["reference_area"] != 1
        or force["reference_velocity"] != run_constants()["freestream_speed"]
    ):
        raise ValueError("Unexpected cube force normalization")
    return coupled, reference


def validate_plot_inputs() -> dict[str, float]:
    """Check physical provenance and exact times; adaptive dt need not match."""
    from .prepare_fine_reference import validate_reference

    validate_reference()
    _validate_metadata_provenance(metadata())
    comparison_configurations()
    profile_times = common_times(
        *(
            line_times(s, n)
            for s in ("fvm", "vpm", "reference")
            for n in ("centreline", "offaxis_y075")
        )
    )
    field_times = common_times(*(slice_times(s) for s in ("fvm", "vpm", "reference")))
    fvm_forces, reference_forces = load_forces("fvm"), load_forces("reference")
    if fvm_forces is None or reference_forces is None:
        raise ValueError("Both force histories are required")
    force_times = common_times(fvm_forces["time"], reference_forces["time"])
    for source, forces in (("coupled", fvm_forces), ("reference", reference_forces)):
        accepted = forces.get("accepted_time_step_size")
        if (
            accepted is None
            or np.any(~np.isfinite(accepted))
            or np.any(accepted[forces["time"] > 0] <= 0)
        ):
            raise ValueError(f"Invalid accepted timestep in {source} force history")
    if any(len(t) == 0 for t in (profile_times, field_times, force_times)):
        raise ValueError("No exactly coincident comparison states")
    return {
        "latest_profile_time": float(profile_times[-1]),
        "latest_field_time": float(field_times[-1]),
        "latest_force_time": float(force_times[-1]),
    }
