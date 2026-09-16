"""Cube comparison data and the fixed-width thesis figure style.

The two FVM fields are sampled offline with the same 3D reconstruction;
forces and VPM velocities come directly from the original samplers.
"""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import csv
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

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
AUXILIARY = FIGURES / "auxiliary"

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
PREPARATION_METHOD = "native-centres-affine-k12-v1"


class ComparisonNotReady(RuntimeError):
    """The runs have not saved the common states needed for comparison."""


def _pvd_frames(pvd: Path) -> list[tuple[float, Path]]:
    if not pvd.is_file():
        raise FileNotFoundError(pvd)
    result = sorted(
        (float(item.attrib["timestep"]), pvd.parent / item.attrib["file"])
        for item in ET.parse(pvd).iter("DataSet")
    )
    if any(b[0] <= a[0] for a, b in zip(result, result[1:])):
        raise ValueError(f"Duplicate states in {pvd}")
    return result


def _frame_at_time(items: list[tuple[float, Path]], time: float) -> Path | None:
    return next(
        (path for value, path in items if np.isclose(value, time, rtol=0, atol=TIME_ATOL)), None
    )


def _file_stamp(path: Path) -> list:
    """Include every MPI piece so rerunning a case invalidates derived data."""
    paths = [path]
    if path.suffix == ".pvtu":
        paths += [path.parent / item.attrib["Source"] for item in ET.parse(path).iter("Piece")]
    return [[str(item.resolve()), item.stat().st_size, item.stat().st_mtime_ns] for item in paths]


def fluid_points(points: np.ndarray) -> np.ndarray:
    """Return points outside the closed unit cube, without an artificial halo."""
    return ~np.all(np.abs(points) <= 0.5 + 1e-12, axis=-1)


class NativeVelocity:
    """Sample archived velocity using native centroids and the FVM affine probe."""

    def __init__(self, mesh_path: Path):
        from source.solvers.fvm.io.mesh_storage import load_native_mesh
        from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

        mesh = load_native_mesh(mesh_path)
        self.centres = compute_mesh_geometry(mesh, compute_lsq=False)["cell_centre"]
        self.probes = {}

    def sample(self, source: Path, queries: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        import pyvista as pv
        from source.solvers.fvm.sampling.fields import _PointProbe

        grid = pv.read(source)
        data = grid.cell_data
        velocity = np.asarray(data["velocity"], dtype=float)
        keep = np.asarray(data.get("vtkGhostType", np.zeros(len(velocity)))) == 0
        ids = np.asarray(data.get("global_cell_id", np.arange(len(velocity))), dtype=int)[keep]
        if len(ids) != len(self.centres) or not np.array_equal(
            np.sort(ids), np.arange(len(self.centres))
        ):
            raise ValueError(f"Snapshot does not cover its native mesh exactly once: {source}")
        ordered = np.empty_like(velocity[keep])
        ordered[ids] = velocity[keep]
        if not np.all(np.isfinite(ordered)):
            raise ValueError(f"Non-finite archived velocity in {source}")
        result = {}
        for name, points in queries.items():
            key = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
            if key not in self.probes:
                inside = (grid.find_containing_cell(points) >= 0) & fluid_points(points)
                self.probes[key] = (_PointProbe(points, k=12, reconstruction="affine"), inside)
            probe, inside = self.probes[key]
            values = probe._interpolate(ordered, self.centres)
            values[~inside] = np.nan
            result[name] = values
        return result


def validate_reference() -> dict:
    """Validate that every comparison uses the registered fine reference."""
    expected = CASE_DIR / "reference_flow" / "samples" / "fine"
    if REFERENCE_SAMPLES.resolve() != expected.resolve():
        raise ValueError(f"Cube comparisons require the fine reference: {expected}")
    info = json.loads((expected / "grid_run.json").read_text())
    if info["case"] != "fine" or not np.isclose(info["cell_size"], 0.06, rtol=0, atol=1e-12):
        raise ValueError("Expected the registered fine reference with dx=0.06")
    return info


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
        raise FileNotFoundError(f"Missing {path}; run assets/postprocess.py first")
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


def prepare_comparison_fields() -> int:
    """Prepare exactly coincident FVM fields on the native comparison lattice."""
    info = validate_reference()
    reference_solution = CASE_DIR / "reference_flow" / "solution" / "fine"
    reference_config = reference_solution / "fvm_metadata.json"
    coupled_config = SOLUTION / "fvm_metadata.json"
    configuration = json.loads(coupled_config.read_text())
    coupled_frames = _pvd_frames(SOLUTION / f"{configuration['case_name']}.pvd")
    reference_frames = _pvd_frames(reference_solution / "fine.pvd")
    vpm_frames = _pvd_frames(SAMPLES / "vpm_slice_z0.pvd")
    fvm_slices = _pvd_frames(SAMPLES / "fvm_slice_z0.pvd")
    line_names = ("centreline", "offaxis_y075")
    times = common_times(
        np.array([time for time, _ in reference_frames]),
        np.array([time for time, _ in coupled_frames]),
        np.array([time for time, _ in vpm_frames]),
        np.array([time for time, _ in fvm_slices]),
        line_times("vpm", "centreline"),
        line_times("vpm", "offaxis_y075"),
    )
    if not len(times):
        raise ComparisonNotReady("No common saved Reference FVM, Coupled FVM and VPM state yet.")
    COMPARISON.mkdir(parents=True, exist_ok=True)
    manifest_path = COMPARISON / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"frames": []}
    meshes = {"reference": reference_solution / "mesh.npz", "fvm": SOLUTION / "mesh.npz"}
    fixed = {
        "method": PREPARATION_METHOD,
        "meshes": {name: _file_stamp(path) for name, path in meshes.items()},
        "configurations": [_file_stamp(reference_config), _file_stamp(coupled_config)],
    }
    samplers: dict[str, NativeVelocity] = {}
    rows = []
    count = 0
    import pyvista as pv

    for time in times:
        raw = {
            "reference": _frame_at_time(reference_frames, time),
            "fvm": _frame_at_time(coupled_frames, time),
        }
        slice_path = _frame_at_time(fvm_slices, time)
        if slice_path is None or any(path is None for path in raw.values()):
            raise ComparisonNotReady(
                f"A matched field at t={time:g} disappeared during preparation"
            )
        grid = pv.read(slice_path)
        ni, nj, _ = grid.dimensions
        shape = (nj, ni)
        points = np.asarray(grid.points, dtype=float)
        queries = {"slice": points}
        for name in line_names:
            line = load_line("vpm", name, time)
            queries[name] = np.column_stack([line[f"position_{axis}"] for axis in "xyz"])
        signature = {
            **fixed,
            "fields": {name: _file_stamp(path) for name, path in raw.items()},
            "slice": _file_stamp(slice_path),
            "points": {
                name: hashlib.sha256(values.tobytes()).hexdigest()
                for name, values in queries.items()
            },
        }
        destination = COMPARISON / f"fields_t{time:.9f}.npz"
        entry = {"time": float(time), "file": destination.name, "inputs": signature}
        cached = any(row == entry for row in previous["frames"]) and destination.is_file()
        if not cached:
            arrays = {
                "slice_x": points[:, 0].reshape(shape),
                "slice_y": points[:, 1].reshape(shape),
            }
            for source, path in raw.items():
                if source not in samplers:
                    print(f"  preparing native {source} sampling geometry", flush=True)
                    samplers[source] = NativeVelocity(meshes[source])
                fields = samplers[source].sample(path, queries)
                arrays[f"slice_{source}"] = fields["slice"].reshape(*shape, 3)
                for name in line_names:
                    arrays[f"{name}_points"] = queries[name]
                    arrays[f"{name}_{source}"] = fields[name]
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, destination)
            count += 1
            print(f"  prepared matched FVM fields at t={time:g}", flush=True)
        rows.append(entry)
    manifest = {"method": PREPARATION_METHOD, "reference": info, "frames": rows}
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    return count


def validate_plot_inputs() -> dict[str, float]:
    """Check physical provenance and exact times; adaptive dt need not match."""
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


def mesh_summary(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        wall = next(b for b in meta["boundary"] if b["name"] == "cube")
        owners = archive["owners"][wall["start_face"] : wall["start_face"] + wall["n_faces"]]
        return {
            "cells": meta["n_cells"],
            "cube_faces": wall["n_faces"],
            "finest_cartesian_spacing": float(np.min(archive["cell_sizes"])),
            "cube_adjacent_cartesian_spacings": np.unique(archive["cell_sizes"][owners]).tolist(),
        }


def build_comparison_report():
    limits = validate_plot_inputs()
    coupled, reference = comparison_configurations()
    end = min(limits.values())
    forces = load_forces("reference")
    selected = (forces["time"] >= 1) & (forces["time"] <= end + TIME_ATOL)
    indices = np.flatnonzero(selected)
    if not len(indices):
        indices = np.flatnonzero(forces["time"] <= end + TIME_ATOL)
    peaks = indices[np.argsort(forces["drag_coefficient"][indices])[-3:][::-1]]
    records = []
    for i in peaks:
        records.append(
            {
                "step": int(forces["step"][i]),
                "time": float(forces["time"][i]),
                "Cd": float(forces["drag_coefficient"][i]),
                "accepted_dt": float(forces["accepted_time_step_size"][i]),
            }
        )
    pressure = []
    peak_step = records[0]["step"]
    with (CASE_DIR / "reference_flow/solution/fine/diagnostics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["step"] in (peak_step - 1, peak_step, peak_step + 1):
                pressure.append(
                    {
                        k: row[k]
                        for k in (
                            "step",
                            "time",
                            "time_step_size",
                            "min_kinematic_pressure",
                            "max_kinematic_pressure",
                            "max_velocity_magnitude",
                            "max_continuity_error",
                        )
                    }
                )
    meshes = {
        "coupled": mesh_summary(SOLUTION / "mesh.npz"),
        "fine": mesh_summary(CASE_DIR / "reference_flow/solution/fine/mesh.npz"),
    }
    if (
        meshes["coupled"]["cube_adjacent_cartesian_spacings"]
        != meshes["fine"]["cube_adjacent_cartesian_spacings"]
    ):
        raise ValueError("The cube-adjacent Cartesian spacings differ")
    before = next((row for row in pressure if row["step"] == peak_step - 1), None)
    at_peak = next((row for row in pressure if row["step"] == peak_step), None)
    ratios = None
    if before and at_peak:
        span_before = before["max_kinematic_pressure"] - before["min_kinematic_pressure"]
        span_peak = at_peak["max_kinematic_pressure"] - at_peak["min_kinematic_pressure"]
        if span_before > 0 and at_peak["time_step_size"] > 0:
            ratios = {
                "pressure_span_increase": span_peak / span_before,
                "timestep_reduction": before["time_step_size"] / at_peak["time_step_size"],
            }
    field_metrics = []
    for comparison in (
        "coupled_fvm_vpm_fields",
        "reference_fvm_vpm_fields",
        "reference_fvm_coupled_fvm_fields",
    ):
        metrics_path = AUXILIARY / f"{comparison}.csv"
        if metrics_path.is_file():
            with metrics_path.open() as stream:
                field_metrics.extend(
                    row
                    for row in csv.DictReader(stream)
                    if np.isclose(float(row["time"]), end, rtol=0, atol=TIME_ATOL)
                )
    return {
        "reference_samples": str(REFERENCE_SAMPLES),
        "reference_grid": "fine",
        "comparison_end_time": end,
        "meshes": meshes,
        "matched_configuration_sections": [
            "transport",
            "initial conditions",
            "schemes",
            "PIMPLE",
            "turbulence",
            "linear tolerances",
            "force normalization",
        ],
        "coupled_time_config": coupled["time"],
        "reference_time_config": reference["time"],
        "largest_reference_Cd_after_t1": records,
        "reference_pressure_around_largest_Cd": pressure,
        "ratios_at_largest_Cd": ratios,
        "latest_field_differences": field_metrics,
        "sampling": comparison_manifest()["method"],
        "definition": "100 * norm(u_test - u_comparison, 2) / U_inf; all three components",
        "rms": "sqrt(sum(area_weight * difference_percent**2) / sum(area_weight))",
        "support": "z=0, common finite fluid quads in [-1.5,1.5]^2; no extrapolation or body halo",
        "display": "bilinear velocity interpolation, then vector norm; full native maximum colour range",
    }


def write_comparison_report(report):
    root = AUXILIARY
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    peak = report["largest_reference_Cd_after_t1"][0]
    mesh = report["meshes"]
    ratios = report["ratios_at_largest_Cd"]
    excursion = (
        ratios is not None
        and ratios["pressure_span_increase"] > 10
        and ratios["timestep_reduction"] > 10
    )
    observation = "Neighbouring pressure/timestep ratios are unavailable."
    if ratios:
        observation = (
            f"At this sample the pressure span increases by a factor of "
            f"{ratios['pressure_span_increase']:.3g}, and the timestep decreases "
            f"by a factor of {ratios['timestep_reduction']:.3g} relative to the preceding step."
        )
    verdict = (
        "The reference drag is not a trustworthy accuracy benchmark until this "
        "pressure/timestep excursion is resolved and checked."
        if excursion
        else "This diagnostic does not establish that the reference is converged or accurate."
    )
    latest_table = "No field metrics have been generated at this comparison time."
    if report["latest_field_differences"]:
        labels = {
            "reference_fvm_coupled_fvm_fields": (
                "Reference FVM / Coupled FVM (primary near field)"
            ),
            "reference_fvm_vpm_fields": "Reference FVM / VPM (auxiliary overlap field)",
            "coupled_fvm_vpm_fields": "Coupled FVM / VPM (overlap consistency)",
        }
        latest_table = "| Comparison | RMS [% U_inf] | Sampled max [% U_inf] | Area [D^2] |\n"
        latest_table += "|---|---:|---:|---:|\n"
        for row in report["latest_field_differences"]:
            latest_table += (
                f"| {labels[row['figure']]} | {float(row['rms_percent']):.3f} | "
                f"{float(row['sampled_max_percent']):.3f} | {float(row['covered_area_D2']):.4f} |\n"
            )
    text = f"""# Cube comparison report

Reference: reference_flow/samples/fine/ and reference_flow/solution/fine/.
Comparison ends at t={report["comparison_end_time"]:g} s. Reference data after
this time are not used in the figures. No simulation was advanced by plotting.

## What is matched

- Density, viscosity, initial conditions, FVM spatial/time schemes, turbulence
  closure, PIMPLE correctors/relaxation, linear tolerances and force definitions
  agree in the saved configurations.
- Both meshes have cube-adjacent Cartesian spacing
  {mesh["fine"]["finest_cartesian_spacing"]:.6g} m (requested fine target: 0.06 m).
  The coupled mesh has {mesh["coupled"]["cells"]:,} cells and
  {mesh["coupled"]["cube_faces"]:,} cube faces; the reference has
  {mesh["fine"]["cells"]:,} cells and {mesh["fine"]["cube_faces"]:,} cube faces.
  Their fitted wall cells and outer boundaries are not identical.
- Both FVM velocity fields use the existing 3D affine reconstruction with 12
  native volume-centroid neighbours and its documented IDW fallback. MPI
  global cell IDs are checked for complete, unique coverage.
- Only coincident saved physical states are used (absolute tolerance 1e-9 s).
  The reference saves full fields at 1 s intervals, so matched profile/field
  figures use that cadence. No interpolation between times is performed.
  The original force histories retain their 0.05 s sampling.
- Forces are the raw pressure-plus-viscous cube-wall forces, normalized by
  0.5 rho U_inf^2 D^2, with rho=1, U_inf=1 and D=1. No smoothing, outlier removal
  or drag-axis clipping is used.

## Meaning of the differences

Colour shows 100 ||u_test - u_comparison||_2 / U_inf, including u_x, u_y and u_z.
The velocity panels themselves show u_x/U_inf. Normalizing by freestream speed
avoids division by a vanishing local velocity. The word “difference” is used
because a finite-mesh reference is not an exact solution.

The three auxiliary field-comparison CSV files record the area-weighted sampled RMS,
sampled maximum, valid-node count and covered area for each figure. Each valid grid rectangle
distributes one quarter of its area to each vertex. The RMS is a quadrature
estimate on this sampled z=0 plane, not a 3D volume norm. All three fields must
be finite at a point, giving every pairing the same support; the cube and
rectangles with missing corners are excluded.
Unresolved strips adjacent to the wall are not counted as zero error. Covered
area is reported so this limitation is visible.

Error contours use bilinearly interpolated velocity vectors; their difference
is formed before taking the vector norm. Metrics are calculated on the original
sample grid (spacing about 0.12 m), independently of display resolution.
Contours cannot recover structures absent from those samples. No nearest-point
extrapolation or 95th-percentile colour clipping is used. Colour ranges are
shared between the two velocity panels of each figure and may change with time.

reference_fvm_coupled_fvm_fields_* compares the primary near-body solution with fine FVM.
reference_fvm_vpm_fields_* and coupled_fvm_vpm_fields_* diagnose VPM in the overlap
region, where VPM is auxiliary. They are not whole-domain hybrid error maps.
The line profiles include the sampled outer wake. A z=0 section of 3D fields
does not establish accuracy everywhere in three dimensions.

At the latest compared time, t={report["comparison_end_time"]:g} s:

{latest_table}

These are instantaneous sampled differences, not time-averaged error estimates.

## Reference drag anomaly and remaining validation limits

The largest saved reference Cd after t=1 within the compared interval is
{peak["Cd"]:.9g} at t={peak["time"]:g} s, with accepted dt={peak["accepted_dt"]:.9g} s.
reference_force_history.* shows raw Cd and the accepted timestep; the vertical
line marks this sample. auxiliary/comparison_report.json records neighbouring pressure
extrema where solver diagnostics are available. These are observations, not
a completed diagnosis of the pressure/timestep algorithm.

{observation}
{verdict}
The diagnostic flags simultaneous pressure-span growth and timestep reduction
above a factor of ten; this is a screening rule, not a convergence criterion,
and it never filters the plotted data. Hiding suspect points would invalidate
the comparison. The fine run uses adaptive timesteps and the coupled FVM uses fixed
0.01 s steps, so temporal error is not isolated even though the schemes match.
The saved fine grid alone supplies no mesh/time convergence or statistical
uncertainty estimate. These figures can document the comparison and anomaly;
they do not yet support a claim of validated hybrid accuracy.

## Figure contract

Vector PDFs are exactly 125 mm wide, with embedded NewPX text/math fonts at
10.95 pt for main text. Include at natural size. PNG previews are 400 dpi.
All axes use equal outer side margins; the shared thesis validator checks font
sizes, a minimum 5 pt text-to-canvas clearance and text overlap before saving.
Line widths are 1.1 pt (primary), 1.0 pt (reference), and 0.5 pt (axes).
Captions belong in the thesis/paper; identify the slice, time, normalization,
common support, auxiliary-field status and finite-reference limitations.
"""
    (root / "comparison_report.md").write_text(text)
    print(
        f"Reference drag check: Cd={peak['Cd']:.6g}, t={peak['time']:g}, dt={peak['accepted_dt']:.6g}"
    )
    print(verdict)


def main() -> int:
    """Prepare and validate the matched data consumed by every cube plotter."""
    try:
        count = prepare_comparison_fields()
    except (FileNotFoundError, ComparisonNotReady) as error:
        detail = (
            f"Required saved output is missing: {error.filename or error.args[0]}"
            if isinstance(error, FileNotFoundError)
            else str(error)
        )
        print(
            f"Comparison plots are not ready yet. {detail}\n"
            "Rerun ./allplot.sh after both simulations have saved full FVM fields "
            "and VPM samples at a common time. Existing results and figures were left unchanged.",
            file=sys.stderr,
        )
        return 2
    result = validate_plot_inputs()
    print(f"Prepared {count} matched comparison state(s).")
    print(
        "Validated plotting inputs: "
        f"profiles through t={result['latest_profile_time']:.6g} s, "
        f"fields through t={result['latest_field_time']:.6g} s, "
        f"forces through t={result['latest_force_time']:.6g} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
