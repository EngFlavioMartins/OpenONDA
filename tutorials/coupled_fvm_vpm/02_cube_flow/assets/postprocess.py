"""Sample the saved cube-flow fields on a common comparison lattice."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv

from openonda import plotting as _THEME
from openonda.results import (
    NativeVelocity,
    load_vpm_particles,
    read_csv_table,
    read_history_table,
    read_json,
    read_npz_arrays,
    read_pvd_frames,
    read_surface_frame,
    write_json,
    write_npz_arrays,
)
from openonda.saved_times import match_saved_times, same_saved_time

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION = CASE_DIR / "solution"
SAMPLES = CASE_DIR / "samples"
REFERENCE_SAMPLES = CASE_DIR / "reference_flow/samples/fine"
_DEFAULT_REFERENCE_SAMPLES = REFERENCE_SAMPLES
FIGURES = CASE_DIR / "figures"
AUXILIARY = FIGURES / "auxiliary"
COMPARISON = SAMPLES / "comparison"

CM = 1.0 / 2.54
FIGURE_WIDTH_CM = 12.5
FIGURE_WIDTH = FIGURE_WIDTH_CM * CM
FIGURE_DPI = _THEME.DEFAULT_DPI
EXPORT_FORMATS = _THEME.FORMAT_CHOICES
FONT_SIZE_PT = _THEME.THESIS_FONT_SIZE_PT
TIME_ATOL = 1.0e-9
PREPARATION_METHOD = "native-centres-affine-k12-v1"

COLORS = dict(_THEME.COLORS)
COLORS.update(
    {
        "fvm": COLORS["teal"],
        "vpm": COLORS["vpm"],
        "cd": COLORS["hybrid"],
        "cl": COLORS["vpm"],
        "accent": COLORS["text"],
        "box": COLORS["background_strong"],
    }
)
COLORMAPS = dict(_THEME.COLORMAPS)
SOURCES = {
    "reference": {"dir": REFERENCE_SAMPLES, "prefix": "", "label": "Reference FVM"},
    "fvm": {"dir": SAMPLES, "prefix": "fvm_", "label": "Coupled FVM"},
    "vpm": {"dir": SAMPLES, "prefix": "vpm_", "label": "Coupled VPM"},
}


def reference_run():
    root = CASE_DIR / "reference_flow"
    name = "fine"
    return SimpleNamespace(
        name=name,
        solution=root / "solution" / name,
        samples=root / "samples" / name,
    )


def _fvm_output_files(solution):
    return solution / "fvm.pvd", solution / "fvm/mesh.npz"


def fluid_points(points):
    return ~np.all(np.abs(points) <= 0.5 + 1e-12, axis=-1)


def label(source):
    return SOURCES[source]["label"]


def colour(source):
    return COLORS["reference"] if source == "reference" else COLORS[source]


def _source_directory(source):
    directory = SOURCES[source]["dir"]
    if source == "reference" and directory == _DEFAULT_REFERENCE_SAMPLES:
        return reference_run().samples
    return Path(directory)


def _path(source, name, suffix):
    return _source_directory(source) / f"{SOURCES[source]['prefix']}{name}{suffix}"


def metadata():
    return read_json(SOLUTION / "run_metadata.json")


def run_constants():
    recorded = metadata()
    physics = recorded["physics"]
    velocity = np.asarray(physics["freestream_velocity"], dtype=float)
    return {
        "freestream_speed": float(np.linalg.norm(velocity)),
        "freestream_velocity": velocity,
        "reference_length": 1.0,
        "kinematic_viscosity": float(physics["kinematic_viscosity"]),
        "box": recorded["fvm_solver"]["fvm_domain"],
    }


def figure_size(height_cm):
    return FIGURE_WIDTH, height_cm * CM


def save(fig, name, fmt, dpi=FIGURE_DPI):
    return _THEME.export_figure(fig, FIGURES / name, figure_format=fmt, dpi=dpi, close=False)[0]


def comparison_info():
    return read_json(COMPARISON / "comparison_info.json")


def comparison_frame(time):
    frames = comparison_info()["frames"]
    match = match_saved_times([time], [row["time"] for row in frames])
    return read_npz_arrays(COMPARISON / frames[match.indices[1][0]]["file"])


def line_times(source, name):
    if source != "vpm":
        return np.array([row["time"] for row in comparison_info()["frames"]])
    return np.unique(read_csv_table(_path(source, name, ".csv"))["time"])


def load_line(source, name, time):
    if source != "vpm":
        frame = comparison_frame(time)
        points = frame[f"{name}_points"]
        velocity = frame[f"{name}_{source}"]
        return {
            "time": time,
            **{f"position_{axis}": points[:, index] for index, axis in enumerate("xyz")},
            **{f"velocity_{axis}": velocity[:, index] for index, axis in enumerate("xyz")},
        }
    table = read_csv_table(_path(source, name, ".csv"))
    times = np.unique(table["time"])
    match = match_saved_times([time], times)
    picked = times[match.indices[1][0]]
    frame = {key: values[table["time"] == picked] for key, values in table.items()}
    order = np.argsort(frame["position_x"])
    return {**{key: values[order] for key, values in frame.items()}, "time": picked}


def slice_frames(source, name="slice_z0"):
    return read_pvd_frames(_path(source, name, ".pvd"))


def slice_times(source, name="slice_z0"):
    if source != "vpm":
        return np.array([row["time"] for row in comparison_info()["frames"]])
    return np.array([time for time, _ in slice_frames(source, name)])


def load_slice(source, time, name="slice_z0"):
    if source != "vpm":
        frame = comparison_frame(time)
        velocity = frame[f"slice_{source}"]
        return {
            "time": time,
            "x": frame["slice_x"],
            "y": frame["slice_y"],
            "velocity": velocity,
            "valid": np.all(np.isfinite(velocity), axis=-1),
            **{f"velocity_{axis}": velocity[..., index] for index, axis in enumerate("xyz")},
        }
    frames = slice_frames(source, name)
    match = match_saved_times([time], [value for value, _ in frames])
    return {"time": time, **read_surface_frame(frames[match.indices[1][0]][1])}


def load_forces(source):
    return read_history_table(_source_directory(source) / "forces_history.csv")


def common_times(*series):
    matched = match_saved_times(*(np.unique(np.asarray(values, dtype=float)) for values in series))
    return np.asarray([time for time in matched.times if time > 0 and not same_saved_time(time, 0)])


def _frame_at_time(frames, time):
    match = match_saved_times([time], [value for value, _ in frames])
    return frames[match.indices[1][0]][1]


def prepare_comparison_fields():
    meshes = {
        "reference": _fvm_output_files(reference_run().solution)[1],
        "fvm": _fvm_output_files(SOLUTION)[1],
    }
    volumes = {
        "reference": read_pvd_frames(_fvm_output_files(reference_run().solution)[0]),
        "fvm": read_pvd_frames(_fvm_output_files(SOLUTION)[0]),
    }
    slices = slice_frames("fvm")
    times = common_times(
        *([time for time, _ in frames] for frames in volumes.values()),
        [time for time, _ in slices],
        slice_times("vpm"),
        line_times("vpm", "centreline"),
        line_times("vpm", "offaxis_y075"),
    )
    samplers = {source: NativeVelocity(path, k=12) for source, path in meshes.items()}
    rows = []
    for time in times:
        grid = pv.read(_frame_at_time(slices, time))
        ni, nj, _ = grid.dimensions
        shape = (nj, ni)
        points = np.asarray(grid.points, dtype=float)
        queries = {"slice": points}
        for name in ("centreline", "offaxis_y075"):
            line = load_line("vpm", name, time)
            queries[name] = np.column_stack([line[f"position_{axis}"] for axis in "xyz"])
        arrays = {"slice_x": points[:, 0].reshape(shape), "slice_y": points[:, 1].reshape(shape)}
        for source, sampler in samplers.items():
            fields = sampler.sample(
                _frame_at_time(volumes[source], time), queries, mask=fluid_points
            )
            arrays[f"slice_{source}"] = fields["slice"].reshape(*shape, 3)
            for name in ("centreline", "offaxis_y075"):
                arrays[f"{name}_points"] = queries[name]
                arrays[f"{name}_{source}"] = fields[name]
        filename = f"fields_t{time:.9f}.npz"
        write_npz_arrays(COMPARISON / filename, **arrays)
        rows.append({"time": float(time), "file": filename})
    write_json(COMPARISON / "comparison_info.json", {"method": PREPARATION_METHOD, "frames": rows})
    return len(rows)


def main():
    print(f"Prepared {prepare_comparison_fields()} comparison states.")


if __name__ == "__main__":
    main()
