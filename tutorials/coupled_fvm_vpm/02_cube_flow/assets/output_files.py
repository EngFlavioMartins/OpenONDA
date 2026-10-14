"""Read and write the solver output files used by this tutorial's figures."""

import csv
import json
from pathlib import Path

from defusedxml import ElementTree
import numpy as np
import pyvista as pv


def read_json(path):
    """Read one JSON object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """Write one JSON object."""
    write_text(path, json.dumps(data, indent=2) + "\n")


def read_json_lines(path):
    """Read one JSON record per line, skipping an unfinished final line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.strip() and line.endswith("\n")]


def read_csv_table(path):
    """Read named CSV columns."""
    data = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    return {name: data[name] for name in data.dtype.names}


def read_history_table(path):
    """Read an accepted history with one row per physical time."""
    return read_csv_table(path)


def read_pvd_frames(path):
    """Return the ``(time, frame path)`` pairs of a VTK collection in time order."""
    path = Path(path)
    return sorted(
        (float(item.attrib["timestep"]), path.parent / item.attrib["file"])
        for item in ElementTree.parse(path).iter("DataSet")
    )


def read_npz_arrays(path):
    """Read every array of a NumPy archive."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def write_text(path, content, encoding="utf-8"):
    """Write a text file, creating its directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding=encoding)


def write_csv_table(path, rows, *, columns):
    """Write a named CSV table, creating its directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def write_npz_arrays(path, **arrays):
    """Write a compressed NumPy archive, creating its directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def matched_surface_vectors(source, target):
    """Return sampled velocity with invalid points masked; both use one grid."""
    values = np.array(source["velocity"], dtype=float, copy=True)
    values[~source["valid"]] = np.nan
    return values


def read_surface_frame(path):
    """Read a structured surface-sampler frame as two-dimensional arrays."""
    grid = pv.read(path)
    ni, nj, _ = grid.dimensions
    shape = (nj, ni)
    points = np.asarray(grid.points, dtype=float)
    valid = np.asarray(grid.point_data.get("vtkValidPointMask", np.ones(grid.n_points)), bool)

    def point_field(name, components):
        values = np.asarray(grid.point_data[name], dtype=float)
        return values.reshape(*shape, components) if components > 1 else values.reshape(shape)

    velocity = point_field("velocity", 3)
    data = {
        "x": points[:, 0].reshape(shape),
        "y": points[:, 1].reshape(shape),
        "z": points[:, 2].reshape(shape),
        "velocity": velocity,
        "valid": valid.reshape(shape),
        **{f"velocity_{axis}": velocity[..., index] for index, axis in enumerate("xyz")},
    }
    for name in ("vorticity", "velocity_standard_error", "vorticity_standard_error"):
        if name in grid.point_data:
            vector = point_field(name, 3)
            data[name] = vector
            data.update({f"{name}_{axis}": vector[..., index] for index, axis in enumerate("xyz")})
    for name in ("velocity_gradient_yx", "velocity_gradient_yx_standard_error"):
        if name in grid.point_data:
            data[name] = point_field(name, 1)
    for name in ("ensemble_size", "confidence_multiplier"):
        if name in grid.field_data:
            data[name] = np.asarray(grid.field_data[name])[0].item()
    return data


class NativeVelocity:
    """Sample saved FVM velocity with affine probes about the native cell centres."""

    def __init__(self, mesh_path, *, k):
        from openonda.fvm.io.mesh_storage import load_native_mesh
        from openonda.fvm.mesh.geometry import compute_mesh_geometry

        mesh = load_native_mesh(mesh_path)
        self.centres = compute_mesh_geometry(mesh, compute_lsq=False)["cell_centre"]
        self.k = k

    def sample(self, source, queries, *, mask=None):
        from openonda.fvm.sampling import interpolate_cell_field

        grid = pv.read(source)
        velocity = np.asarray(grid.cell_data["velocity"], dtype=float)
        keep = np.asarray(grid.cell_data.get("vtkGhostType", np.zeros(len(velocity)))) == 0
        ids = np.asarray(grid.cell_data.get("global_cell_id", np.arange(len(velocity))), int)
        ordered = np.empty_like(velocity[keep])
        ordered[ids[keep]] = velocity[keep]
        local_grid = grid.extract_cells(keep)
        result = {}
        for name, points in queries.items():
            inside = local_grid.find_containing_cell(points) >= 0
            if mask is not None:
                inside &= mask(points)
            values = interpolate_cell_field(
                ordered, self.centres, points, n_neighbours=self.k, reconstruction="affine"
            )
            values[~inside] = np.nan
            result[name] = values
        return result
