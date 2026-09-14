"""Sample saved FVM fields with one 3D reconstruction, without rerunning.

Only exactly coincident states are compared. Derived data live in the coupled
samples/comparison directory; original fields and samples are read-only.
"""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from . import _plotutil as util

METHOD = "native-centres-affine-k12-v1"


class ComparisonNotReady(RuntimeError):
    """The runs have not saved the common states needed for comparison."""


def frames(pvd: Path) -> list[tuple[float, Path]]:
    if not pvd.is_file():
        raise FileNotFoundError(pvd)
    result = sorted(
        (float(item.attrib["timestep"]), pvd.parent / item.attrib["file"])
        for item in ET.parse(pvd).iter("DataSet")
    )
    if any(b[0] <= a[0] for a, b in zip(result, result[1:])):
        raise ValueError(f"Duplicate states in {pvd}")
    return result


def at_time(items: list[tuple[float, Path]], time: float) -> Path | None:
    return next((p for t, p in items if np.isclose(t, time, rtol=0, atol=util.TIME_ATOL)), None)


def file_stamp(path: Path) -> list:
    """Include every MPI piece, so rerunning a case invalidates derived data."""
    paths = [path]
    if path.suffix == ".pvtu":
        paths += [path.parent / item.attrib["Source"] for item in ET.parse(path).iter("Piece")]
    return [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in paths]


def fluid_points(points: np.ndarray) -> np.ndarray:
    # The cube has side D=1. Exclude its closed surface, not an arbitrary halo.
    return ~np.all(np.abs(points) <= 0.5 + 1e-12, axis=-1)


class NativeVelocity:
    """Use native volume centroids and the existing FVM affine point probe.

    VTK locates containing fluid cells only; it does not interpolate fields.
    This prevents nearest-neighbour extrapolation outside the saved domain.
    """

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
    expected = util.CASE_DIR / "reference_flow" / "samples" / "fine"
    if util.REFERENCE_SAMPLES.resolve() != expected.resolve():
        raise ValueError(f"Cube comparisons require the fine reference: {expected}")
    info = json.loads((expected / "grid_run.json").read_text())
    if info["case"] != "fine" or not np.isclose(info["cell_size"], 0.06, rtol=0, atol=1e-12):
        raise ValueError("Expected the registered fine reference with dx=0.06")
    return info


def prepare() -> int:
    info = validate_reference()
    reference_solution = util.CASE_DIR / "reference_flow" / "solution" / "fine"
    ref_config = reference_solution / "fvm_metadata.json"
    fvm_config = util.SOLUTION / "fvm_metadata.json"
    config = json.loads(fvm_config.read_text())
    fvm_frames = frames(util.SOLUTION / f"{config['case_name']}.pvd")
    reference_frames = frames(reference_solution / "fine.pvd")
    vpm_frames = frames(util.SAMPLES / "vpm_slice_z0.pvd")
    # FVM coordinates define the nominal slice lattice. VPM stores equivalent
    # coordinates with f32 roundoff; the plotter explicitly checks this.
    fvm_slices = frames(util.SAMPLES / "fvm_slice_z0.pvd")
    line_names = ("centreline", "offaxis_y075")
    times = util.common_times(
        np.array([t for t, _ in reference_frames]),
        np.array([t for t, _ in fvm_frames]),
        np.array([t for t, _ in vpm_frames]),
        np.array([t for t, _ in fvm_slices]),
        util.line_times("vpm", "centreline"),
        util.line_times("vpm", "offaxis_y075"),
    )
    if not len(times):
        raise ComparisonNotReady("No common saved Reference FVM, Coupled FVM and VPM state yet.")
    util.COMPARISON.mkdir(parents=True, exist_ok=True)
    manifest_path = util.COMPARISON / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"frames": []}
    meshes = {"reference": reference_solution / "mesh.npz", "fvm": util.SOLUTION / "mesh.npz"}
    fixed = {
        "method": METHOD,
        "meshes": {name: file_stamp(path) for name, path in meshes.items()},
        "configurations": [file_stamp(ref_config), file_stamp(fvm_config)],
    }
    samplers, rows = {}, []
    count = 0
    import pyvista as pv

    for time in times:
        raw = {"reference": at_time(reference_frames, time), "fvm": at_time(fvm_frames, time)}
        slice_path = at_time(fvm_slices, time)
        grid = pv.read(slice_path)
        ni, nj, _ = grid.dimensions
        shape = (nj, ni)
        points = np.asarray(grid.points, dtype=float)
        queries = {"slice": points}
        for name in line_names:
            frame = util.load_line("vpm", name, time)
            queries[name] = np.column_stack([frame[f"position_{axis}"] for axis in "xyz"])
        signature = {
            **fixed,
            "fields": {name: file_stamp(path) for name, path in raw.items()},
            "slice": file_stamp(slice_path),
            "points": {
                name: hashlib.sha256(values.tobytes()).hexdigest()
                for name, values in queries.items()
            },
        }
        destination = util.COMPARISON / f"fields_t{time:.9f}.npz"
        entry = {"time": float(time), "file": destination.name, "inputs": signature}
        cached = any(row == entry for row in old["frames"]) and destination.is_file()
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
    manifest = {"method": METHOD, "reference": info, "frames": rows}
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    return count


def main() -> int:
    try:
        count = prepare()
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
        # allplot.sh stops here, before validators or plotters consume stale data.
        return 2
    print(f"Prepared {count} matched comparison state(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
