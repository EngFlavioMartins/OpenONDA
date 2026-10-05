"""Read-only reconstruction of missing cylinder FVM velocity profiles.

The original cylinder run did not record transverse FVM line probes. Native
field outputs supply those profiles at their own saved clocks, using the same
affine, 12-centre reconstruction as the online FVM samplers. No solver is
constructed, no field is extrapolated outside its domain, and no time state is
interpolated. Derived samples are cached separately from the solver outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from openonda.saved_times import match_saved_times, read_pvd_times

from . import postprocess as data

RECONSTRUCTION = "native-centres-affine-k12-v1"


@dataclass(frozen=True)
class ProfileGeometry:
    diameter: float
    speed: float
    fvm_box: dict[str, float]
    transfer_box: dict[str, float]


def profile_geometry() -> ProfileGeometry:
    """Read plotting scales and bounds from the saved run, never current setup."""
    solution = data.CASE_DIR / "solution"
    metadata = json.loads((solution / "run_metadata.json").read_text())
    configuration = json.loads((solution / "fvm_metadata.json").read_text())["configuration"]
    samplers = [
        sampler for sampler in configuration["samplers"]
        if sampler.get("type") == "ForceSampler" and "cylinder" in sampler["patch_names"]
    ]
    if len(samplers) != 1:
        raise ValueError("Saved cylinder run must identify one force normalization")
    diameter, speed = (float(samplers[0][key]) for key in ("reference_length", "reference_velocity"))
    if not np.isfinite([diameter, speed]).all() or min(diameter, speed) <= 0:
        raise ValueError("Saved cylinder velocity and length scales must be positive and finite")
    boxes = [
        {key: float(value) for key, value in box.items()}
        for box in (
            metadata["fvm_solver"]["fvm_domain"],
            metadata["coupler"]["transfer_region_bounds"],
        )
    ]
    for box in boxes:
        if set(box) != {f"{axis}{side}" for axis in "xyz" for side in ("min", "max")}:
            raise ValueError("Saved cylinder domains require all six bounds")
        if not np.isfinite(list(box.values())).all() or any(
            box[f"{axis}min"] >= box[f"{axis}max"] for axis in "xyz"
        ):
            raise ValueError("Saved cylinder domain bounds must be finite and ordered")
    return ProfileGeometry(diameter, speed, *boxes)


def native_fvm_frames() -> tuple[list[tuple[float, Path]], Path]:
    """Identify the complete saved native mesh/field pair for this run."""
    solution = data.CASE_DIR / "solution"
    metadata = json.loads((solution / "fvm_metadata.json").read_text())
    for pvd, mesh in (
        (solution / "fvm.pvd", solution / "fvm/mesh.npz"),
        (solution / f"{metadata['case_name']}.pvd", solution / "mesh.npz"),
    ):
        if pvd.is_file() and mesh.is_file():
            read_pvd_times(pvd)
            return sorted(
                (float(item.attrib["timestep"]), pvd.parent / item.attrib["file"])
                for item in ET.parse(pvd).iter("DataSet")
            ), mesh
    raise FileNotFoundError("Missing native FVM mesh/field outputs for transverse profiles")


def _file_stamp(path: Path) -> list:
    paths = [path]
    if path.suffix == ".pvtu":
        paths += [path.parent / item.attrib["Source"] for item in ET.parse(path).iter("Piece")]
    return [[str(item.resolve()), item.stat().st_size, item.stat().st_mtime_ns] for item in paths]


def ordered_velocity(fields, cell_count: int, source: Path) -> np.ndarray:
    """Require each native cell exactly once, independent of MPI piece order."""
    velocity = np.asarray(fields["velocity"], dtype=float)
    keep = np.asarray(fields.get("vtkGhostType", np.zeros(len(velocity)))) == 0
    ids = np.asarray(fields.get("global_cell_id", np.arange(len(velocity))), dtype=int)[keep]
    if len(ids) != cell_count or not np.array_equal(np.sort(ids), np.arange(cell_count)):
        raise ValueError(f"Snapshot does not cover its native mesh exactly once: {source}")
    ordered = np.empty((cell_count, 3), dtype=float)
    ordered[ids] = velocity[keep]
    if not np.isfinite(ordered).all():
        raise ValueError(f"Non-finite saved FVM velocity: {source}")
    return ordered


class NativeProfiles:
    def __init__(self, mesh: Path):
        self.mesh = mesh
        self.centres = None
        self.probes = {}

    def sample(self, source: Path, points: np.ndarray) -> tuple[np.ndarray, Path]:
        """Cache one immutable field/profile pair, with full MPI piece stamps."""
        identity = {
            "method": RECONSTRUCTION,
            "mesh": _file_stamp(self.mesh),
            "field": _file_stamp(source),
            "points_sha256": hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
        }
        cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = data.AUXILIARY / "velocity_profile_cache" / f"{cache_key}.npz"
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as saved:
                values = np.array(saved["velocity"], copy=True)
                if str(saved["identity"]) != json.dumps(identity, sort_keys=True):
                    raise ValueError(f"Saved profile cache provenance changed: {cache}")
            if values.shape != points.shape or not np.isfinite(values).all():
                raise ValueError(f"Malformed saved profile cache: {cache}")
            return values, cache

        import pyvista as pv
        from source.solvers.fvm.io.mesh_storage import load_native_mesh
        from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
        from source.solvers.fvm.sampling.fields import _PointProbe

        if self.centres is None:
            self.centres = compute_mesh_geometry(
                load_native_mesh(self.mesh), compute_lsq=False,
            )["cell_centre"]
        grid = pv.read(source)
        ordered = ordered_velocity(grid.cell_data, len(self.centres), source)
        if np.any(grid.find_containing_cell(points) < 0):
            raise ValueError(f"Profile queries leave the native FVM fluid mesh: {source}")
        key = identity["points_sha256"]
        if key not in self.probes:
            self.probes[key] = _PointProbe(points, k=12, reconstruction="affine")
        values = self.probes[key]._interpolate(ordered, self.centres)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite reconstructed FVM profile: {source}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        staging = cache.with_name(f".{cache.stem}-{os.getpid()}.npz")
        np.savez_compressed(staging, velocity=values, identity=json.dumps(identity, sort_keys=True))
        staging.replace(cache)
        return values, cache


def coincident_velocity_profiles(profiles, geometry: ProfileGeometry):
    """Yield exact three-source states wherever the FVM physically exists."""
    paths = tuple(path for _, *paths in profiles for path in paths if path is not None)
    native_columns = [
        index for index, (x, _, _, fvm) in enumerate(profiles)
        if fvm is None and geometry.fvm_box["xmin"] <= x <= geometry.fvm_box["xmax"]
    ]
    states = list(data.coincident_profiles(paths))
    if not native_columns:
        for time, samples in states:
            yield time, profiles, samples, {}
        return
    fields, mesh = native_fvm_frames()
    common = match_saved_times([time for time, _ in states], [time for time, _ in fields])
    if not common.times:
        raise ValueError("Velocity profiles have no coincident native FVM field state")
    native = NativeProfiles(mesh)
    for state_index, field_index in zip(*common.indices, strict=True):
        time, saved_samples = states[state_index]
        native_time, field_path = fields[field_index]
        updated = list(profiles)
        samples, provenance = dict(saved_samples), {}
        for column in native_columns:
            x, reference, vpm, _ = profiles[column]
            positions = samples[reference][["position_x", "position_y", "position_z"]].to_numpy(float)
            if not np.allclose(positions[:, 0], x, rtol=0, atol=1e-10):
                raise ValueError(f"Transverse profile coordinates disagree with its filename: {reference}")
            keep = np.ones(len(positions), dtype=bool)
            for axis, name in enumerate("xyz"):
                keep &= (positions[:, axis] >= geometry.fvm_box[f"{name}min"] - 1e-12)
                keep &= (positions[:, axis] <= geometry.fvm_box[f"{name}max"] + 1e-12)
            positions = positions[keep]
            if len(positions) < 2:
                raise ValueError(f"Insufficient profile points inside FVM domain at x={x:g}")
            velocity, cache = native.sample(field_path, positions)
            frame = pd.DataFrame(positions, columns=[f"position_{axis}" for axis in "xyz"])
            frame["time"] = native_time
            frame[list(data.VELOCITY_COLUMNS)] = velocity
            samples[cache] = frame
            updated[column] = (x, reference, vpm, cache)
            provenance[f"x{x:g}"] = {
                "method": RECONSTRUCTION,
                "native_time": native_time,
                "native_field": str(field_path.relative_to(data.CASE_DIR)),
                "cache": str(cache.relative_to(data.CASE_DIR)),
                "point_count": len(positions),
                "sampled_y_interval": [float(positions[:, 1].min()), float(positions[:, 1].max())],
            }
        yield time, updated, samples, provenance
