from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from _plotutil import load_vpm_particles

ROOT = Path(__file__).resolve().parents[1]


def _coupling_time_step_size() -> float:
    metadata = ROOT / "solution" / "run_metadata.json"
    if metadata.exists():
        return float(json.loads(metadata.read_text())["vpm_time_step_size"])
    return 0.05


MATCH_TOL = 0.25 * _coupling_time_step_size()


def _pvd_times(pvd: Path) -> list[tuple[float, Path]]:
    matches = re.finditer(r'timestep="([^"]+)"\s+[^>]*file="([^"]+)"', pvd.read_text())
    return sorted((float(match.group(1)), pvd.parent / match.group(2)) for match in matches)


def plot_frames(
    hybrid_pvd: Path,
    reference_pvd: Path,
) -> list[tuple[float, Path, Path | None]]:
    """Return every hybrid frame with its same-time reference when available."""
    reference = _pvd_times(reference_pvd) if reference_pvd.exists() else []
    frames = []
    for time, hybrid_vtu in _pvd_times(hybrid_pvd):
        if time <= 1e-9:
            continue
        ref_vtu = None
        if reference:
            ref_time, candidate = min(reference, key=lambda item: abs(item[0] - time))
            ref_vtu = candidate if abs(ref_time - time) <= MATCH_TOL else None
        frames.append((time, hybrid_vtu, ref_vtu))
    return frames


def _particle_times() -> list[tuple[float, Path]]:
    import h5py

    snapshots = []
    for path in sorted((ROOT / "solution").glob("vpm_*.h5")):
        with h5py.File(path, "r") as handle:
            snapshots.append((float(handle["solver"].attrs["flow_time"]), path))
    return snapshots


def particles_at(time: float) -> dict:
    snapshots = _particle_times()
    if not snapshots:
        raise FileNotFoundError("no VPM snapshots in solution/")
    snapshot_time, path = min(snapshots, key=lambda item: abs(item[0] - time))
    if abs(snapshot_time - time) > MATCH_TOL:
        raise ValueError(f"no VPM snapshot at t={time:.6g}")
    return load_vpm_particles(path)


def reference_velocity(vtu: Path | None, points: np.ndarray) -> np.ndarray:
    if vtu is None:
        return np.full((len(points), 3), np.nan)
    return sample_vtu(vtu, points)["U"]


def sample_vtu(vtu: Path, points: np.ndarray) -> dict[str, np.ndarray]:
    import pyvista as pv

    mesh = pv.read(vtu)
    cell_fields = {
        name: np.asarray(mesh.cell_data[name])
        for name in ("U", "p", "vorticity")
        if name in mesh.cell_data
    }
    if "U" not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()

    sampled = pv.PolyData(np.ascontiguousarray(points, dtype=np.float64)).sample(
        mesh,
        tolerance=1e-6,
    )
    valid = np.asarray(sampled["vtkValidPointMask"], dtype=bool)

    missing = np.flatnonzero(~valid)
    if missing.size:
        cell_ids, closest = mesh.find_closest_cell(points[missing], return_closest_point=True)
        recovered = missing[np.linalg.norm(points[missing] - closest, axis=1) < 1e-8]
        valid[recovered] = True
    else:
        cell_ids = np.empty(0, dtype=int)
        recovered = np.empty(0, dtype=int)

    fields: dict[str, np.ndarray] = {"valid": valid}
    for name in ("U", "p", "vorticity"):
        if name not in sampled.point_data:
            continue
        values = np.asarray(sampled[name], dtype=np.float64)
        if recovered.size and name in cell_fields:
            recovered_cells = np.asarray(cell_ids)[np.isin(missing, recovered)]
            values[recovered] = cell_fields[name][recovered_cells]
        values[~valid] = np.nan
        fields[name] = values
    return fields
