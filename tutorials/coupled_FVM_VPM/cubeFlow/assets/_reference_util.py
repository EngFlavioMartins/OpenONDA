from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from _plotutil import load_vpm_particles  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "referenceFlow"


def _dt_vpm() -> float:
    """Coupling step, read from the run the coupler actually wrote.

    Hardcoding this drifts silently: it sets MATCH_TOL below, so a stale value
    changes which snapshots are considered simultaneous.  ``run_metadata.json``
    is written by ``FVMVPMCoupler._write_run_metadata`` and is authoritative.
    """
    meta = ROOT / "solution" / "run_metadata.json"
    if meta.is_file():
        try:
            return float(json.loads(meta.read_text())["dt_vpm"])
        except (KeyError, ValueError, OSError):
            pass
    return 0.05  # tutorial default (cubeFlow_setup.DT_VPM)


DT_VPM = _dt_vpm()


def _pvd_times(pvd: Path | None) -> list[tuple[float, Path]]:
    if pvd is None or not pvd.exists():
        return []
    out = []
    for match in re.finditer(r'timestep="([^"]+)"\s+[^>]*file="([^"]+)"', pvd.read_text()):
        out.append((float(match.group(1)), pvd.parent / match.group(2)))
    return sorted(out)


def nearest_vtu(pvd: Path, time: float) -> tuple[float, Path] | None:
    entries = _pvd_times(pvd)
    return min(entries, key=lambda item: abs(item[0] - time)) if entries else None


# Time-match tolerance: the hybrid and reference cases are configured to write
# snapshots at the SAME cadence (0.15 s), so coincident frames match to well
# within this.  Comparisons are only drawn at a shared time (see matched_times).
MATCH_TOL = 0.25 * DT_VPM


def matched_times(hybrid_pvd: Path, reference_pvd: Path) -> list[float]:
    """Times present in BOTH collections to within ``MATCH_TOL`` — the only
    instants at which hybrid-vs-reference field comparisons are physically
    valid (both sampled at the same t)."""
    h = _pvd_times(hybrid_pvd)
    r = _pvd_times(reference_pvd)
    out = []
    for th, _ in h:
        rt = min((t for t, _ in r), key=lambda t: abs(t - th), default=None)
        if rt is not None and abs(rt - th) <= MATCH_TOL:
            out.append(round(0.5 * (th + rt), 6))
    return out


def assert_same_time(t_hybrid: float, t_reference: float) -> float:
    """Guard used by every comparison frame: refuse to plot mismatched times."""
    if abs(t_hybrid - t_reference) > MATCH_TOL:
        raise ValueError(
            f"refusing to compare hybrid t={t_hybrid:.4f} against reference "
            f"t={t_reference:.4f} (|Δt|={abs(t_hybrid - t_reference):.4f} "
            f"> {MATCH_TOL:.4f}); align the write cadences"
        )
    return 0.5 * (t_hybrid + t_reference)


def _bracket(entries: list[tuple[float, Path]], time: float):
    if not entries:
        raise FileNotFoundError("empty time collection")
    lo = max((item for item in entries if item[0] <= time), default=entries[0])
    hi = min((item for item in entries if item[0] >= time), default=entries[-1])
    return lo, hi


def nearest_vpm_h5(time: float, case: Path = ROOT) -> Path | None:
    entries = _h5_times(case)
    return min(entries, key=lambda item: abs(item[0] - time))[1] if entries else None


def _h5_times(case: Path) -> list[tuple[float, Path]]:
    import h5py

    entries = []
    for path in sorted((case / "solution").glob("vpm_vpm_solution_*.h5")):
        try:
            with h5py.File(path, "r") as handle:
                entries.append((float(handle["solver"].attrs["flow_time"]), path))
        except OSError:
            continue
    return sorted(entries)


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
    cloud = pv.PolyData(np.ascontiguousarray(points, dtype=np.float64))
    sampled = cloud.sample(mesh, tolerance=1e-6)
    valid = np.asarray(sampled["vtkValidPointMask"]).astype(bool)
    fallback: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if (~valid).any():
        missing = np.flatnonzero(~valid)
        cell_ids, closest = mesh.find_closest_cell(points[missing], return_closest_point=True)
        inside = np.linalg.norm(points[missing] - closest, axis=1) < 1e-8
        for name, values in cell_fields.items():
            fallback[name] = (missing[inside], values[np.asarray(cell_ids)[inside]])
        valid[missing[inside]] = True
    fields: dict[str, np.ndarray] = {"valid": valid}
    for name in ("U", "p", "vorticity"):
        if name in sampled.point_data:
            value = np.asarray(sampled[name], dtype=np.float64)
            if name in fallback:
                indices, values = fallback[name]
                value[indices] = values
            value[~valid] = np.nan
            fields[name] = value
    return fields


def sample_pvd_at_time(pvd: Path, time: float, points: np.ndarray) -> dict[str, np.ndarray]:
    (t0, p0), (t1, p1) = _bracket(_pvd_times(pvd), time)
    f0 = sample_vtu(p0, points)
    if t1 == t0:
        return f0
    f1 = sample_vtu(p1, points)
    alpha = (time - t0) / (t1 - t0)
    fields: dict[str, np.ndarray] = {}
    for name in f0.keys() & f1.keys():
        if name == "valid":
            fields[name] = f0[name] & f1[name]
        else:
            fields[name] = (1.0 - alpha) * f0[name] + alpha * f1[name]
    return fields


def _constants(case: Path) -> dict:
    meta = json.loads((case / "solution/run_metadata.json").read_text())
    u_inf = np.asarray(meta["physics"]["u_inf"], dtype=float)
    return {
        "U_inf": float(np.linalg.norm(u_inf)) or 1.0,
        "u_inf_vec": u_inf,
        "box": meta["fvm_solver"]["fvm_domain"],
    }


def _particle_states(case: Path, time: float):
    entries = _h5_times(case)
    if not entries:
        return []
    lo, hi = _bracket(entries, time)
    states = [(lo[0], load_vpm_particles(lo[1]))]
    if hi[1] != lo[1]:
        states.append((hi[0], load_vpm_particles(hi[1])))
    return states


def _vpm_velocity(states, time: float, points: np.ndarray, u_inf: np.ndarray) -> np.ndarray:
    from source.coupler.core.helpers.interior_bs import gaussian_bs_velocity

    def evaluate(state):
        return u_inf + gaussian_bs_velocity(
            np.asarray(points, dtype=np.float64),
            np.asarray(state["position"], dtype=np.float64),
            np.asarray(state["circulation"], dtype=np.float64),
            np.asarray(state["radius"], dtype=np.float64),
        )

    value0 = evaluate(states[0][1])
    if len(states) == 1:
        return value0
    value1 = evaluate(states[1][1])
    alpha = (time - states[0][0]) / (states[1][0] - states[0][0])
    return (1.0 - alpha) * value0 + alpha * value1


def hybrid_u_at_time(
    pvd: Path,
    particle_states,
    time: float,
    points: np.ndarray,
    box: dict,
    u_inf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inside = (
        (points[:, 0] >= box["xmin"])
        & (points[:, 0] <= box["xmax"])
        & (points[:, 1] >= box["ymin"])
        & (points[:, 1] <= box["ymax"])
        & (points[:, 2] >= box["zmin"])
        & (points[:, 2] <= box["zmax"])
    )
    velocity = np.full((len(points), 3), np.nan)
    if inside.any():
        velocity[inside] = sample_pvd_at_time(pvd, time, points[inside])["U"]
    if (~inside).any() and particle_states:
        velocity[~inside] = _vpm_velocity(particle_states, time, points[~inside], u_inf)
    return velocity, inside
