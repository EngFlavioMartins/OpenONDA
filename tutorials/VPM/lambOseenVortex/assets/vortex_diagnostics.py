#!/usr/bin/env python3
"""Vortex diagnostics for the Lamb-Oseen tutorial: paths, physics constants,
sampled-field readers, the field-based center/core-radius extraction
algorithm, and RWM ensemble averaging.

The center/core-radius/separation/merged status for all three physics cases
(single vortex, dipole, merging) are derived from the *sampled velocity
field* itself (the ``*_zq_*.vts`` planes written by the ``SurfaceSampler``
at z = +L/4) — one consistent method, independent of the viscous diffusion
scheme and the physics case:

  * vortex centres   — vorticity-weighted centroids of the signed (dipole) or
    twin-peak (merging) regions, or the single peak (lone vortex);
  * core radius a_c  — radius where the azimuthally-averaged tangential
    velocity |u_theta(r)| peaks, divided by BETA_RMAX (the Lamb--Oseen
    u_theta peak sits at r = BETA_RMAX * a_c);
  * separation and orbital angle (dipole/merging pair).

Run directly to extract ``<case>/field_diagnostics.csv`` next to the sampled
planes for every case under ``samples/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# -- Directory layout --------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent  # …/assets/
SCRIPT_DIR = ASSETS_DIR.parent  # …/lambOseenVortex/
FIGURES_DIR = SCRIPT_DIR / "figures"
SAMPLES_DIR = SCRIPT_DIR / "samples"
REF_DIR = ASSETS_DIR / "references"

# -- Tutorial constants -------------------------------------------------------
SCHEMES = ("cs", "rwm", "dvh", "gbd")

BETA_RMAX = 1.12
GAMMA = 1.0
REYNOLDS_NUMBER = 530.0
CORE_RADIUS = 0.125
SEPARATION = 1.0
COLUMN_LENGTH = 50.0 * CORE_RADIUS  # mirrors lambossen_setup.py::COLUMN_LENGTH
FIELD_SPACING = 0.30 * CORE_RADIUS  # mirrors lambossen_setup.py::FIELD_SPACING
TOTAL_TIME = 30.0  # fallback reference time [s] when no run data is available

VTS_STEP_RE = re.compile(r"_(\d+)\.vts$")

FIELD_CSV_COLUMNS = [
    "flow_time",
    "time_step",
    "center0_x",
    "center0_y",
    "center1_x",
    "center1_y",
    "separation",
    "a_c0",
    "a_c1",
    "a_c_mean",
    "angle_rad",
    "merged",
]


# =============================================================
# Run-metadata / sampled-field readers
# =============================================================


def read_flow_time(csv_path: Path) -> float | None:
    """Extract ``flow_time`` from the first comment line of a sampled CSV."""
    with open(csv_path) as handle:
        first_line = handle.readline().strip()
    if first_line.startswith("# flow_time="):
        return float(first_line.split("=", 1)[1])
    return None


def read_run_metadata(samples_dir: Path, prefix: str = "vortex") -> dict:
    """Load the physical constants stored with the sampled results.

    Plotting must not depend on a dense particle backup: those files are sparse
    restart checkpoints, while ``samples/<case>/run_metadata.json`` is written
    alongside the data used by each figure.
    """
    for scheme in SCHEMES:
        path = samples_dir / f"{prefix}_{scheme}" / "run_metadata.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def resolve_runtime_physics(
    samples_dir: Path,
    gamma: float,
    fallback_nu: float,
    b0: float,
    a0_over_b0: float,
) -> dict[str, float]:
    """Return the physical constants for the analytic reference.

    ``nu`` and the core radius come from ``samples/<case>/run_metadata.json``.
    If a run has no metadata, the tutorial constants provide the reference.
    The analytic reference is never inferred from the schemes' own output.
    """
    metadata = read_run_metadata(samples_dir)
    ac0 = float(metadata.get("core_radius", a0_over_b0 * b0))
    nu = float(metadata.get("viscosity", fallback_nu))
    if nu <= 0.0:
        nu = fallback_nu
    sigma0 = ac0 / BETA_RMAX
    return {"nu": nu, "t0": sigma0**2 / (4.0 * nu), "ac0": ac0}


def pvd_time_map(samples_dir: Path, prefix: str, scheme: str) -> dict[int, float]:
    """Read the surface-sample PVD to get a step → physical-time mapping.

    This keeps sampled fields tied to physical time without reading checkpoints.
    """
    import xml.etree.ElementTree as ET

    pvd = samples_dir / f"{prefix}_{scheme}" / f"{prefix}_{scheme}_zq.pvd"
    if not pvd.exists():
        return {}
    tree = ET.parse(pvd)  # nosec B314
    result: dict[int, float] = {}
    for ds in tree.getroot().iter("DataSet"):
        fname = ds.attrib.get("file", "")
        m = VTS_STEP_RE.search(fname)
        if m:
            result[int(m.group(1))] = float(ds.attrib.get("timestep", 0.0))
    return result


def read_surface_field(path: Path) -> dict:
    """Read x/y/velocity/vorticity 2D arrays (Fortran order) from a sampled
    z=L/4 SurfaceSampler plane, matching how ``SurfaceSampler.save_vtp`` writes it."""
    import pyvista as pv

    grid = pv.read(str(path))
    points = np.asarray(grid.points, dtype=np.float64)
    velocity = np.asarray(grid.point_data["Velocity"], dtype=np.float64)
    vorticity = np.asarray(grid.point_data["Vorticity"], dtype=np.float64)

    ni = len(np.unique(np.round(points[:, 0], 9)))
    nj = len(np.unique(np.round(points[:, 1], 9)))
    if ni * nj != points.shape[0]:
        raise ValueError(f"{path.name}: expected {ni}x{nj} structured grid")
    shape = (ni, nj)

    def _f(flat):
        return flat.reshape(shape, order="F")

    return {
        "x": _f(points[:, 0]),
        "y": _f(points[:, 1]),
        "Ux": _f(velocity[:, 0]),
        "Uy": _f(velocity[:, 1]),
        "omega_z": _f(vorticity[:, 2]),
    }


# =============================================================
# Vortex properties from the 2D field
# =============================================================


def _weighted_centroid(x: np.ndarray, y: np.ndarray, weight: np.ndarray):
    total = float(weight.sum())
    if total <= np.finfo(float).tiny:
        return np.array([np.nan, np.nan])
    return np.array([float(np.dot(weight, x)) / total, float(np.dot(weight, y)) / total])


def _initial_centers(field: dict, physics: str) -> list[np.ndarray]:
    """Robust seeds for the vortex centre(s) from the omega_z field."""
    x, y, wz = field["x"], field["y"], field["omega_z"]
    w = np.abs(wz)
    mask = w > 0.05 * float(w.max())
    xs, ys = x[mask], y[mask]
    ws = w[mask]
    wzs = wz[mask]

    if physics == "dipole":
        positive = wzs > 0.0
        if not positive.any() or positive.all():
            return [np.array([np.nan, np.nan]), np.array([np.nan, np.nan])]
        c0 = _weighted_centroid(xs[positive], ys[positive], ws[positive])
        c1 = _weighted_centroid(xs[~positive], ys[~positive], ws[~positive])
        return [c0, c1]

    if physics == "vortex":
        # No second core exists, by construction — a peak search would
        # occasionally mistake numerical noise far from the core for a
        # second vortex, so don't search for one.
        flat = w.ravel()
        i0 = int(np.argmax(flat))
        return [np.array([float(x.ravel()[i0]), float(y.ravel()[i0])])]

    # merging: two co-rotating positive cores -> peak search.
    flat = w.ravel()
    i0 = int(np.argmax(flat))
    c0 = np.array([float(x.ravel()[i0]), float(y.ravel()[i0])])
    d2 = (x - c0[0]) ** 2 + (y - c0[1]) ** 2
    second = mask & (np.sqrt(d2) > 0.55 * SEPARATION)
    if np.count_nonzero(second) < 10:
        return [c0]
    i1 = int(np.argmax(np.where(second, w, -1.0)))
    c1 = np.array([float(x.ravel()[i1]), float(y.ravel()[i1])])
    return [c0, c1]


def _refine_centers(field: dict, centers: list[np.ndarray]) -> list[np.ndarray]:
    """Iterated vorticity-weighted centroid (Voronoi cell) refinement."""
    x, y, wz = field["x"], field["y"], field["omega_z"]
    w = np.abs(wz)
    mask = w > 0.05 * float(w.max())
    cells = [c.copy() for c in centers]
    for _ in range(8):
        new_cells = []
        for index, cell in enumerate(cells):
            d2 = (x - cell[0]) ** 2 + (y - cell[1]) ** 2
            if len(cells) == 1:
                owned = mask
            elif index == 0:
                owned = mask & (d2 <= (x - cells[1][0]) ** 2 + (y - cells[1][1]) ** 2)
            else:
                owned = mask & (d2 < (x - cells[0][0]) ** 2 + (y - cells[0][1]) ** 2)
            new_cells.append(
                _weighted_centroid(x[owned], y[owned], w[owned])
                if np.count_nonzero(owned) > 0
                else cell
            )
        if all(np.allclose(a, b) for a, b in zip(new_cells, cells, strict=True)):
            return new_cells
        cells = new_cells
    return cells


def _core_radius_utheta(
    field: dict,
    center: np.ndarray,
    r_max: float,
    bin_width: float = FIELD_SPACING,
) -> float:
    """Core radius a_c from the azimuthally-averaged tangential velocity peak."""
    if not np.isfinite(center).all():
        return float("nan")
    x, y = field["x"], field["y"]
    dx = x - center[0]
    dy = y - center[1]
    r = np.sqrt(dx * dx + dy * dy)
    keep = (r > 0.5 * bin_width) & (r < r_max)
    if np.count_nonzero(keep) < 20:
        return float("nan")

    e_theta_x = -dy / np.where(r > 0, r, 1.0)
    e_theta_y = dx / np.where(r > 0, r, 1.0)
    u_theta = np.abs(field["Ux"] * e_theta_x + field["Uy"] * e_theta_y)

    edges = np.arange(0.5 * bin_width, r_max + bin_width, bin_width)
    bin_index = np.clip(np.searchsorted(edges, r[keep], side="right") - 1, 0, None)
    radii, magnitudes = [], []
    for b in np.unique(bin_index):
        selected = bin_index == b
        if np.count_nonzero(selected) < 3:
            continue
        radii.append(0.5 * (edges[b] + edges[min(b + 1, len(edges) - 1)]))
        magnitudes.append(float(u_theta[keep][selected].mean()))
    radii = np.asarray(radii)
    magnitudes = np.asarray(magnitudes)
    if radii.size < 3:
        return float("nan")

    i = int(np.argmax(magnitudes))
    if 0 < i < radii.size - 1:
        u1, u2, u3 = magnitudes[i - 1 : i + 2]
        r1, r2, r3 = radii[i - 1 : i + 2]
        denominator = u1 - 2.0 * u2 + u3
        if abs(denominator) > 1e-12:
            r_peak = r2 + 0.5 * (u1 - u3) / denominator * (r2 - r1)
        else:
            r_peak = r2
    else:
        r_peak = radii[i]
    return float(r_peak / BETA_RMAX)


def _search_radius(physics: str, separation: float) -> float:
    """u_theta-peak search window. Dipole/merging scale off the nominal pair
    separation; a lone vortex has no separation to scale off, so use a
    generous multiple of the *expected* final core radius instead (the core
    can diffuse well past a fixed fraction of SEPARATION by t=TOTAL_TIME)."""
    if physics == "vortex":
        expected_final_ac = np.sqrt(
            CORE_RADIUS**2 + 4.0 * BETA_RMAX**2 * (GAMMA / REYNOLDS_NUMBER) * TOTAL_TIME
        )
        return 2.0 * BETA_RMAX * expected_final_ac
    return min(0.5, 0.45 * (separation if np.isfinite(separation) else SEPARATION))


def _diagnostics_row(field: dict, physics: str) -> list:
    centers = _refine_centers(field, _initial_centers(field, physics))
    c0 = centers[0] if len(centers) >= 1 else np.array([np.nan, np.nan])
    c1 = centers[1] if len(centers) >= 2 else np.array([np.nan, np.nan])

    separation = (
        float(np.linalg.norm(c0 - c1))
        if np.isfinite(c0).all() and np.isfinite(c1).all()
        else float("nan")
    )
    merged = not np.isfinite(separation) or not (
        0.05 * SEPARATION <= separation <= 1.15 * SEPARATION
    )

    r_max = _search_radius(physics, separation)
    a_c0 = _core_radius_utheta(field, c0, r_max) if np.isfinite(c0).all() else float("nan")
    a_c1 = _core_radius_utheta(field, c1, r_max) if np.isfinite(c1).all() else float("nan")

    a_c_mean = (
        float(np.mean([a for a in (a_c0, a_c1) if np.isfinite(a)]))
        if any(np.isfinite(a) for a in (a_c0, a_c1))
        else float("nan")
    )

    angle = float("nan")
    if np.isfinite(c0).all() and np.isfinite(c1).all():
        midpoint = 0.5 * (c0 + c1)
        angle = float(np.arctan2(c0[1] - midpoint[1], c0[0] - midpoint[0]))

    return [
        field["time"],
        field["step"],
        c0[0],
        c0[1],
        c1[0],
        c1[1],
        separation,
        a_c0,
        a_c1,
        a_c_mean,
        angle,
        merged,
    ]


def _find_cases(samples_dir: Path) -> dict[str, list[Path]]:
    cases: dict[str, list[Path]] = {}
    for case_dir in sorted(samples_dir.glob("*/")):
        if not case_dir.is_dir():
            continue
        vts = sorted(case_dir.glob("*_zq_*.vts"))
        if vts:
            cases[case_dir.name] = vts
    return cases


def extract_field_diagnostics(samples_dir: Path, case: str | None = None) -> None:
    """Write ``<case>/field_diagnostics.csv`` for every sampled case (or just ``case``)."""
    samples_dir = Path(samples_dir)
    if not samples_dir.is_dir():
        print(f"  [field] no samples directory: {samples_dir}")
        return

    cases = _find_cases(samples_dir)
    if case is not None:
        cases = {name: vts for name, vts in cases.items() if name == case}
    if not cases:
        print(f"  [field] no *_zq_*.vts planes under {samples_dir}")
        return

    for case_name, vts in sorted(cases.items()):
        physics, scheme = case_name.split("_", 1)
        timeline = pvd_time_map(samples_dir, physics, scheme)
        rows = []
        for path in vts:
            step = int(VTS_STEP_RE.search(path.name).group(1))
            field = read_surface_field(path)
            field["step"] = step
            field["time"] = timeline.get(step, float("nan"))
            rows.append(_diagnostics_row(field, physics))

        out = samples_dir / case_name / "field_diagnostics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(FIELD_CSV_COLUMNS)
            writer.writerows(rows)
        print(f"  [field] {case_name}: wrote field_diagnostics.csv ({len(rows)} steps)")


# =============================================================
# RWM ensemble averaging
# =============================================================

# The subset of FIELD_CSV_COLUMNS that's a plain numeric average across
# realizations; flow_time/time_step are validated to match, angle_rad needs
# unwrapping, and merged needs a majority vote — all handled separately.
ENSEMBLE_FIELD_COLUMNS = (
    "center0_x",
    "center0_y",
    "center1_x",
    "center1_y",
    "separation",
    "a_c0",
    "a_c1",
    "a_c_mean",
)


def _nanmean(values: np.ndarray) -> np.ndarray:
    """Average columns that may intentionally contain only NaNs."""
    result = np.full(values.shape[1:], np.nan)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    populated = counts > 0
    result[populated] = np.nansum(values, axis=0)[populated] / counts[populated]
    return result


def _average_vts(target: Path, sources: list[Path]) -> None:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    grids = []
    for path in sources:
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        grids.append(reader.GetOutput())

    output = vtk.vtkStructuredGrid()
    output.DeepCopy(grids[0])
    point_data = output.GetPointData()

    for index in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(index)
        if name in {"VelocityMagnitude", "VorticityMagnitude"}:
            continue
        arrays = [vtk_to_numpy(grid.GetPointData().GetArray(name)) for grid in grids]
        vtk_to_numpy(point_data.GetArray(name))[:] = np.mean(arrays, axis=0)

    velocity = vtk_to_numpy(point_data.GetArray("Velocity"))
    vorticity = vtk_to_numpy(point_data.GetArray("Vorticity"))
    vtk_to_numpy(point_data.GetArray("VelocityMagnitude"))[:] = np.linalg.norm(velocity, axis=1)
    vtk_to_numpy(point_data.GetArray("VorticityMagnitude"))[:] = np.linalg.norm(vorticity, axis=1)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(target))
    writer.SetInputData(output)
    if writer.Write() != 1:
        raise OSError(f"Could not write averaged RWM surface to {target}")


def average_final_samples(
    target_dir: Path,
    member_dirs: list[Path],
    *,
    realizations: int,
    particle_replicas: int,
) -> None:
    """Replace the primary final samples with the ensemble mean."""

    case_name = target_dir.name
    surface_name = max(target_dir.glob(f"{case_name}_zq_*.vts")).name
    _average_vts(
        target_dir / surface_name,
        [directory / surface_name for directory in member_dirs],
    )

    metadata_path = target_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "rwm_realizations": realizations,
            "rwm_particle_replicas": particle_replicas,
            "ensemble_averaged_samples": [surface_name],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def average_field_diagnostics(
    target_dir: Path,
    member_dirs: list[Path],
    *,
    realizations: int,
) -> None:
    """Replace a stochastic field-diagnostics history with its independent ensemble mean."""

    frames = [pd.read_csv(directory / "field_diagnostics.csv") for directory in member_dirs]
    reference_steps = frames[0]["time_step"].to_numpy()
    reference_times = frames[0]["flow_time"].to_numpy()
    for frame in frames[1:]:
        if not np.array_equal(frame["time_step"].to_numpy(), reference_steps):
            raise ValueError("RWM ensemble time steps differ")
        if not np.allclose(frame["flow_time"].to_numpy(), reference_times):
            raise ValueError("RWM ensemble flow times differ")

    averaged = frames[0].copy()
    for column in ENSEMBLE_FIELD_COLUMNS:
        values = np.stack([frame[column].to_numpy(float) for frame in frames])
        averaged[column] = _nanmean(values)

    angles = np.stack([np.unwrap(frame["angle_rad"].to_numpy(float)) for frame in frames])
    averaged["angle_rad"] = _nanmean(angles)
    averaged["merged"] = (
        np.mean(
            [frame["merged"].astype(str).str.lower().eq("true") for frame in frames],
            axis=0,
        )
        >= 0.5
    )
    averaged.to_csv(target_dir / "field_diagnostics.csv", index=False)

    integral_frames = [pd.read_csv(directory / "flow_integrals.csv") for directory in member_dirs]
    integral_times = integral_frames[0]["time"].to_numpy()
    if any(
        not np.allclose(frame["time"].to_numpy(), integral_times) for frame in integral_frames[1:]
    ):
        raise ValueError("RWM ensemble flow-integral times differ")
    averaged_integrals = integral_frames[0].copy()
    numeric_columns = averaged_integrals.select_dtypes(include=[np.number]).columns
    averaged_integrals[numeric_columns] = sum(
        frame[numeric_columns] for frame in integral_frames
    ) / len(integral_frames)
    averaged_integrals.to_csv(target_dir / "flow_integrals.csv", index=False)

    metadata_path = target_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "rwm_realizations": realizations,
            "ensemble_averaged_samples": [
                "field_diagnostics.csv",
                "flow_integrals.csv",
            ],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# =============================================================
# CLI entry point
# =============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=SAMPLES_DIR,
        help="directory holding the sampled *_zq_*.vts planes per case",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="only process one case (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extract_field_diagnostics(args.samples_dir, args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
