#!/usr/bin/env python3
"""Vortex diagnostics for the Lamb-Oseen tutorial: paths, physics constants,
sampled-field readers, the field-based center/core-radius extraction
algorithm, and RWM ensemble averaging.

The center/core-radius/separation/merged status for all three physics cases
(single vortex, dipole, merging) are derived from the *sampled velocity
field* itself (the ``*_zq_*.vts`` planes written by the ``SurfaceSampler``
at z = +L/4) — one consistent method, independent of the viscous diffusion
scheme and the physics case:

  * vortex centres   — sub-grid locations of peak vorticity, matching the
    Cerretelli--Williamson separation/orientation definition;
  * core radius a_c  — radius where the azimuthally-averaged tangential
    velocity |u_theta(r)| peaks, measured on the outward semicircle before
    merger and over the full circle after merger;
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
from scipy import ndimage

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
CORE_RADIUS = 0.125  # paper's radius of maximum azimuthal velocity
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX
SEPARATION = 1.0
COLUMN_LENGTH = 50.0 * CORE_RADIUS  # mirrors lambossen_setup.py::COLUMN_LENGTH
FIELD_SPACING = 0.15 * CORE_RADIUS  # mirrors lambossen_setup.py::FIELD_SPACING
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
    "a_c0_boundary_limited",
    "a_c1_boundary_limited",
]


def unwrap_pair_orientation(angle_rad: np.ndarray) -> np.ndarray:
    """Unwrap an undirected pair axis, whose physical period is pi."""
    angle = np.asarray(angle_rad, dtype=float)
    result = np.full_like(angle, np.nan)
    finite = np.isfinite(angle)
    if finite.any():
        result[finite] = 0.5 * np.unwrap(2.0 * angle[finite])
    return result


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
    prefix: str = "vortex",
) -> dict[str, float]:
    """Return the physical constants for the analytic reference.

    ``nu`` and the core radius come from ``samples/<case>/run_metadata.json``.
    If a run has no metadata, the tutorial constants provide the reference.
    The analytic reference is never inferred from the schemes' own output.
    """
    metadata = read_run_metadata(samples_dir, prefix)
    configured_core = float(metadata.get("core_radius", a0_over_b0 * b0))
    # Runs produced before core_radius_definition was added used CORE_RADIUS
    # for r(u_theta,max), even though every plot interpreted it as the
    # Gaussian-equivalent radius a = r(u_theta,max)/BETA_RMAX.  Normalize
    # legacy data with the radius that was actually initialized.
    if metadata.get("core_radius_definition") == "gaussian_1_over_e_vorticity_radius":
        ac0 = configured_core
        velocity_peak_radius0 = float(
            metadata.get("velocity_peak_radius", BETA_RMAX * configured_core)
        )
    elif metadata:
        ac0 = configured_core / BETA_RMAX
        velocity_peak_radius0 = configured_core
    else:
        velocity_peak_radius0 = a0_over_b0 * b0
        ac0 = velocity_peak_radius0 / BETA_RMAX
    nu = float(metadata.get("viscosity", fallback_nu))
    if nu <= 0.0:
        nu = fallback_nu
    column_length = 2.0 * float(metadata.get("column_half_length", COLUMN_LENGTH / 2.0))
    circulations = metadata.get("circulations", [gamma])
    run_gamma = abs(float(circulations[0])) if circulations else abs(float(gamma))
    return {
        "nu": nu,
        "t0": ac0**2 / (4.0 * nu),
        "ac0": ac0,
        "velocity_peak_radius0": velocity_peak_radius0,
        "gamma": run_gamma,
        "separation": float(metadata.get("separation", b0)),
        "column_length": column_length,
    }


def pvd_time_map(samples_dir: Path, prefix: str, scheme: str) -> dict[int, float]:
    """Read the surface-sample PVD to get a step → physical-time mapping.

    This keeps sampled fields tied to physical time without reading checkpoints.
    """
    import xml.etree.ElementTree as ET

    pvd = samples_dir / f"{prefix}_{scheme}" / f"{prefix}_{scheme}_zq.pvd"
    if not pvd.exists():
        return {}
    try:
        tree = ET.parse(pvd)  # nosec B314
    except (OSError, ET.ParseError) as exc:
        print(f"  [field] skipping unreadable live index {pvd.name}: {exc}")
        return {}
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


def _subgrid_peak_center(
    x: np.ndarray,
    y: np.ndarray,
    signed_vorticity: np.ndarray,
    peak_index: tuple[int, int],
) -> np.ndarray:
    """Peak-vorticity location with independent parabolic sub-grid offsets."""
    i, j = peak_index
    peak = float(signed_vorticity[peak_index])
    if not np.isfinite(peak) or peak <= 0.0:
        return np.array([np.nan, np.nan])

    center = np.array([float(x[peak_index]), float(y[peak_index])])
    for axis, coordinate in ((0, x), (1, y)):
        minus = (i - 1, j) if axis == 0 else (i, j - 1)
        plus = (i + 1, j) if axis == 0 else (i, j + 1)
        f_minus = float(signed_vorticity[minus])
        f_plus = float(signed_vorticity[plus])
        denominator = f_minus - 2.0 * peak + f_plus
        if denominator >= -np.finfo(float).eps:
            continue
        offset_cells = 0.5 * (f_minus - f_plus) / denominator
        offset_cells = float(np.clip(offset_cells, -0.75, 0.75))
        spacing = float(coordinate[plus] - coordinate[peak_index])
        center[axis] += offset_cells * spacing
    return center


def _peak_candidates(values: np.ndarray, minimum_relative_peak: float = 0.20):
    """Return local maxima ordered by magnitude, excluding grid boundaries."""
    if values.size == 0 or not np.isfinite(values).any():
        return []
    maximum = float(np.nanmax(values))
    if maximum <= 0.0:
        return []
    local_max = values == ndimage.maximum_filter(values, size=3, mode="nearest")
    local_max &= values >= minimum_relative_peak * maximum
    local_max[[0, -1], :] = False
    local_max[:, [0, -1]] = False
    indices = [tuple(index) for index in np.argwhere(local_max)]
    return sorted(indices, key=lambda index: float(values[index]), reverse=True)


def _merging_peak_pair(
    field: dict,
    candidates: list[tuple[int, int]],
    previous_centers: list[np.ndarray] | None,
) -> list[tuple[int, int]]:
    """Select the physical two-peak branch using strength and continuity."""
    x, y, values = field["x"], field["y"], field["omega_z"]
    central = [index for index in candidates if np.hypot(x[index], y[index]) <= 0.75 * SEPARATION][
        :12
    ]
    if len(central) < 2:
        return central

    pairs = []
    for i, first in enumerate(central[:-1]):
        for second in central[i + 1 :]:
            p0 = np.array([x[first], y[first]], dtype=float)
            p1 = np.array([x[second], y[second]], dtype=float)
            separation = float(np.linalg.norm(p0 - p1))
            if separation < 1.5 * FIELD_SPACING:
                continue
            strength_reward = float(values[first] + values[second]) / max(
                float(np.nanmax(values)), np.finfo(float).tiny
            )
            if previous_centers is not None and len(previous_centers) == 2:
                direct = (
                    np.linalg.norm(p0 - previous_centers[0]) ** 2
                    + np.linalg.norm(p1 - previous_centers[1]) ** 2
                )
                swapped = (
                    np.linalg.norm(p1 - previous_centers[0]) ** 2
                    + np.linalg.norm(p0 - previous_centers[1]) ** 2
                )
                score = min(direct, swapped) - 0.02 * strength_reward
            else:
                midpoint_penalty = float(np.linalg.norm(0.5 * (p0 + p1))) ** 2
                separation_penalty = 0.15 * (separation - SEPARATION) ** 2
                score = midpoint_penalty + separation_penalty - 0.02 * strength_reward
            pairs.append((score, first, second))
    if not pairs:
        return central[:1]
    _, first, second = min(pairs, key=lambda item: item[0])
    return [first, second]


def _vorticity_peak_centers(
    field: dict,
    physics: str,
    previous_centers: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Vortex centres from sub-grid peak-vorticity locations."""
    x, y, wz = field["x"], field["y"], field["omega_z"]

    if physics == "dipole":
        centers = []
        for sign in (1.0, -1.0):
            signed = sign * wz
            candidates = _peak_candidates(signed)
            if not candidates:
                centers.append(np.array([np.nan, np.nan]))
                continue
            centers.append(_subgrid_peak_center(x, y, signed, candidates[0]))
        return centers

    signed = np.abs(wz) if physics == "vortex" else wz
    candidates = _peak_candidates(signed, 0.35 if physics == "merging" else 0.20)
    if not candidates:
        return [np.array([np.nan, np.nan])]

    if physics == "merging":
        peaks = _merging_peak_pair(field, candidates, previous_centers)
        return [_subgrid_peak_center(x, y, signed, peak) for peak in peaks]

    center = _subgrid_peak_center(x, y, signed, candidates[0])
    return [center] if np.isfinite(center).all() else [np.array([np.nan, np.nan])]


def _match_centers_to_previous(
    centers: list[np.ndarray], previous_centers: list[np.ndarray] | None
) -> list[np.ndarray]:
    """Keep center identities continuous without changing pair geometry."""
    if len(centers) != 2:
        return centers
    if previous_centers is None or len(previous_centers) != 2:
        return sorted(centers, key=lambda center: (center[1], center[0]), reverse=True)
    direct = sum(np.linalg.norm(a - b) ** 2 for a, b in zip(centers, previous_centers, strict=True))
    swapped = sum(
        np.linalg.norm(a - b) ** 2 for a, b in zip(centers[::-1], previous_centers, strict=True)
    )
    return centers if direct <= swapped else centers[::-1]


def _core_radius_utheta(
    field: dict,
    center: np.ndarray,
    r_max: float,
    bin_width: float | None = None,
) -> float:
    """Velocity-peak core radius from the signed u_theta profile."""
    return _core_radius_diagnostic(field, center, r_max, bin_width)[0]


def _core_radius_diagnostic(
    field: dict,
    center: np.ndarray,
    r_max: float,
    bin_width: float | None = None,
    support_mask: np.ndarray | None = None,
) -> tuple[float, bool]:
    if not np.isfinite(center).all():
        return float("nan"), False
    x, y = field["x"], field["y"]
    if bin_width is None:
        dx_values = np.abs(np.diff(x[:, 0]))
        dy_values = np.abs(np.diff(y[0, :]))
        spacings = np.concatenate([dx_values[dx_values > 0.0], dy_values[dy_values > 0.0]])
        bin_width = float(np.median(spacings)) if spacings.size else FIELD_SPACING
    dx = x - center[0]
    dy = y - center[1]
    r = np.sqrt(dx * dx + dy * dy)
    keep = (r > 0.5 * bin_width) & (r < r_max)
    if support_mask is not None:
        keep &= support_mask
    if np.count_nonzero(keep) < 20:
        return float("nan"), False

    e_theta_x = -dy / np.where(r > 0, r, 1.0)
    e_theta_y = dx / np.where(r > 0, r, 1.0)
    nearest = np.unravel_index(int(np.argmin(r)), r.shape)
    translation_x = float(field["Ux"][nearest])
    translation_y = float(field["Uy"][nearest])
    sign = np.sign(float(field["omega_z"][nearest])) or 1.0
    u_theta = sign * (
        (field["Ux"] - translation_x) * e_theta_x + (field["Uy"] - translation_y) * e_theta_y
    )

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
        return float("nan"), False

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
    boundary_limited = i == radii.size - 1 or r_peak >= r_max - 1.5 * bin_width
    return float(r_peak), bool(boundary_limited)


def _search_radius(physics: str, separation: float) -> float:
    """u_theta-peak search window. Dipole/merging scale off the nominal pair
    separation; a lone vortex has no separation to scale off, so use a
    generous multiple of the *expected* final core radius instead (the core
    can diffuse well past a fixed fraction of SEPARATION by t=TOTAL_TIME)."""
    if physics == "vortex":
        expected_final_gaussian_radius = np.sqrt(
            GAUSSIAN_CORE_RADIUS**2 + 4.0 * (GAMMA / REYNOLDS_NUMBER) * TOTAL_TIME
        )
        return 2.0 * BETA_RMAX * expected_final_gaussian_radius
    return 0.5


def _diagnostics_row(
    field: dict,
    physics: str,
    previous_centers: list[np.ndarray] | None = None,
) -> list:
    centers = _match_centers_to_previous(
        _vorticity_peak_centers(field, physics, previous_centers), previous_centers
    )
    c0 = centers[0] if len(centers) >= 1 else np.array([np.nan, np.nan])
    c1 = centers[1] if len(centers) >= 2 else np.array([np.nan, np.nan])

    separation = (
        float(np.linalg.norm(c0 - c1))
        if np.isfinite(c0).all() and np.isfinite(c1).all()
        else float("nan")
    )
    merged = physics == "merging" and not np.isfinite(separation)

    r_max = _search_radius(physics, separation)
    support0 = support1 = None
    if np.isfinite(c0).all() and np.isfinite(c1).all():
        x, y = field["x"], field["y"]
        # The paper excludes the zone directly between the two vortices and
        # averages u_theta on the outward semicircle of each core.
        outward0 = c0 - c1
        outward1 = -outward0
        support0 = (x - c0[0]) * outward0[0] + (y - c0[1]) * outward0[1] >= 0.0
        support1 = (x - c1[0]) * outward1[0] + (y - c1[1]) * outward1[1] >= 0.0
    a_c0, limited0 = (
        _core_radius_diagnostic(field, c0, r_max, support_mask=support0)
        if np.isfinite(c0).all()
        else (float("nan"), False)
    )
    a_c1, limited1 = (
        _core_radius_diagnostic(field, c1, r_max, support_mask=support1)
        if np.isfinite(c1).all()
        else (float("nan"), False)
    )
    if limited0:
        a_c0 = float("nan")
    if limited1:
        a_c1 = float("nan")

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
        limited0,
        limited1,
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
        previous_centers = None
        merged_phase = False
        for path in vts:
            step = int(VTS_STEP_RE.search(path.name).group(1))
            try:
                field = read_surface_field(path)
            except Exception as exc:
                # A live solver may still be writing its newest VTS.  Keep
                # every complete sample and make the transient skip explicit.
                print(f"  [field] {case_name}: skipping unreadable {path.name}: {exc}")
                continue
            field["step"] = step
            field["time"] = timeline.get(step, float("nan"))
            if physics == "merging" and merged_phase:
                # Merger is a topological event.  Treat it as an absorbing
                # state: late-time grid noise must not resurrect a vortex
                # pair and create a fictitious separation or angle history.
                row = _diagnostics_row(field, "vortex")
                row[11] = True
            else:
                row = _diagnostics_row(field, physics, previous_centers)
            rows.append(row)
            if physics == "merging" and bool(row[11]):
                merged_phase = True
                previous_centers = None
            elif np.isfinite(row[2:6]).all():
                previous_centers = [np.asarray(row[2:4]), np.asarray(row[4:6])]

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


def average_surface_histories(target_dir: Path, member_dirs: list[Path]) -> list[str]:
    """Average every common RWM sampled plane before extracting diagnostics.

    Averaging the Eulerian fields is essential: centre finding is nonlinear,
    so averaging already-extracted noisy centre trajectories is not equivalent
    to diagnosing the ensemble-mean vorticity field.
    """
    if not member_dirs:
        raise ValueError("at least one RWM member directory is required")
    surface_sets = [
        {path.name for path in directory.glob("*_zq_*.vts")} for directory in member_dirs
    ]
    common = sorted(set.intersection(*surface_sets))
    if not common:
        raise ValueError("RWM ensemble members have no common sampled surfaces")
    if any(names != surface_sets[0] for names in surface_sets[1:]):
        raise ValueError("RWM ensemble sampled-surface histories differ")

    target_dir.mkdir(parents=True, exist_ok=True)
    for name in common:
        _average_vts(target_dir / name, [directory / name for directory in member_dirs])
    return common


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

    angles = np.stack(
        [unwrap_pair_orientation(frame["angle_rad"].to_numpy(float)) for frame in frames]
    )
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
