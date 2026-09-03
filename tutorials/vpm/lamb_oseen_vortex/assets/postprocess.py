#!/usr/bin/env python3
"""Single post-processing authority for the Lamb--Oseen VPM tutorial.

This module owns sampled-field reading, physical feature definitions, RWM
ensemble reconstruction and uncertainty, theory/reference transformations,
plot-ready data preparation, certification, and provenance.  Plot scripts are
deliberately presentation-only.

  * vortex centres   — geometric centres of the connected areas enclosed by
    the 80%-of-peak vorticity contours, following Cerretelli--Williamson;
  * core radius a_c  — radius where the azimuthally-averaged tangential
    velocity |u_theta(r)| peaks, measured on the outward semicircle before
    merger and over the full circle after merger;
  * vortex_separation and orbital angle (dipole/merging pair).

Run ``postprocess.py --extract-fields`` to rebuild deterministic feature CSVs,
or use the other command-line modes for RWM aggregation and certification.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import h5py
from scipy import signal, stats
from scipy.special import expi

# -- Directory layout --------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent  # …/assets/
SCRIPT_DIR = ASSETS_DIR.parent  # …/lamb_oseen_vortex/
CASE_DIR = SCRIPT_DIR
FIGURES_DIR = SCRIPT_DIR / "figures"
SAMPLES_DIR = SCRIPT_DIR / "samples"
SOLUTION_DIR = SCRIPT_DIR / "solution"
REF_DIR = ASSETS_DIR / "references"

# -- Tutorial constants -------------------------------------------------------
SCHEMES = ("cs", "rwm", "dvh", "gbd")
# Draw CS last so its circle markers remain visible wherever methods overlap.
# The high CS layer also keeps the circles above analytic/reference curves.
SCHEME_DRAW_ORDER = tuple(scheme for scheme in SCHEMES if scheme != "cs") + ("cs",)
_SCHEME_ZORDER = {
    scheme: (200 if scheme == "cs" else 10 + index)
    for index, scheme in enumerate(SCHEME_DRAW_ORDER)
}
CASES = ("vortex", "dipole", "merging")
ENERGY_CASES = (
    ("vortex", "Single vortex", 1),
    ("dipole", "Vortex dipole", 2),
    ("merging", "Co-rotating merger", 2),
)
DIRECT_ENERGY_PARTICLE_LIMIT = 50_000

# Legacy-input fallbacks used only when a result folder lacks run_metadata.json.
# They are NOT a second source of truth: the authoritative values are read from
# samples/<case>/run_metadata.json, with setup.py as the physical definition.
# These fallbacks match the maintained setup exactly.
BETA_RMAX = 1.12
REFERENCE_CIRCULATION = 1.0
REYNOLDS_NUMBER = 530.0
CORE_RADIUS = 0.125  # paper's radius of maximum azimuthal velocity
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX
SEPARATION = 1.0
COLUMN_LENGTH = 40.0 * CORE_RADIUS  # mirrors setup.py::COLUMN_LENGTH
FIELD_SPACING = 0.15 * CORE_RADIUS  # mirrors setup.py::FIELD_SPACING
TOTAL_TIME = 30.0  # fallback reference time [s] when no run data is available

VTS_STEP_RE = re.compile(r"_(\d+)\.vts$")

FIELD_CSV_COLUMNS = [
    "time",
    "step",
    "vortex_centre_0_x",
    "vortex_centre_0_y",
    "vortex_centre_1_x",
    "vortex_centre_1_y",
    "vortex_separation",
    "core_radius_0",
    "core_radius_1",
    "mean_core_radius",
    "angle_radians",
    "is_pair_unresolved",
    "is_core_radius_0_boundary_limited",
    "is_core_radius_1_boundary_limited",
    "peak_saddle_contrast",
    "peak_saddle_contrast_standard_error",
    "peak_saddle_signal_to_noise",
    "orientation_anisotropy",
    "is_peak_coalesced",
]


def unwrap_pair_orientation(angle_radians: np.ndarray) -> np.ndarray:
    """Unwrap an undirected pair axis, whose physical period is pi."""
    angle = np.asarray(angle_radians, dtype=float)
    result = np.full_like(angle, np.nan)
    finite = np.isfinite(angle)
    if finite.any():
        result[finite] = 0.5 * np.unwrap(2.0 * angle[finite])
    return result


# =============================================================
# Run-metadata / sampled-field readers
# =============================================================


def read_run_metadata(samples_dir: Path, prefix: str = "vortex") -> dict:
    """Load the physical constants stored with the sampled results.

    Plotting must not depend on a dense particle backup: those files are sparse
    restart backups, while ``samples/<case>/run_metadata.json`` is written
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
    circulation: float,
    fallback_kinematic_viscosity: float,
    b0: float,
    a0_over_b0: float,
    prefix: str = "vortex",
) -> dict[str, float]:
    """Return the physical constants for the analytic reference.

    ``kinematic_viscosity`` and the core radius come from ``samples/<case>/run_metadata.json``.
    If a run has no metadata, the tutorial constants provide the reference.
    The analytic reference is never inferred from the schemes' own output.
    """
    metadata = read_run_metadata(samples_dir, prefix)
    configured_core = float(metadata.get("core_radius", a0_over_b0 * b0))
    ac0 = configured_core
    velocity_peak_radius0 = float(metadata.get("velocity_peak_radius", BETA_RMAX * configured_core))
    kinematic_viscosity = float(metadata.get("kinematic_viscosity", fallback_kinematic_viscosity))
    if kinematic_viscosity <= 0.0:
        kinematic_viscosity = fallback_kinematic_viscosity
    column_length = 2.0 * float(metadata.get("column_half_length", COLUMN_LENGTH / 2.0))
    circulations = metadata.get("circulations", [circulation])
    run_circulation = abs(float(circulations[0])) if circulations else abs(float(circulation))
    return {
        "kinematic_viscosity": kinematic_viscosity,
        "t0": ac0**2 / (4.0 * kinematic_viscosity),
        "ac0": ac0,
        "velocity_peak_radius0": velocity_peak_radius0,
        "circulation": run_circulation,
        "vortex_separation": float(metadata.get("vortex_separation", b0)),
        "column_length": column_length,
    }


def pvd_time_map(samples_dir: Path, prefix: str, scheme: str) -> dict[int, float]:
    """Read the surface-sample PVD to get a step → physical-time mapping.

    This keeps sampled fields tied to physical time without reading backups.
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
    velocity = np.asarray(grid.point_data["velocity"], dtype=np.float64)
    vorticity = np.asarray(grid.point_data["vorticity"], dtype=np.float64)

    ni = len(np.unique(np.round(points[:, 0], 9)))
    nj = len(np.unique(np.round(points[:, 1], 9)))
    if ni * nj != points.shape[0]:
        raise ValueError(f"{path.name}: expected {ni}x{nj} structured grid")
    shape = (ni, nj)

    def _f(flat):
        return flat.reshape(shape, order="F")

    field = {
        "x": _f(points[:, 0]),
        "y": _f(points[:, 1]),
        "velocity_x": _f(velocity[:, 0]),
        "velocity_y": _f(velocity[:, 1]),
        "vorticity_z": _f(vorticity[:, 2]),
    }
    for vtk_name, field_name in (
        ("velocity_standard_error", "velocity_standard_error"),
        ("vorticity_standard_error", "vorticity_standard_error"),
    ):
        values = grid.point_data.get(vtk_name)
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        field[f"{field_name}_x"] = _f(values[:, 0])
        field[f"{field_name}_y"] = _f(values[:, 1])
        field[f"{field_name}_z"] = _f(values[:, 2])
    for name in ("ensemble_size", "confidence_multiplier"):
        values = grid.field_data.get(name)
        if values is not None and np.asarray(values).size:
            field[name] = float(np.asarray(values).reshape(-1)[0])
    for name in ("velocity_gradient_yx", "velocity_gradient_yx_standard_error"):
        values = grid.point_data.get(name)
        if values is not None:
            field[name] = _f(np.asarray(values, dtype=np.float64).reshape(-1))
    return field


# =============================================================
# Vortex properties from the 2D field
# =============================================================


def _subgrid_peak_centre(
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

    centre = np.array([float(x[peak_index]), float(y[peak_index])])
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
        centre[axis] += offset_cells * spacing
    return centre


def _high_vorticity_region_centre(
    x: np.ndarray,
    y: np.ndarray,
    signed_vorticity: np.ndarray,
    peak_index: tuple[int, int],
    contour_fraction: float = 0.80,
) -> np.ndarray:
    """Centre of the connected area enclosed by the 80%-of-peak contour.

    This is Cerretelli & Williamson's centre definition.  It is appreciably
    less sensitive than the location of a single grid maximum and remains a
    measurement of the Eulerian field rather than a particle-label statistic.
    """
    peak = float(signed_vorticity[peak_index])
    if not np.isfinite(peak) or peak <= 0.0:
        return np.array([np.nan, np.nan])
    mask = np.isfinite(signed_vorticity) & (signed_vorticity >= contour_fraction * peak)
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    label = int(labels[peak_index])
    if label == 0:
        return _subgrid_peak_centre(x, y, signed_vorticity, peak_index)
    region = labels == label
    if np.count_nonzero(region) < 3:
        return _subgrid_peak_centre(x, y, signed_vorticity, peak_index)
    return np.array([float(np.mean(x[region])), float(np.mean(y[region]))])


def _peak_candidates(values: np.ndarray, min_relative_peak: float = 0.20):
    """Return local maxima ordered by magnitude, excluding grid boundaries."""
    if values.size == 0 or not np.isfinite(values).any():
        return []
    maximum = float(np.nanmax(values))
    if maximum <= 0.0:
        return []
    local_max = values == ndimage.maximum_filter(values, size=3, mode="nearest")
    local_max &= values >= min_relative_peak * maximum
    local_max[[0, -1], :] = False
    local_max[:, [0, -1]] = False
    indices = [tuple(index) for index in np.argwhere(local_max)]
    return sorted(indices, key=lambda index: float(values[index]), reverse=True)


def _merging_peak_pair(
    field: dict,
    candidates: list[tuple[int, int]],
    previous_centres: list[np.ndarray] | None,
) -> list[tuple[int, int]]:
    """Select the physical two-peak branch using strength and continuity."""
    x, y, values = field["x"], field["y"], field["vorticity_z"]
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
            vortex_separation = float(np.linalg.norm(p0 - p1))
            if vortex_separation < 1.5 * FIELD_SPACING:
                continue
            strength_reward = float(values[first] + values[second]) / max(
                float(np.nanmax(values)), np.finfo(float).tiny
            )
            if previous_centres is not None and len(previous_centres) == 2:
                direct = (
                    np.linalg.norm(p0 - previous_centres[0]) ** 2
                    + np.linalg.norm(p1 - previous_centres[1]) ** 2
                )
                swapped = (
                    np.linalg.norm(p1 - previous_centres[0]) ** 2
                    + np.linalg.norm(p0 - previous_centres[1]) ** 2
                )
                score = min(direct, swapped) - 0.02 * strength_reward
            else:
                midpoint_penalty = float(np.linalg.norm(0.5 * (p0 + p1))) ** 2
                separation_penalty = 0.15 * (vortex_separation - SEPARATION) ** 2
                score = midpoint_penalty + separation_penalty - 0.02 * strength_reward
            pairs.append((score, first, second))
    if not pairs:
        return central[:1]
    _, first, second = min(pairs, key=lambda item: item[0])
    return [first, second]


def _vorticity_peak_centres(
    field: dict,
    physics: str,
    previous_centres: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Vortex centres from connected 80%-of-peak vorticity regions."""
    x, y, wz = field["x"], field["y"], field["vorticity_z"]

    if physics == "dipole":
        centres = []
        for sign in (1.0, -1.0):
            signed = sign * wz
            candidates = _peak_candidates(signed)
            if not candidates:
                centres.append(np.array([np.nan, np.nan]))
                continue
            centres.append(_high_vorticity_region_centre(x, y, signed, candidates[0]))
        peaks = []
        for sign in (1.0, -1.0):
            candidates = _peak_candidates(sign * wz)
            peaks.append(candidates[0] if candidates else (-1, -1))
        return centres, peaks

    signed = np.abs(wz) if physics == "vortex" else wz
    candidates = _peak_candidates(signed, 0.05 if physics == "merging" else 0.20)
    if not candidates:
        return [np.array([np.nan, np.nan])], []

    if physics == "merging":
        peaks = _merging_peak_pair(field, candidates, previous_centres)
        return [_high_vorticity_region_centre(x, y, signed, peak) for peak in peaks], peaks

    centre = _high_vorticity_region_centre(x, y, signed, candidates[0])
    centres = [centre] if np.isfinite(centre).all() else [np.array([np.nan, np.nan])]
    return centres, [candidates[0]]


def _pair_resolution_diagnostic(
    field: dict,
    peaks: list[tuple[int, int]],
    centres: list[np.ndarray],
) -> tuple[bool, float, float, float]:
    """Test whether two same-sign peaks are resolved above Monte Carlo noise.

    A pair is resolved only when it has two spatially distinct 80%-contour
    regions and the smaller peak rises above the intervening saddle by more
    than the ensemble confidence multiplier times the contrast standard error.
    Deterministic fields use a one-percent contrast floor in place of a Monte
    Carlo uncertainty estimate.
    """
    if len(peaks) != 2 or len(centres) != 2 or not np.isfinite(centres).all():
        return False, float("nan"), float("nan"), float("nan")
    wz = np.asarray(field["vorticity_z"], dtype=float)
    first, second = peaks
    if first == second or min(*first, *second) < 0:
        return False, float("nan"), float("nan"), float("nan")

    dx_values = np.diff(field["x"][:, 0])
    dy_values = np.diff(field["y"][0, :])
    grid_spacing = float(
        np.median(np.abs(np.concatenate((dx_values[dx_values != 0], dy_values[dy_values != 0]))))
    )
    centre_separation = float(np.linalg.norm(centres[0] - centres[1]))
    if centre_separation < 2.0 * grid_spacing:
        return False, float("nan"), float("nan"), float("nan")

    n_line = max(16, int(np.ceil(np.linalg.norm(np.subtract(first, second)))) * 4 + 1)
    line_i = np.linspace(first[0], second[0], n_line)
    line_j = np.linspace(first[1], second[1], n_line)
    line = ndimage.map_coordinates(wz, (line_i, line_j), order=1, mode="nearest")
    if line.size < 5 or not np.isfinite(line).all():
        return False, float("nan"), float("nan"), float("nan")
    interior = line[1:-1]
    saddle_offset = int(np.argmin(interior)) + 1
    saddle = float(line[saddle_offset])
    peak_values = np.array([float(wz[first]), float(wz[second])])
    contrast = float(np.min(peak_values) - saddle)

    se_field = field.get("vorticity_standard_error_z")
    if se_field is None:
        threshold = 0.01 * max(float(np.max(peak_values)), np.finfo(float).tiny)
        return contrast > threshold, contrast, float("nan"), float("inf")

    se_field = np.asarray(se_field, dtype=float)
    line_se = ndimage.map_coordinates(se_field, (line_i, line_j), order=1, mode="nearest")
    lower_peak = first if peak_values[0] <= peak_values[1] else second
    contrast_se = float(np.hypot(se_field[lower_peak], line_se[saddle_offset]))
    multiplier = float(field.get("confidence_multiplier", 1.96))
    peak_significant = all(
        value > multiplier * se_field[index]
        for value, index in zip(peak_values, (first, second), strict=True)
    )
    signal_to_noise = contrast / max(contrast_se, np.finfo(float).tiny)
    return (
        bool(peak_significant and contrast > multiplier * contrast_se),
        contrast,
        contrast_se,
        signal_to_noise,
    )


def _merged_vortex_orientation(field: dict, centre: np.ndarray) -> tuple[float, float]:
    """Orientation and anisotropy of the merged positive-vorticity structure.

    Once two vorticity maxima have coalesced, their joining line no longer
    exists.  Cerretelli & Williamson nevertheless continue ``theta`` as the
    orientation of the merged elliptical vortex.  We estimate that undirected
    axis from the vorticity-weighted second central moment on the connected
    5%-of-peak support.  The quadrupole moment is stable under grid-scale peak
    motion and has the same period pi as the pre-merger pair axis.
    """
    if not np.isfinite(centre).all():
        return float("nan"), float("nan")
    wz = np.asarray(field["vorticity_z"], dtype=float)
    positive = np.clip(wz, 0.0, None)
    peak = float(np.nanmax(positive))
    if not np.isfinite(peak) or peak <= 0.0:
        return float("nan"), float("nan")
    support = positive >= 0.05 * peak
    labels, _ = ndimage.label(support, structure=np.ones((3, 3), dtype=np.int8))
    nearest = np.unravel_index(
        int(np.argmin((field["x"] - centre[0]) ** 2 + (field["y"] - centre[1]) ** 2)),
        positive.shape,
    )
    label = int(labels[nearest])
    if label:
        support = labels == label
    weights = np.where(support, positive, 0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= np.finfo(float).tiny:
        return float("nan"), float("nan")
    dx = np.asarray(field["x"], dtype=float) - centre[0]
    dy = np.asarray(field["y"], dtype=float) - centre[1]
    qxx = float(np.sum(weights * dx * dx) / weight_sum)
    qyy = float(np.sum(weights * dy * dy) / weight_sum)
    qxy = float(np.sum(weights * dx * dy) / weight_sum)
    discriminant = float(np.hypot(qxx - qyy, 2.0 * qxy))
    trace = qxx + qyy
    if trace <= np.finfo(float).tiny:
        return float("nan"), float("nan")
    return 0.5 * float(np.arctan2(2.0 * qxy, qxx - qyy)), discriminant / trace


def _match_centres_to_previous(
    centres: list[np.ndarray], previous_centres: list[np.ndarray] | None
) -> list[np.ndarray]:
    """Keep centre identities continuous without changing pair geometry."""
    if len(centres) != 2:
        return centres
    if previous_centres is None or len(previous_centres) != 2:
        return sorted(centres, key=lambda centre: (centre[1], centre[0]), reverse=True)
    direct = sum(np.linalg.norm(a - b) ** 2 for a, b in zip(centres, previous_centres, strict=True))
    swapped = sum(
        np.linalg.norm(a - b) ** 2 for a, b in zip(centres[::-1], previous_centres, strict=True)
    )
    return centres if direct <= swapped else centres[::-1]


def _core_radius_diagnostic(
    field: dict,
    centre: np.ndarray,
    r_max: float,
    bin_width: float | None = None,
    support_mask: np.ndarray | None = None,
) -> tuple[float, bool]:
    if not np.isfinite(centre).all():
        return float("nan"), False
    x, y = field["x"], field["y"]
    if bin_width is None:
        dx_values = np.abs(np.diff(x[:, 0]))
        dy_values = np.abs(np.diff(y[0, :]))
        spacings = np.concatenate([dx_values[dx_values > 0.0], dy_values[dy_values > 0.0]])
        bin_width = float(np.median(spacings)) if spacings.size else FIELD_SPACING
    dx = x - centre[0]
    dy = y - centre[1]
    r = np.sqrt(dx * dx + dy * dy)
    keep = (r > 0.5 * bin_width) & (r < r_max)
    if support_mask is not None:
        keep &= support_mask
    if np.count_nonzero(keep) < 20:
        return float("nan"), False

    e_theta_x = -dy / np.where(r > 0, r, 1.0)
    e_theta_y = dx / np.where(r > 0, r, 1.0)
    nearest = np.unravel_index(int(np.argmin(r)), r.shape)
    translation_x = float(field["velocity_x"][nearest])
    translation_y = float(field["velocity_y"][nearest])
    sign = np.sign(float(field["vorticity_z"][nearest])) or 1.0
    u_theta = sign * (
        (field["velocity_x"] - translation_x) * e_theta_x
        + (field["velocity_y"] - translation_y) * e_theta_y
    )

    edges = np.arange(0.5 * bin_width, r_max + bin_width, bin_width)
    bin_index = np.clip(np.searchsorted(edges, r[keep], side="right") - 1, 0, None)
    core_radius, magnitudes = [], []
    for b in np.unique(bin_index):
        selected = bin_index == b
        if np.count_nonzero(selected) < 3:
            continue
        core_radius.append(0.5 * (edges[b] + edges[min(b + 1, len(edges) - 1)]))
        magnitudes.append(float(u_theta[keep][selected].mean()))
    core_radius = np.asarray(core_radius)
    magnitudes = np.asarray(magnitudes)
    if core_radius.size < 3:
        return float("nan"), False

    i = int(np.argmax(magnitudes))
    if 0 < i < core_radius.size - 1:
        u1, u2, u3 = magnitudes[i - 1 : i + 2]
        r1, r2, r3 = core_radius[i - 1 : i + 2]
        denominator = u1 - 2.0 * u2 + u3
        if abs(denominator) > 1e-12:
            r_peak = r2 + 0.5 * (u1 - u3) / denominator * (r2 - r1)
        else:
            r_peak = r2
    else:
        r_peak = core_radius[i]
    boundary_limited = i == core_radius.size - 1 or r_peak >= r_max - 1.5 * bin_width
    return float(r_peak), bool(boundary_limited)


def _search_radius(physics: str, vortex_separation: float) -> float:
    """Search far enough to bracket the velocity maximum at the final time.

    A fixed half-separation window incorrectly censors a diffusing pair once
    its core grows beyond ``b/2``.  The outward-semicircle mask already removes
    the other vortex, so the search radius should follow the viscous diffusion
    scale for single and paired vortices alike.
    """
    expected_final_gaussian_radius = np.sqrt(
        GAUSSIAN_CORE_RADIUS**2 + 4.0 * (REFERENCE_CIRCULATION / REYNOLDS_NUMBER) * TOTAL_TIME
    )
    return 2.0 * BETA_RMAX * expected_final_gaussian_radius


def _diagnostics_row(
    field: dict,
    physics: str,
    previous_centres: list[np.ndarray] | None = None,
    force_merged: bool = False,
) -> list:
    raw_centres, peaks = _vorticity_peak_centres(field, physics, previous_centres)
    centres = _match_centres_to_previous(raw_centres, previous_centres)
    contrast = contrast_se = contrast_snr = float("nan")
    pair_resolved = True
    peak_coalesced = False
    if physics == "merging" and not force_merged:
        pair_resolved, contrast, contrast_se, contrast_snr = _pair_resolution_diagnostic(
            field, peaks, centres
        )
        peak_coalesced = len(peaks) < 2
        if peak_coalesced:
            merged_centres, _ = _vorticity_peak_centres(field, "vortex")
            centres = merged_centres
    elif physics == "merging":
        pair_resolved = False
        peak_coalesced = True
        merged_centres, _ = _vorticity_peak_centres(field, "vortex")
        centres = merged_centres
    c0 = centres[0] if len(centres) >= 1 else np.array([np.nan, np.nan])
    c1 = centres[1] if len(centres) >= 2 else np.array([np.nan, np.nan])

    vortex_separation = (
        float(np.linalg.norm(c0 - c1))
        if np.isfinite(c0).all() and np.isfinite(c1).all()
        else float("nan")
    )
    pair_unresolved = physics == "merging" and not pair_resolved
    if pair_unresolved and not peak_coalesced:
        # A confidence failure says that a pair feature is not estimable; it
        # does not prove physical coalescence.  Preserve the distinction.
        vortex_separation = float("nan")
    elif peak_coalesced:
        # In the paper b is the distance between vorticity maxima.  It is
        # exactly zero once only one maximum remains; NaN would incorrectly
        # truncate a physically defined post-merger history.
        vortex_separation = 0.0

    r_max = _search_radius(physics, vortex_separation)
    support0 = support1 = None
    if not pair_unresolved and np.isfinite(c0).all() and np.isfinite(c1).all():
        x, y = field["x"], field["y"]
        # The paper excludes the zone directly between the two vortices and
        # averages u_theta on the outward semicircle of each core.
        outward0 = c0 - c1
        outward1 = -outward0
        support0 = (x - c0[0]) * outward0[0] + (y - c0[1]) * outward0[1] >= 0.0
        support1 = (x - c1[0]) * outward1[0] + (y - c1[1]) * outward1[1] >= 0.0
    core_radius_0, limited0 = (
        _core_radius_diagnostic(field, c0, r_max, support_mask=support0)
        if np.isfinite(c0).all()
        else (float("nan"), False)
    )
    core_radius_1, limited1 = (
        _core_radius_diagnostic(field, c1, r_max, support_mask=support1)
        if np.isfinite(c1).all()
        else (float("nan"), False)
    )
    if limited0:
        core_radius_0 = float("nan")
    if limited1:
        core_radius_1 = float("nan")
    if pair_unresolved and not peak_coalesced:
        core_radius_0 = core_radius_1 = float("nan")

    mean_core_radius = (
        float(np.mean([a for a in (core_radius_0, core_radius_1) if np.isfinite(a)]))
        if any(np.isfinite(a) for a in (core_radius_0, core_radius_1))
        else float("nan")
    )

    angle = float("nan")
    orientation_anisotropy = float("nan")
    if not pair_unresolved and np.isfinite(c0).all() and np.isfinite(c1).all():
        midpoint = 0.5 * (c0 + c1)
        angle = float(np.arctan2(c0[1] - midpoint[1], c0[0] - midpoint[0]))
    elif peak_coalesced:
        angle, orientation_anisotropy = _merged_vortex_orientation(field, c0)

    return [
        field["time"],
        field["step"],
        c0[0],
        c0[1],
        c1[0],
        c1[1],
        vortex_separation,
        core_radius_0,
        core_radius_1,
        mean_core_radius,
        angle,
        pair_unresolved,
        limited0,
        limited1,
        contrast,
        contrast_se,
        contrast_snr,
        orientation_anisotropy,
        peak_coalesced,
    ]


def diagnostics_row(
    field: dict,
    physics: str,
    previous_centres: list[np.ndarray] | None = None,
    force_merged: bool = False,
) -> list:
    """Public entry point used by ensemble and deterministic postprocessing."""
    return _diagnostics_row(field, physics, previous_centres, force_merged)


def _mask_lost_pair_features(row: list) -> list:
    """Suppress pair-dependent values after statistical identifiability is lost."""
    row = list(row)
    for name in (
        "vortex_separation",
        "core_radius_0",
        "core_radius_1",
        "mean_core_radius",
        "angle_radians",
    ):
        row[FIELD_CSV_COLUMNS.index(name)] = float("nan")
    row[FIELD_CSV_COLUMNS.index("is_pair_unresolved")] = True
    return row


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
        # Ensemble aggregation already applied the nonlinear diagnostics to
        # the mean field and used a delete-one-member jackknife for their
        # uncertainty.  Re-extracting these rows here would silently discard
        # the intervals immediately before plotting.
        if scheme == "rwm":
            metadata_path = samples_dir / case_name / "run_metadata.json"
            diagnostics_path = samples_dir / case_name / "field_diagnostics.csv"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                columns = set(pd.read_csv(diagnostics_path, nrows=0).columns)
            except (OSError, ValueError, pd.errors.ParserError):
                metadata = {}
                columns = set()
            if (
                str(metadata.get("statistical_estimator", "")).startswith(
                    "fixed_time_seed_ensemble_mean"
                )
                and "mean_core_radius_standard_error" in columns
            ):
                print(f"  [field] {case_name}: preserving ensemble/jackknife diagnostics")
                continue
        timeline = pvd_time_map(samples_dir, physics, scheme)
        rows = []
        previous_centres = None
        coalesced_phase = False
        pair_lost = False
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
            if physics == "merging" and coalesced_phase:
                # Loss of a resolvable pair is absorbing for these pair
                # observables: late-time grid noise must not resurrect two
                # centres and create a fictitious separation or angle history.
                row = _diagnostics_row(field, "merging", force_merged=True)
            else:
                row = _diagnostics_row(field, physics, previous_centres)
                if (
                    physics == "merging"
                    and pair_lost
                    and not bool(row[FIELD_CSV_COLUMNS.index("is_peak_coalesced")])
                ):
                    row = _mask_lost_pair_features(row)
            rows.append(row)
            if physics == "merging" and bool(row[FIELD_CSV_COLUMNS.index("is_peak_coalesced")]):
                coalesced_phase = True
                previous_centres = None
            elif physics == "merging" and bool(row[11]):
                pair_lost = True
            elif np.isfinite(row[2:6]).all():
                previous_centres = [np.asarray(row[2:4]), np.asarray(row[4:6])]

        out = samples_dir / case_name / "field_diagnostics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(FIELD_CSV_COLUMNS)
            writer.writerows(rows)
        print(f"  [field] {case_name}: wrote field_diagnostics.csv ({len(rows)} samples)")


# =============================================================
# CLI entry point
# =============================================================


# =============================================================
# Plotting utilities (absorbed from plot_style.py)
# =============================================================

THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"

_THEME_MODULE = None


def _theme():
    global _THEME_MODULE
    if _THEME_MODULE is None:
        if not THEME_PATH.exists():
            raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
        spec = importlib.util.spec_from_file_location("openonda_matplotlib_setup", THEME_PATH)
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        _THEME_MODULE = theme
    return _THEME_MODULE


def load_theme() -> tuple[dict[str, str], object | None]:
    """Load the OpenONDA matplotlib theme and return (COLORS dict, theme module)."""
    theme = _theme()
    theme.set_style()
    return dict(theme.COLORS), theme


def build_style_map(colors: dict[str, str]) -> dict[str, dict]:
    """Map scheme names to plot style dicts (color, marker, label)."""
    return {name: dict(style) for name, style in _theme().LAMB_OSEEN_SCHEME_STYLE.items()}


def scheme_zorder(scheme: str, offset: int = 0) -> int:
    """Layer one scheme consistently, with CS above every comparison curve."""
    return _SCHEME_ZORDER[scheme] + offset


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a named figure size in inches from the shared theme."""
    return _theme().figure_size(name)


def save_fig(fig, path: Path, dpi: int) -> None:
    """Save without tight layout or cropping; manual subplots_adjust() takes precedence."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    outputs = (
        (out.with_suffix(".png"), out.with_suffix(".pdf")) if out.suffix == ".both" else (out,)
    )
    for output in outputs:
        fig.savefig(output, dpi=dpi, bbox_inches=None)
        print(f"  Saved: {output}")
    plt.close(fig)


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse as _argparse

    p = _argparse.ArgumentParser(description=description)
    p.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Raster resolution.")
    p.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="png",
        help="Output figure format.",
    )
    kinematic_viscosity = REFERENCE_CIRCULATION / REYNOLDS_NUMBER
    p.set_defaults(
        samples_dir=SAMPLES_DIR,
        figures_dir=FIGURES_DIR,
        circulation=REFERENCE_CIRCULATION,
        kinematic_viscosity=kinematic_viscosity,
        b0=SEPARATION,
        a0_over_b0=CORE_RADIUS / SEPARATION,
    )
    return p


# =============================================================
# Plot-ready data products
# =============================================================

MERGING_NORMALIZED_END_TIME = 3.0
THETA_REFERENCE = REF_DIR / "theta_vs_tau.csv"
CORE_REFERENCE = REF_DIR / "a2_over_b02.csv"
SEPARATION_DIMENSIONAL_REFERENCE = REF_DIR / "b_over_b0_time.csv"
# Figures 4 and 5 report the same Re=530 experiment in dimensional and
# viscous time, respectively.  Their common final acquisition is digitized as
# t=33.60 s in figure 4 and tau=nu*t/b0^2=0.04744 in figure 5(b).
REFERENCE_FINAL_TIME_SECONDS = 33.60
REFERENCE_FINAL_VISCOUS_TIME = 0.04744
REFERENCE_VISCOUS_TIME_PER_SECOND = REFERENCE_FINAL_VISCOUS_TIME / REFERENCE_FINAL_TIME_SECONDS


def lamb_oseen_profile(
    radius: np.ndarray,
    time: float,
    circulation: float,
    kinematic_viscosity: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact Lamb--Oseen velocity and vorticity at physical vortex age ``time``."""
    core_squared = 4.0 * kinematic_viscosity * time
    gaussian_core = float(np.sqrt(core_squared))
    vorticity = circulation / (np.pi * core_squared) * np.exp(-(radius**2) / core_squared)
    velocity = np.zeros_like(radius)
    nonzero = np.abs(radius) > 1.0e-12
    velocity[nonzero] = (
        circulation
        / (2.0 * np.pi * radius[nonzero])
        * (1.0 - np.exp(-(radius[nonzero] ** 2) / core_squared))
    )
    return velocity, vorticity, gaussian_core


def lamb_oseen_gradient(
    radius: np.ndarray,
    time: float,
    circulation: float,
    kinematic_viscosity: float,
) -> np.ndarray:
    """Exact radial derivative of Lamb--Oseen azimuthal velocity."""
    core_squared = 4.0 * kinematic_viscosity * time
    gradient = np.zeros_like(radius)
    nonzero = np.abs(radius) > 1.0e-12
    exponential = np.exp(-(radius[nonzero] ** 2) / core_squared)
    gradient[nonzero] = (
        circulation
        / (2.0 * np.pi)
        * (2.0 * exponential / core_squared - (1.0 - exponential) / radius[nonzero] ** 2)
    )
    gradient[~nonzero] = circulation / (2.0 * np.pi * core_squared)
    return gradient


def load_profile(
    samples_dir: Path,
    scheme: str,
    target_time: float | None = None,
    include_uncertainty: bool = False,
) -> tuple | None:
    """Return the y=0 single-vortex profile at a common physical time."""
    timeline = pvd_time_map(samples_dir, "vortex", scheme)
    if not timeline:
        return None
    if target_time is None:
        ordered_steps = sorted(timeline, key=timeline.get, reverse=True)
    else:
        ordered_steps = sorted(
            timeline,
            key=lambda step: (abs(timeline[step] - target_time), timeline[step] > target_time),
        )
    field = None
    selected_step = None
    for step in ordered_steps:
        path = samples_dir / f"vortex_{scheme}" / f"vortex_{scheme}_zq_{step:06d}.vts"
        if not path.is_file():
            continue
        try:
            field = read_surface_field(path)
        except Exception as exc:
            print(f"  [vortex] skipping unreadable live sample {path.name}: {exc}")
            continue
        selected_step = step
        break
    if field is None or selected_step is None or np.abs(field["velocity_y"]).max() <= 1.0e-10:
        return None
    row = int(np.argmin(np.abs(field["y"][0, :])))
    x = field["x"][:, row]
    velocity = field["velocity_y"][:, row]
    vorticity = field["vorticity_z"][:, row]
    if not include_uncertainty:
        return x, velocity, vorticity, timeline[selected_step]
    velocity_se = field.get("velocity_standard_error_y", np.full_like(field["velocity_y"], np.nan))[
        :, row
    ]
    vorticity_se = field.get(
        "vorticity_standard_error_z", np.full_like(field["vorticity_z"], np.nan)
    )[:, row]
    gradient_se = field.get(
        "velocity_gradient_yx_standard_error", np.full_like(field["velocity_y"], np.nan)
    )[:, row]
    return (
        x,
        velocity,
        vorticity,
        timeline[selected_step],
        velocity_se,
        vorticity_se,
        gradient_se,
        float(field.get("confidence_multiplier", 1.96)),
    )


def latest_common_time(samples_dir: Path, prefix: str = "vortex") -> float | None:
    """Latest physical time reached by every available method for ``prefix``."""
    latest = [
        max(timeline.values())
        for scheme in SCHEME_DRAW_ORDER
        if (timeline := pvd_time_map(samples_dir, prefix, scheme))
    ]
    return min(latest) if latest else None


def extract_dipole_timeseries(samples_dir: Path, scheme: str) -> dict | None:
    path = samples_dir / f"dipole_{scheme}" / "field_diagnostics.csv"
    try:
        data = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["time", "step"])
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"  [dipole] skipping unreadable live CSV for {scheme!r}: {exc}")
        return None
    data = data.sort_values("step").drop_duplicates("step", keep="last")
    if data.empty:
        return None
    data = data[uniform_cadence_mask(data["step"].to_numpy(int))]
    output = {
        "t": data["time"].to_numpy(float),
        "x_core": data["vortex_centre_0_x"].to_numpy(float),
        "a_c": data["core_radius_0"].to_numpy(float),
    }
    for column, key in (
        ("vortex_centre_0_x_ci_lower", "x_core_ci_lower"),
        ("vortex_centre_0_x_ci_upper", "x_core_ci_upper"),
        ("core_radius_0_ci_lower", "a_c_ci_lower"),
        ("core_radius_0_ci_upper", "a_c_ci_upper"),
    ):
        output[key] = (
            data[column].to_numpy(float)
            if column in data
            else np.full(len(data), np.nan, dtype=float)
        )
    return output


def viscous_filament_velocity(
    time: np.ndarray,
    circulation: float,
    separation: float,
    kinematic_viscosity: float,
    vortex_age: float,
    column_length: float,
    sample_plane_fraction: float = 0.25,
) -> np.ndarray:
    """Translation speed induced by one finite Lamb--Oseen filament."""
    time = np.asarray(time, dtype=float)
    if min(separation, kinematic_viscosity, vortex_age, column_length) <= 0.0:
        raise ValueError("separation, viscosity, vortex age, and column length must be positive")
    if np.any(time < 0.0):
        raise ValueError("time must be non-negative")
    sample_z = sample_plane_fraction * column_length
    lower = -0.5 * column_length - sample_z
    upper = 0.5 * column_length - sample_z
    endpoint_factor = upper / np.sqrt(separation**2 + upper**2) - lower / np.sqrt(
        separation**2 + lower**2
    )
    diffusion_time = vortex_age + time
    core_factor = 1.0 - np.exp(-(separation**2) / (4.0 * kinematic_viscosity * diffusion_time))
    return circulation * endpoint_factor * core_factor / (4.0 * np.pi * separation)


def theoretical_dipole_trajectory(
    time: np.ndarray,
    circulation: float,
    separation: float,
    kinematic_viscosity: float,
    vortex_age: float,
    column_length: float,
    sample_plane_fraction: float = 0.25,
) -> np.ndarray:
    """Analytical fixed-spacing finite-filament dipole trajectory."""
    time = np.asarray(time, dtype=float)
    if np.any(time < 0.0):
        raise ValueError("time must be non-negative")
    endpoint_speed = viscous_filament_velocity(
        np.zeros(1),
        circulation,
        separation,
        kinematic_viscosity,
        vortex_age,
        column_length,
        sample_plane_fraction,
    )[0]
    inverse_diffusion_time = separation**2 / (4.0 * kinematic_viscosity)

    def antiderivative(value: np.ndarray) -> np.ndarray:
        argument = -inverse_diffusion_time / value
        return value * np.exp(argument) + inverse_diffusion_time * expi(argument)

    return endpoint_speed * (
        time
        - antiderivative(time + vortex_age)
        + antiderivative(np.asarray(vortex_age, dtype=float))
    )


def uniform_cadence_mask(step: np.ndarray) -> np.ndarray:
    """Keep rows on the dominant diagnostic cadence."""
    if step.size < 2:
        return np.ones_like(step, dtype=bool)
    deltas = np.diff(step)
    positive = deltas[deltas > 0]
    if not positive.size:
        return np.ones_like(step, dtype=bool)
    cadence = int(np.median(positive))
    return (step - step[0]) % cadence == 0


def extract_merging_timeseries(
    samples_dir: Path,
    scheme: str,
    kinematic_viscosity: float,
    vortex_separation: float,
    core_radius: float,
) -> dict | None:
    """Return merger features on ``nu*t/a_c0^2`` through the literature horizon."""
    path = samples_dir / f"merging_{scheme}" / "field_diagnostics.csv"
    try:
        data = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["time", "step"])
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"  [merging] skipping unreadable live CSV for {scheme!r}: {exc}")
        return None
    data = data.sort_values("step").drop_duplicates("step", keep="last")
    if data.empty:
        return None
    data = data[uniform_cadence_mask(data["step"].to_numpy(int))]
    tau = kinematic_viscosity * data["time"].to_numpy(float) / core_radius**2
    # Retain the first sample beyond three so discrete deterministic cadences
    # visibly demonstrate complete coverage of the requested endpoint.
    beyond = np.flatnonzero(tau >= MERGING_NORMALIZED_END_TIME)
    stop = int(beyond[0] + 1) if beyond.size else len(data)
    data = data.iloc[:stop]
    tau = tau[:stop]

    angle = data["angle_radians"].to_numpy(float)
    finite = np.isfinite(angle)
    angle_degrees = np.full_like(angle, np.nan)
    if finite.any():
        unwrapped = unwrap_pair_orientation(angle[finite])
        angle_degrees[finite] = np.degrees(unwrapped - unwrapped[0])
    output = {
        "tau": tau,
        "theta_deg": angle_degrees,
        "a_c2_over_b02": data["mean_core_radius"].to_numpy(float) ** 2 / vortex_separation**2,
        "b_over_b0": data["vortex_separation"].to_numpy(float) / vortex_separation,
        "is_pair_unresolved": data["is_pair_unresolved"]
        .astype(str)
        .str.lower()
        .isin(("true", "1"))
        .to_numpy(bool),
        "orientation_anisotropy": data.get(
            "orientation_anisotropy", pd.Series(np.full(len(data), np.nan), index=data.index)
        ).to_numpy(float),
    }
    theta_half_width = np.full(len(data), np.nan)
    if {"angle_radians_ci_lower", "angle_radians_ci_upper"}.issubset(data.columns):
        theta_half_width = 0.5 * np.degrees(
            data["angle_radians_ci_upper"].to_numpy(float)
            - data["angle_radians_ci_lower"].to_numpy(float)
        )
    output["theta_ci_lower"] = angle_degrees - theta_half_width
    output["theta_ci_upper"] = angle_degrees + theta_half_width
    for base, key, transform in (
        ("mean_core_radius", "a_c2_over_b02", lambda value: value**2 / vortex_separation**2),
        ("vortex_separation", "b_over_b0", lambda value: value / vortex_separation),
    ):
        for bound in ("lower", "upper"):
            column = f"{base}_ci_{bound}"
            output[f"{key}_ci_{bound}"] = (
                transform(data[column].to_numpy(float))
                if column in data
                else np.full(len(data), np.nan)
            )
    if scheme == "rwm":
        # A nonlinear feature is not reportable merely because its point
        # estimate is finite. Suppress it when the delete-one-member interval
        # is unavailable (phase classification differs across replicates) or
        # is too broad to identify the feature. Post-coalescence b=0 remains a
        # topological property of the ensemble-mean field.
        theta_half_width = 0.5 * (output["theta_ci_upper"] - output["theta_ci_lower"])
        theta_reliable = np.isfinite(theta_half_width) & (theta_half_width <= 45.0)
        core_half_width = 0.5 * (
            output["a_c2_over_b02_ci_upper"] - output["a_c2_over_b02_ci_lower"]
        )
        core_reliable = (
            np.isfinite(core_half_width)
            & (output["a_c2_over_b02_ci_lower"] >= 0.0)
            & (core_half_width <= 0.5 * np.maximum(output["a_c2_over_b02"], 1.0e-12))
        )
        separation_half_width = 0.5 * (output["b_over_b0_ci_upper"] - output["b_over_b0_ci_lower"])
        separation_reliable = output["is_pair_unresolved"] | (
            np.isfinite(separation_half_width)
            & (output["b_over_b0_ci_lower"] >= 0.0)
            & (separation_half_width <= 0.5 * np.maximum(output["b_over_b0"], 1.0e-12))
        )
        for key in ("theta_deg", "theta_ci_lower", "theta_ci_upper"):
            output[key] = np.where(theta_reliable, output[key], np.nan)
        for key in ("a_c2_over_b02", "a_c2_over_b02_ci_lower", "a_c2_over_b02_ci_upper"):
            output[key] = np.where(core_reliable, output[key], np.nan)
        output["b_over_b0"] = np.where(separation_reliable, output["b_over_b0"], np.nan)
        for key in ("b_over_b0_ci_lower", "b_over_b0_ci_upper"):
            output[key] = np.where(
                separation_reliable & ~output["is_pair_unresolved"], output[key], np.nan
            )
    return output


def load_merging_references(core_radius: float, vortex_separation: float) -> dict[str, np.ndarray]:
    """Load the Re=530 reference histories on the simulation time coordinate."""
    scale = (core_radius / vortex_separation) ** 2
    output = {}
    for name, path in (("theta", THETA_REFERENCE), ("core", CORE_REFERENCE)):
        if path.is_file():
            values = np.loadtxt(path, delimiter=",")
            output[name] = np.column_stack((values[:, 0] / scale, values[:, 1]))
    if SEPARATION_DIMENSIONAL_REFERENCE.is_file():
        values = np.loadtxt(SEPARATION_DIMENSIONAL_REFERENCE, delimiter=",")
        paper_viscous_time = values[:, 0] * REFERENCE_VISCOUS_TIME_PER_SECOND
        output["separation"] = np.column_stack((paper_viscous_time / scale, values[:, 1]))
    return output


def _direct_energy_rate_mask(data_frame: pd.DataFrame) -> np.ndarray:
    """Identify dE/dt samples formed from one consistent unbounded energy."""
    if "kinetic_energy_rate_source" in data_frame:
        source = data_frame["kinetic_energy_rate_source"].astype(str).to_numpy()
        return source == "direct_energy_backward_difference"
    if "n_particles_total" in data_frame:
        # Historical files predate the explicit provenance column. The default
        # crossover used for these tutorial runs was 50,000 particles.
        return data_frame["n_particles_total"].to_numpy(float) <= DIRECT_ENERGY_PARTICLE_LIMIT
    return np.ones(len(data_frame), dtype=bool)


def read_flow_integrals(csv_path: Path) -> dict | None:
    """Load a validated, non-initialized flow-integral history."""
    try:
        data_frame = pd.read_csv(csv_path, on_bad_lines="skip").dropna(subset=["time"])
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"  [energy] skipping unreadable live CSV {csv_path.name}: {exc}")
        return None
    if data_frame.empty or "kinetic_energy_rate" not in data_frame:
        return None
    kinetic_energy_rate = data_frame["kinetic_energy_rate"].to_numpy(float)
    reportable_energy_rate = _direct_energy_rate_mask(data_frame)
    kinetic_energy_rate = np.where(reportable_energy_rate, kinetic_energy_rate, np.nan)

    data = {
        "time": data_frame["time"].to_numpy(float),
        "kinetic_energy_rate": kinetic_energy_rate,
        "viscous_kinetic_energy_rate": data_frame["viscous_kinetic_energy_rate"].to_numpy(float),
    }
    for measure in ("kinetic_energy_rate", "viscous_kinetic_energy_rate"):
        for bound in ("lower", "upper"):
            column = f"{measure}_ci_{bound}"
            if column in data_frame:
                data[column] = data_frame[column].to_numpy(float)
    nonzero = np.flatnonzero(
        (np.isfinite(data["kinetic_energy_rate"]) & (data["kinetic_energy_rate"] != 0.0))
        | (
            np.isfinite(data["viscous_kinetic_energy_rate"])
            & (data["viscous_kinetic_energy_rate"] != 0.0)
        )
    )
    if not nonzero.size:
        return None
    first = int(nonzero[0])
    if first:
        print(f"  [energy] skipping {first} leading zero dE/dt sample(s) in {csv_path.parent.name}")
        data = {key: values[first:] for key, values in data.items()}
    return data


def prepend_initial_point(
    data: dict,
    circulation: float,
    t0: float,
    n_vortices: int,
    column_length: float,
) -> dict:
    """Add the exact Lamb--Oseen initial power when solver diagnostics start later."""
    if len(data["time"]) == 0 or data["time"][0] == 0.0:
        return data
    initial_power = -n_vortices * circulation**2 * column_length / (8.0 * np.pi * t0)
    output = {key: np.insert(values, 0, 0.0) for key, values in data.items()}
    for key in output:
        output[key][0] = 0.0 if key == "time" else initial_power
    return output


def _quadrant_split(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    return int(np.searchsorted(y[:, 0], 0.0)), int(np.searchsorted(x[0], 0.0))


def _boundary_edges(field: np.ndarray, row: int, column: int, tx: float, ty: float):
    boundary_column = field[:, column - 1] * (1.0 - tx) + field[:, column] * tx
    boundary_row = field[row - 1, :] * (1.0 - ty) + field[row, :] * ty
    corner = boundary_row[column - 1] * (1.0 - tx) + boundary_row[column] * tx
    return boundary_column, boundary_row, corner


def _tile_coordinates(
    x: np.ndarray, y: np.ndarray, quadrant: str, column: int, row: int
) -> tuple[np.ndarray, np.ndarray]:
    if quadrant == "TL":
        return np.append(x[:column], 0.0), np.insert(y[row:], 0, 0.0)
    if quadrant == "TR":
        return np.insert(x[column:], 0, 0.0), np.insert(y[row:], 0, 0.0)
    if quadrant == "BL":
        return np.append(x[:column], 0.0), np.append(y[:row], 0.0)
    if quadrant == "BR":
        return np.insert(x[column:], 0, 0.0), np.append(y[:row], 0.0)
    raise ValueError(f"unknown quadrant {quadrant!r}")


def _tile_field(field, quadrant, column, row, boundary_column, boundary_row, corner):
    if quadrant == "TL":
        tile = np.column_stack([field[row:, :column], boundary_column[row:, None]])
        return np.vstack([np.append(boundary_row[:column], corner)[None, :], tile])
    if quadrant == "TR":
        tile = np.column_stack([boundary_column[row:, None], field[row:, column:]])
        return np.vstack([np.insert(boundary_row[column:], 0, corner)[None, :], tile])
    if quadrant == "BL":
        tile = np.column_stack([field[:row, :column], boundary_column[:row, None]])
        return np.vstack([tile, np.append(boundary_row[:column], corner)[None, :]])
    if quadrant == "BR":
        tile = np.column_stack([boundary_column[:row, None], field[:row, column:]])
        return np.vstack([tile, np.insert(boundary_row[column:], 0, corner)[None, :]])
    raise ValueError(f"unknown quadrant {quadrant!r}")


def surface_plot_tiles(
    samples_dir: Path,
    layout: list[tuple],
    core_radius: float,
    velocity_scale: float,
    vorticity_scale: float,
) -> tuple[list[dict], float | None]:
    """Return normalized, seam-free quadrant tiles at one common physical time."""
    comparison_time = latest_common_time(samples_dir, "vortex")
    tiles = []
    selected_times = []
    for scheme, quadrant, *rest in layout:
        timeline = pvd_time_map(samples_dir, "vortex", scheme)
        if not timeline:
            continue
        steps = sorted(
            timeline,
            key=lambda step: (
                abs(timeline[step] - comparison_time)
                if comparison_time is not None
                else -timeline[step],
                timeline[step] > comparison_time if comparison_time is not None else False,
            ),
        )
        path = next(
            (
                samples_dir / f"vortex_{scheme}" / f"vortex_{scheme}_zq_{step:06d}.vts"
                for step in steps
                if (
                    samples_dir / f"vortex_{scheme}" / f"vortex_{scheme}_zq_{step:06d}.vts"
                ).is_file()
            ),
            None,
        )
        if path is None:
            continue
        try:
            field = read_surface_field(path)
        except Exception as exc:
            print(f"  [surface] read error {path.name}: {exc}")
            continue
        x = field["x"].T / core_radius
        y = field["y"].T / core_radius
        velocity = np.hypot(field["velocity_x"], field["velocity_y"]).T / velocity_scale
        vorticity = np.clip(field["vorticity_z"].T, 0.0, None) / vorticity_scale
        row, column = _quadrant_split(x, y)
        x_axis, y_axis = x[0], y[:, 0]
        tx = -x_axis[column - 1] / (x_axis[column] - x_axis[column - 1])
        ty = -y_axis[row - 1] / (y_axis[row] - y_axis[row - 1])
        grid_x, grid_y = np.meshgrid(*_tile_coordinates(x_axis, y_axis, quadrant, column, row))
        tiled = {}
        for name, values in (("velocity", velocity), ("vorticity", vorticity)):
            tiled[name] = _tile_field(
                values,
                quadrant,
                column,
                row,
                *_boundary_edges(values, row, column, tx, ty),
            )
        selected_time = timeline[int(path.stem[-6:])]
        selected_times.append(selected_time)
        tiles.append({"scheme": scheme, "quadrant": quadrant, "x": grid_x, "y": grid_y, **tiled})
    if selected_times:
        print(
            f"  [surface] plotting {len(tiles)}/{len(SCHEMES)} methods at common "
            f"t={comparison_time:.3g}s (selected samples {min(selected_times):.3g}-"
            f"{max(selected_times):.3g}s)"
        )
    return tiles, comparison_time


from tutorials.vpm.lamb_oseen_vortex.setup import (
    COLUMN_LENGTH,
    CORE_RADIUS,
    FIELD_SPACING,
    MAX_PARTICLES,
    SPACING,
    TIME_STEP_SIZE,
    TOTAL_TIME as CONFIGURED_TOTAL_TIME,
)


STEP_RE = re.compile(r"_(\d{6})\.h5$")
PHYSICS_CASES = ("vortex", "dipole", "merging")
MINIMUM_ENSEMBLE_SIZE = 4
CONFIDENCE_LEVEL = 0.95
JACKKNIFE_COLUMNS = (
    "vortex_centre_0_x",
    "vortex_centre_0_y",
    "vortex_centre_1_x",
    "vortex_centre_1_y",
    "vortex_separation",
    "core_radius_0",
    "core_radius_1",
    "mean_core_radius",
    "angle_radians",
    "peak_saddle_contrast",
)


@dataclass(frozen=True)
class Member:
    index: int
    seed: int
    solution_dir: Path
    samples_dir: Path
    metadata: dict
    backups: dict[int, Path]


@dataclass
class DiagnosticState:
    previous_centres: list[np.ndarray] | None = None
    pair_lost: bool = False
    coalesced: bool = False


def _backup_map(folder: Path) -> dict[int, Path]:
    result = {}
    for path in sorted(folder.glob("*.h5")):
        match = STEP_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def discover_members(
    solution_root: Path,
    samples_root: Path,
    case_name: str,
    expected_members: int | None = None,
) -> list[Member]:
    member_re = re.compile(rf"^{re.escape(case_name)}_(\d+)$")
    members: list[Member] = []
    for solution_dir in sorted(solution_root.glob(f"{case_name}_*")):
        match = member_re.fullmatch(solution_dir.name)
        if not solution_dir.is_dir() or match is None:
            continue
        index = int(match.group(1))
        member_samples = Path(samples_root) / solution_dir.name
        metadata_path = member_samples / "run_metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"unreadable RWM member metadata {metadata_path}: {error}") from error
        if metadata.get("status") != "complete" or metadata.get("completed") is not True:
            raise ValueError(f"RWM member {case_name}/{solution_dir.name} is not complete")
        backups = _backup_map(solution_dir)
        if not backups:
            raise ValueError(f"RWM member {case_name}/{solution_dir.name} has no backups")
        members.append(
            Member(
                index=index,
                seed=int(metadata["random_seed"]),
                solution_dir=solution_dir,
                samples_dir=member_samples,
                metadata=metadata,
                backups=backups,
            )
        )

    if len(members) < MINIMUM_ENSEMBLE_SIZE:
        raise ValueError(
            f"{case_name}: need at least {MINIMUM_ENSEMBLE_SIZE} independent members, "
            f"found {len(members)}"
        )
    if expected_members is not None and len(members) != expected_members:
        raise ValueError(f"{case_name}: expected {expected_members} members, found {len(members)}")
    indices = [member.index for member in members]
    if indices != list(range(len(members))):
        raise ValueError(f"{case_name}: ensemble member indices are not contiguous from zero")
    seeds = [member.seed for member in members]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{case_name}: random seeds are not unique")
    reference_steps = set(members[0].backups)
    for member in members[1:]:
        if set(member.backups) != reference_steps:
            raise ValueError(f"{case_name}: backup steps differ between ensemble members")
    return members


def _template_grid(samples_root: Path, physics: str, metadata: dict) -> dict:
    folder = Path(samples_root) / f"{physics}_cs"
    candidates = sorted(folder.glob(f"{physics}_cs_zq_*.vts"))
    if not candidates:
        raise FileNotFoundError(f"missing deterministic grid template under {folder}")
    template = read_surface_field(candidates[0])
    if physics != "vortex":
        return template

    # The historical single-vortex surface window was sized from the initial
    # core and truncates several percent of the late diffused circulation.
    # Build a wider uniform reconstruction grid from the configured diffusion
    # scale.  Plotters already select their common comparison window.
    dx = float(np.median(np.diff(template["x"][:, 0])))
    core_radius = float(metadata["core_radius"])
    kinematic_viscosity = float(metadata["kinematic_viscosity"])
    end_time = float(metadata["end_time"])
    final_gaussian_radius = math.sqrt(core_radius**2 + 4.0 * kinematic_viscosity * end_time)
    brownian_standard_deviation = math.sqrt(2.0 * kinematic_viscosity * end_time)
    old_half_width = max(float(np.max(np.abs(template["x"]))), float(np.max(np.abs(template["y"]))))
    half_width = max(
        4.0 * final_gaussian_radius,
        old_half_width + 4.0 * brownian_standard_deviation,
    )
    half_cells = int(math.ceil(half_width / dx))
    axis = np.arange(-half_cells, half_cells + 1, dtype=float) * dx
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return {"x": x, "y": y}


def _deposit_circulation_cic(
    x: np.ndarray,
    y: np.ndarray,
    position: np.ndarray,
    circulation_per_length: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Conservative cloud-in-cell deposition on the canonical uniform grid."""
    x_axis = np.asarray(x[:, 0], dtype=float)
    y_axis = np.asarray(y[0, :], dtype=float)
    dx = float(np.median(np.diff(x_axis)))
    dy = float(np.median(np.diff(y_axis)))
    nx, ny = x.shape
    fx = (position[:, 0] - x_axis[0]) / dx
    fy = (position[:, 1] - y_axis[0]) / dy
    inside = (fx >= 0.0) & (fx <= nx - 1) & (fy >= 0.0) & (fy <= ny - 1)
    total_absolute = float(np.sum(np.abs(circulation_per_length)))
    captured_absolute = float(np.sum(np.abs(circulation_per_length[inside])))
    capture_fraction = captured_absolute / max(total_absolute, np.finfo(float).tiny)

    fx = fx[inside]
    fy = fy[inside]
    values = circulation_per_length[inside]
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    i0 = np.clip(i0, 0, nx - 2)
    j0 = np.clip(j0, 0, ny - 2)
    tx = np.clip(fx - i0, 0.0, 1.0)
    ty = np.clip(fy - j0, 0.0, 1.0)
    deposited = np.zeros((nx, ny), dtype=np.float64)
    for di, dj, weight in (
        (0, 0, (1.0 - tx) * (1.0 - ty)),
        (1, 0, tx * (1.0 - ty)),
        (0, 1, (1.0 - tx) * ty),
        (1, 1, tx * ty),
    ):
        np.add.at(deposited, (i0 + di, j0 + dj), values * weight)
    return deposited / (dx * dy), capture_fraction


def _biot_savart_velocity(
    vorticity: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Free-space 2-D Biot--Savart convolution of a gridded vorticity field."""
    nx, ny = vorticity.shape
    offset_x = np.arange(-(nx - 1), nx, dtype=float) * dx
    offset_y = np.arange(-(ny - 1), ny, dtype=float) * dy
    rx, ry = np.meshgrid(offset_x, offset_y, indexing="ij")
    radius_squared = rx * rx + ry * ry
    kernel_x = np.zeros_like(radius_squared)
    kernel_y = np.zeros_like(radius_squared)
    nonzero = radius_squared > 0.0
    kernel_x[nonzero] = -ry[nonzero] / (2.0 * np.pi * radius_squared[nonzero])
    kernel_y[nonzero] = rx[nonzero] / (2.0 * np.pi * radius_squared[nonzero])
    area = dx * dy
    velocity_x = signal.fftconvolve(vorticity, kernel_x, mode="same") * area
    velocity_y = signal.fftconvolve(vorticity, kernel_y, mode="same") * area
    return velocity_x, velocity_y


def project_backup(path: Path, template: dict, column_length: float) -> tuple[dict, dict]:
    """Return the column-projected 2-D field represented by one backup."""
    with h5py.File(path, "r") as handle:
        position = np.asarray(handle["particles/position"], dtype=np.float64)
        strength = np.asarray(handle["particles/vortex_strength"], dtype=np.float64)
        core_radius = np.asarray(handle["particles/core_radius"], dtype=np.float64)
        step = int(handle["solver"].attrs["step"])
        time = float(handle["solver"].attrs["time"])
    if position.shape != strength.shape or position.shape[1:] != (3,):
        raise ValueError(f"{path}: invalid particle position/strength shape")
    if np.ptp(core_radius) > 1.0e-6 * max(float(np.mean(core_radius)), np.finfo(float).tiny):
        raise ValueError(f"{path}: projected RWM estimator requires a common Gaussian core radius")

    x = template["x"]
    y = template["y"]
    dx = float(np.median(np.diff(x[:, 0])))
    dy = float(np.median(np.diff(y[0, :])))
    circulation = strength[:, 2] / column_length
    positive = np.clip(circulation, 0.0, None)
    negative = np.clip(circulation, None, 0.0)
    deposited_positive, _ = _deposit_circulation_cic(x, y, position, positive)
    deposited_negative, _ = _deposit_circulation_cic(x, y, position, negative)
    blob_radius = float(np.mean(core_radius))
    # OpenONDA's Gaussian blob is exp(-r^2/sigma^2)/(pi sigma^2) after
    # integrating the 3-D kernel along z.  scipy's Gaussian standard deviation
    # is therefore sigma/sqrt(2).
    filter_options = {
        "sigma": (
            blob_radius / (math.sqrt(2.0) * dx),
            blob_radius / (math.sqrt(2.0) * dy),
        ),
        "mode": "constant",
        "cval": 0.0,
        "truncate": 4.5,
    }
    vorticity_positive = ndimage.gaussian_filter(deposited_positive, **filter_options)
    vorticity_negative = ndimage.gaussian_filter(deposited_negative, **filter_options)
    vorticity_z = vorticity_positive + vorticity_negative
    represented_absolute_circulation = float(
        (np.sum(vorticity_positive) - np.sum(vorticity_negative)) * dx * dy
    )
    requested_absolute_circulation = float(np.sum(np.abs(circulation)))
    capture_fraction = represented_absolute_circulation / max(
        requested_absolute_circulation, np.finfo(float).tiny
    )
    velocity_x, velocity_y = _biot_savart_velocity(vorticity_z, dx, dy)
    return (
        {
            "x": x,
            "y": y,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "vorticity_z": vorticity_z,
            "velocity_gradient_yx": np.gradient(velocity_y, x[:, 0], axis=0),
            "step": step,
            "time": time,
        },
        {
            "absolute_circulation_capture_fraction": capture_fraction,
            "particle_count": len(position),
            "particle_core_radius": blob_radius,
        },
    )


def _stack_mean_field(fields: list[dict], confidence_multiplier: float) -> tuple[dict, dict]:
    velocity = np.stack(
        [np.stack((field["velocity_x"], field["velocity_y"]), axis=-1) for field in fields]
    )
    vorticity = np.stack([field["vorticity_z"] for field in fields])
    gradient = np.stack([field["velocity_gradient_yx"] for field in fields])
    n_members = len(fields)
    mean_velocity = velocity.mean(axis=0)
    mean_vorticity = vorticity.mean(axis=0)
    velocity_se = velocity.std(axis=0, ddof=1) / math.sqrt(n_members)
    vorticity_se = vorticity.std(axis=0, ddof=1) / math.sqrt(n_members)
    gradient_se = gradient.std(axis=0, ddof=1) / math.sqrt(n_members)
    field = {
        "x": fields[0]["x"],
        "y": fields[0]["y"],
        "velocity_x": mean_velocity[..., 0],
        "velocity_y": mean_velocity[..., 1],
        "vorticity_z": mean_vorticity,
        "velocity_standard_error_x": velocity_se[..., 0],
        "velocity_standard_error_y": velocity_se[..., 1],
        "vorticity_standard_error_z": vorticity_se,
        "velocity_gradient_yx": gradient.mean(axis=0),
        "velocity_gradient_yx_standard_error": gradient_se,
        "ensemble_size": float(n_members),
        "confidence_multiplier": confidence_multiplier,
        "step": fields[0]["step"],
        "time": fields[0]["time"],
    }
    norm_velocity = np.linalg.norm(mean_velocity)
    norm_vorticity = np.linalg.norm(mean_vorticity)
    half = max(1, n_members // 2)
    diagnostics = {
        "relative_standard_error_l2_velocity": float(np.linalg.norm(velocity_se) / norm_velocity),
        "relative_standard_error_l2_vorticity": float(
            np.linalg.norm(vorticity_se) / norm_vorticity
        ),
        "half_ensemble_relative_difference_velocity": float(
            np.linalg.norm(velocity[:half].mean(axis=0) - mean_velocity) / norm_velocity
        ),
        "half_ensemble_relative_difference_vorticity": float(
            np.linalg.norm(vorticity[:half].mean(axis=0) - mean_vorticity) / norm_vorticity
        ),
    }
    field["_member_velocity"] = velocity
    field["_member_vorticity"] = vorticity
    field["_member_gradient"] = gradient
    return field, diagnostics


def _leave_one_out_field(field: dict, omitted: int, confidence_multiplier: float) -> dict:
    velocity = np.delete(field["_member_velocity"], omitted, axis=0)
    vorticity = np.delete(field["_member_vorticity"], omitted, axis=0)
    gradient = np.delete(field["_member_gradient"], omitted, axis=0)
    n_members = len(velocity)
    velocity_se = velocity.std(axis=0, ddof=1) / math.sqrt(n_members)
    vorticity_se = vorticity.std(axis=0, ddof=1) / math.sqrt(n_members)
    return {
        "x": field["x"],
        "y": field["y"],
        "velocity_x": velocity[..., 0].mean(axis=0),
        "velocity_y": velocity[..., 1].mean(axis=0),
        "vorticity_z": vorticity.mean(axis=0),
        "velocity_standard_error_x": velocity_se[..., 0],
        "velocity_standard_error_y": velocity_se[..., 1],
        "vorticity_standard_error_z": vorticity_se,
        "velocity_gradient_yx": gradient.mean(axis=0),
        "ensemble_size": float(n_members),
        "confidence_multiplier": confidence_multiplier,
        "step": field["step"],
        "time": field["time"],
    }


def _advance_diagnostic(field: dict, physics: str, state: DiagnosticState) -> list:
    if physics == "merging" and state.coalesced:
        row = diagnostics_row(field, "merging", force_merged=True)
    else:
        row = diagnostics_row(field, physics, state.previous_centres)
    coalesced_index = FIELD_CSV_COLUMNS.index("is_peak_coalesced")
    if physics == "merging" and state.pair_lost and not bool(row[coalesced_index]):
        row = _mask_lost_pair_features(row)
    if physics == "merging" and bool(row[coalesced_index]):
        state.coalesced = True
        state.previous_centres = None
    elif physics == "merging" and bool(row[11]):
        state.pair_lost = True
    elif np.isfinite(row[2:6]).all():
        state.previous_centres = [np.asarray(row[2:4]), np.asarray(row[4:6])]
    elif np.isfinite(row[2:4]).all():
        state.previous_centres = [np.asarray(row[2:4])]
    return row


def _jackknife_record(
    estimate: list,
    leave_one_out: list[list],
    confidence_multiplier: float,
) -> dict:
    record = dict(zip(FIELD_CSV_COLUMNS, estimate, strict=True))
    loo = np.asarray(leave_one_out, dtype=object)
    n_members = len(leave_one_out)
    for name in JACKKNIFE_COLUMNS:
        index = FIELD_CSV_COLUMNS.index(name)
        values = np.asarray(loo[:, index], dtype=float)
        point = float(estimate[index])
        if not np.isfinite(point) or not np.isfinite(values).all():
            standard_error = lower = upper = float("nan")
        else:
            if name == "angle_radians":
                differences = (values - point + 0.5 * np.pi) % np.pi - 0.5 * np.pi
                centred = differences - differences.mean()
            else:
                centred = values - values.mean()
            standard_error = float(
                np.sqrt((n_members - 1.0) / n_members * np.sum(centred * centred))
            )
            lower = point - confidence_multiplier * standard_error
            upper = point + confidence_multiplier * standard_error
        record[f"{name}_standard_error"] = standard_error
        record[f"{name}_ci_lower"] = lower
        record[f"{name}_ci_upper"] = upper
    record["ensemble_size"] = n_members
    record["pair_resolved_leave_one_out_fraction"] = float(
        np.mean([not bool(row[11]) for row in leave_one_out])
    )
    return record


def _write_vts(path: Path, field: dict, sample_z: float) -> None:
    import pyvista as pv

    x = field["x"]
    y = field["y"]
    z = np.full_like(x, sample_z)
    grid = pv.StructuredGrid(x, y, z)

    def vector(x_component, y_component, z_component=None):
        if z_component is None:
            z_component = np.zeros_like(x_component)
        return np.column_stack(
            (
                np.asarray(x_component).ravel(order="F"),
                np.asarray(y_component).ravel(order="F"),
                np.asarray(z_component).ravel(order="F"),
            )
        ).astype(np.float32)

    grid.point_data["velocity"] = vector(field["velocity_x"], field["velocity_y"])
    grid.point_data["vorticity"] = vector(
        np.zeros_like(field["vorticity_z"]),
        np.zeros_like(field["vorticity_z"]),
        field["vorticity_z"],
    )
    grid.point_data["velocity_standard_error"] = vector(
        field["velocity_standard_error_x"], field["velocity_standard_error_y"]
    )
    grid.point_data["vorticity_standard_error"] = vector(
        np.zeros_like(field["vorticity_standard_error_z"]),
        np.zeros_like(field["vorticity_standard_error_z"]),
        field["vorticity_standard_error_z"],
    )
    grid.point_data["velocity_gradient_yx"] = np.asarray(
        field["velocity_gradient_yx"], dtype=np.float32
    ).ravel(order="F")
    grid.point_data["velocity_gradient_yx_standard_error"] = np.asarray(
        field["velocity_gradient_yx_standard_error"], dtype=np.float32
    ).ravel(order="F")
    grid.field_data["ensemble_size"] = np.array([field["ensemble_size"]], dtype=np.int32)
    grid.field_data["confidence_multiplier"] = np.array(
        [field["confidence_multiplier"]], dtype=np.float64
    )
    temporary = path.with_name(f".{path.stem}.tmp.vts")
    grid.save(temporary, binary=True)
    temporary.replace(path)


def _write_pvd(path: Path, entries: list[tuple[int, float]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    stem = path.stem
    for step, time in entries:
        lines.append(f'    <DataSet timestep="{time:.12g}" file="{stem}_{step:06d}.vts"/>')
    lines.extend(("  </Collection>", "</VTKFile>"))
    temporary = path.with_suffix(".pvd.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _aggregate_flow_integrals(members: list[Member], output: Path, multiplier: float) -> None:
    frames = [pd.read_csv(member.samples_dir / "flow_integrals.csv") for member in members]
    common_steps = sorted(set.intersection(*(set(frame["step"].astype(int)) for frame in frames)))
    if not common_steps:
        raise ValueError(f"{output.parent.name}: no common flow-integral steps")
    aligned = [frame.set_index("step").loc[common_steps].reset_index() for frame in frames]
    result_columns: dict[str, object] = {"step": common_steps}
    for column in aligned[0].columns:
        if column == "step":
            continue
        series = [frame[column] for frame in aligned]
        if all(pd.api.types.is_numeric_dtype(item) for item in series):
            values = np.stack([item.to_numpy(float) for item in series])
            mean = values.mean(axis=0)
            standard_error = values.std(axis=0, ddof=1) / math.sqrt(len(members))
            result_columns[column] = mean
            if column != "time":
                result_columns[f"{column}_standard_error"] = standard_error
                result_columns[f"{column}_ci_lower"] = mean - multiplier * standard_error
                result_columns[f"{column}_ci_upper"] = mean + multiplier * standard_error
        else:
            result_columns[column] = series[0].astype(str).to_numpy()
    result = pd.DataFrame(result_columns)
    result.to_csv(output, index=False)


def aggregate_case(
    solution_root: Path,
    samples_root: Path,
    physics: str,
    expected_members: int | None = None,
) -> dict:
    case_name = f"{physics}_rwm"
    members = discover_members(solution_root, samples_root, case_name, expected_members)
    metadata0 = members[0].metadata
    template = _template_grid(samples_root, physics, metadata0)
    column_length = 2.0 * float(metadata0["column_half_length"])
    sample_z = 0.25 * column_length
    n_members = len(members)
    multiplier = float(stats.t.ppf(0.5 + 0.5 * CONFIDENCE_LEVEL, n_members - 1))
    loo_multiplier = float(stats.t.ppf(0.5 + 0.5 * CONFIDENCE_LEVEL, n_members - 2))
    output_dir = Path(samples_root) / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_state = DiagnosticState()
    loo_states = [DiagnosticState() for _ in members]
    feature_records = []
    convergence_records = []
    pvd_entries = []
    stochastic_distinctness_checked = False
    minimum_capture = 1.0

    steps = sorted(members[0].backups)
    for step in steps:
        member_fields = []
        projection_qa = []
        for member in members:
            field, qa = project_backup(member.backups[step], template, column_length)
            member_fields.append(field)
            projection_qa.append(qa)
        times = np.asarray([field["time"] for field in member_fields])
        if not np.allclose(times, times[0], rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{case_name}: physical times disagree at step {step}")

        aggregate, convergence = _stack_mean_field(member_fields, multiplier)
        if step > 0 and not stochastic_distinctness_checked:
            first = aggregate["_member_vorticity"][0]
            if any(
                np.array_equal(first, aggregate["_member_vorticity"][index])
                for index in range(1, n_members)
            ):
                raise ValueError(
                    f"{case_name}: at least two nonzero-time members are identical; "
                    "the backend did not produce independent seeded walks"
                )
            stochastic_distinctness_checked = True

        estimate = _advance_diagnostic(aggregate, physics, diagnostic_state)
        loo_rows = []
        for omitted, state in enumerate(loo_states):
            loo_field = _leave_one_out_field(aggregate, omitted, loo_multiplier)
            loo_rows.append(_advance_diagnostic(loo_field, physics, state))
        feature_records.append(_jackknife_record(estimate, loo_rows, multiplier))

        capture_values = [item["absolute_circulation_capture_fraction"] for item in projection_qa]
        minimum_capture = min(minimum_capture, *capture_values)
        convergence_records.append(
            {
                "time": aggregate["time"],
                "step": step,
                "ensemble_size": n_members,
                **convergence,
                "minimum_absolute_circulation_capture_fraction": min(capture_values),
                "mean_absolute_circulation_capture_fraction": float(np.mean(capture_values)),
            }
        )
        output_vts = output_dir / f"{case_name}_zq_{step:06d}.vts"
        _write_vts(output_vts, aggregate, sample_z)
        pvd_entries.append((step, aggregate["time"]))
        print(
            f"  [RWM] {case_name}: step {step:06d}, t={aggregate['time']:.4g}, "
            f"relative MCSE(omega)={convergence['relative_standard_error_l2_vorticity']:.3%}"
        )

    if not stochastic_distinctness_checked:
        raise ValueError(f"{case_name}: no nonzero backup available to verify member independence")
    _write_pvd(output_dir / f"{case_name}_zq.pvd", pvd_entries)
    pd.DataFrame(feature_records).to_csv(output_dir / "field_diagnostics.csv", index=False)
    pd.DataFrame(convergence_records).to_csv(output_dir / "rwm_convergence.csv", index=False)
    _aggregate_flow_integrals(members, output_dir / "flow_integrals.csv", multiplier)

    metadata = dict(metadata0)
    metadata.pop("ensemble_member", None)
    metadata.pop("random_seed", None)
    metadata.update(
        {
            "status": "complete",
            "completed": True,
            "case": physics,
            "scheme": "rwm",
            "statistical_estimator": "fixed_time_seed_ensemble_mean_of_column_projected_fields",
            "raw_output_estimator": "particle_backups_for_column_projection",
            "ensemble_size": n_members,
            "ensemble_member_indices": [member.index for member in members],
            "random_seeds": [member.seed for member in members],
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_multiplier": multiplier,
            "column_projection": (
                "omega_bar_z(x,y)=L^-1 integral omega_z(x,y,z) dz, reconstructed from "
                "Gaussian particle backups; velocity from free-space 2-D Biot-Savart"
            ),
            "feature_definitions": {
                "vortex_centre": "centre_of_connected_area_inside_80_percent_peak_vorticity_contour",
                "vortex_separation": (
                    "distance_between_vorticity_centres_before_peak_coalescence; zero thereafter"
                ),
                "orientation": (
                    "undirected_axis_joining_centres before merger; vorticity-quadrupole "
                    "major axis of merged ellipse after merger"
                ),
                "core_radius": (
                    "radius_of_maximum_outward_semicircle_mean_azimuthal_velocity_before_merger; "
                    "full_circle_mean_after_merger"
                ),
                "pair_resolution": "peak_to_saddle_contrast_exceeds_95_percent_ensemble_uncertainty",
            },
            "uncertainty": (
                "pointwise Student-t intervals across independent seeds; delete-one-member "
                "jackknife intervals for nonlinear extracted features"
            ),
            "minimum_absolute_circulation_capture_fraction": minimum_capture,
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def aggregate_rwm_ensemble(
    solution_root: Path,
    samples_root: Path,
    expected_members: int | dict[str, int] | None = None,
) -> dict[str, dict]:
    """Aggregate all three benchmark physics cases and return their metadata."""
    return {
        physics: aggregate_case(
            solution_root,
            samples_root,
            physics,
            expected_members.get(physics)
            if isinstance(expected_members, dict)
            else expected_members,
        )
        for physics in PHYSICS_CASES
    }


EXPECTED_END_TIME = CONFIGURED_TOTAL_TIME
EXPECTED_DT = 0.291 / 9.0


# =============================================================
# Validation helpers
# =============================================================


def _read_csv(
    path: Path,
    failures: list[str],
    *,
    allow_nonfinite: bool = False,
) -> pd.DataFrame | None:
    if not path.is_file():
        failures.append(f"missing {path}")
        return None
    try:
        data = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        failures.append(f"unreadable {path}: {error}")
        return None
    numeric = data.select_dtypes(include=[np.number])
    if (
        data.empty
        or numeric.empty
        or (not allow_nonfinite and not np.isfinite(numeric.to_numpy()).all())
    ):
        failures.append(f"{path}: empty or non-finite numeric data")
    return data


def merging_normalization_audit(
    samples_dir: Path = SAMPLES_DIR,
    schemes: tuple[str, ...] = SCHEMES,
) -> tuple[dict, list[str]]:
    """Audit the paper-to-simulation mapping and requested merger horizon."""
    failures: list[str] = []
    reference = load_merging_references(CORE_RADIUS, SEPARATION)
    reference_horizons = {name: float(values[:, 0].max()) for name, values in reference.items()}
    for name, horizon in reference_horizons.items():
        if not 2.9 <= horizon <= 3.2:
            failures.append(f"merging reference {name}: normalized horizon {horizon:.6g} is not 3")

    run_report = {}
    for scheme in schemes:
        folder = Path(samples_dir) / f"merging_{scheme}"
        metadata = _metadata(folder / "run_metadata.json")
        if not metadata:
            failures.append(f"merging_{scheme}: missing metadata for normalization audit")
            continue
        circulation = abs(float(metadata.get("circulations", [np.nan])[0]))
        viscosity = float(metadata.get("kinematic_viscosity", np.nan))
        a_c0 = float(metadata.get("velocity_peak_radius", np.nan))
        b0 = float(metadata.get("vortex_separation", np.nan))
        reynolds = circulation / viscosity
        core_ratio = a_c0 / b0
        physical_time_at_three = MERGING_NORMALIZED_END_TIME * a_c0**2 / viscosity
        if not np.isclose(reynolds, REYNOLDS_NUMBER, rtol=1.0e-10):
            failures.append(f"merging_{scheme}: Re_Gamma={reynolds:.9g}, expected 530")
        if not np.isclose(core_ratio, CORE_RADIUS / SEPARATION, rtol=1.0e-10):
            failures.append(f"merging_{scheme}: a_c0/b0={core_ratio:.9g}, expected 0.125")
        try:
            fields = pd.read_csv(folder / "field_diagnostics.csv")
            normalized_time = viscosity * fields["time"].to_numpy(float) / a_c0**2
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            failures.append(f"merging_{scheme}: normalization data unreadable ({exc})")
            continue
        final_normalized_time = float(np.nanmax(normalized_time))
        if final_normalized_time < MERGING_NORMALIZED_END_TIME:
            failures.append(f"merging_{scheme}: normalized horizon {final_normalized_time:.6g} < 3")
        first_at_or_after = np.flatnonzero(normalized_time >= MERGING_NORMALIZED_END_TIME)
        endpoint = (
            float(normalized_time[first_at_or_after[0]])
            if first_at_or_after.size
            else final_normalized_time
        )
        coalesced = fields.get("is_peak_coalesced")
        if coalesced is not None:
            coalesced = coalesced.astype(str).str.lower().isin(("true", "1")).to_numpy(bool)
            if coalesced.any():
                separation = fields["vortex_separation"].to_numpy(float)
                if not np.allclose(separation[coalesced], 0.0, rtol=0.0, atol=1.0e-12):
                    failures.append(f"merging_{scheme}: b is not zero after peak coalescence")
        run_report[scheme] = {
            "circulation_reynolds_number": reynolds,
            "initial_velocity_core_to_separation_ratio": core_ratio,
            "time_scale_seconds": a_c0**2 / viscosity,
            "physical_time_at_normalized_3": physical_time_at_three,
            "available_normalized_horizon": final_normalized_time,
            "first_sample_at_or_after_3": endpoint,
        }
        timeseries = extract_merging_timeseries(Path(samples_dir), scheme, viscosity, b0, a_c0)
        agreement = {}
        if timeseries is not None:
            unresolved = np.asarray(timeseries["is_pair_unresolved"], dtype=bool)
            first_unresolved = np.flatnonzero(unresolved)
            separation_history = np.asarray(timeseries["b_over_b0"], dtype=float)
            first_zero = np.flatnonzero(
                np.isfinite(separation_history)
                & np.isclose(separation_history, 0.0, rtol=0.0, atol=1.0e-12)
            )
            finite_theta = np.flatnonzero(np.isfinite(timeseries["theta_deg"]))
            run_report[scheme]["feature_transition"] = {
                "first_statistically_unresolved_pair_time": (
                    float(timeseries["tau"][first_unresolved[0]]) if first_unresolved.size else None
                ),
                "first_zero_peak_separation_time": (
                    float(timeseries["tau"][first_zero[0]]) if first_zero.size else None
                ),
                "last_reportable_theta_time": (
                    float(timeseries["tau"][finite_theta[-1]]) if finite_theta.size else None
                ),
                "theta_definition": (
                    "axis joining the two vorticity centres before coalescence; "
                    "major-axis orientation of the merged elliptic vortex thereafter"
                ),
            }
            for feature, reference_name in (
                ("theta_deg", "theta"),
                ("a_c2_over_b02", "core"),
                ("b_over_b0", "separation"),
            ):
                if reference_name not in reference:
                    continue
                reference_values = reference[reference_name]
                x = timeseries["tau"]
                y = timeseries[feature]
                comparable = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & (x >= reference_values[:, 0].min())
                    & (x <= reference_values[:, 0].max())
                )
                if not comparable.any():
                    continue
                expected = np.interp(x[comparable], reference_values[:, 0], reference_values[:, 1])
                residual = y[comparable] - expected
                value_range = float(np.ptp(reference_values[:, 1]))
                rmse = float(np.sqrt(np.mean(residual * residual)))
                normalized_rmse = rmse / value_range
                agreement[feature] = {
                    "comparable_samples": int(np.count_nonzero(comparable)),
                    "rmse": rmse,
                    "rmse_over_reference_range": normalized_rmse,
                    "mean_bias": float(np.mean(residual)),
                    "assessment": (
                        "good"
                        if normalized_rmse <= 0.10
                        else "moderate"
                        if normalized_rmse <= 0.25
                        else "poor"
                    ),
                }
        run_report[scheme]["reference_agreement"] = agreement
    report = {
        "displayed_time": "nu*t/a_c0^2",
        "paper_time": "nu*t/b0^2",
        "paper_to_display_factor": (SEPARATION / CORE_RADIUS) ** 2,
        "reference_normalized_horizons": reference_horizons,
        "separation_reference_status": "plotted from the figure 4 b/b0 measurements",
        "separation_time_conversion": {
            "figure_4_final_time_seconds": REFERENCE_FINAL_TIME_SECONDS,
            "figure_5_final_viscous_time": REFERENCE_FINAL_VISCOUS_TIME,
            "viscous_time_per_second": REFERENCE_VISCOUS_TIME_PER_SECOND,
            "basis": ("common final acquisition of the same Re=530 experiment and timescale"),
        },
        "runs": run_report,
    }
    return report, failures


def single_vortex_error_audit(
    samples_dir: Path = SAMPLES_DIR,
    schemes: tuple[str, ...] = SCHEMES,
) -> dict:
    """Profile and core-growth errors against the exact Lamb--Oseen solution."""
    timelines = {scheme: pvd_time_map(samples_dir, "vortex", scheme) for scheme in schemes}
    latest = min(max(values.values()) for values in timelines.values() if values)
    runtime = resolve_runtime_physics(samples_dir, 1.0, 1.0 / 530.0, 1.0, 0.125 / 1.12)
    output: dict[str, dict] = {}
    for scheme in schemes:
        profile = load_profile(samples_dir, scheme, latest)
        if profile is None:
            raise ValueError(f"vortex_{scheme}: no readable common-time field profile")
        x, velocity, vorticity, selected_time = profile
        exact_velocity, exact_vorticity, _ = lamb_oseen_profile(
            x,
            runtime["t0"] + selected_time,
            runtime["circulation"],
            runtime["kinematic_viscosity"],
        )
        exact_gradient = lamb_oseen_gradient(
            x,
            runtime["t0"] + selected_time,
            runtime["circulation"],
            runtime["kinematic_viscosity"],
        )
        numerical_gradient = np.gradient(velocity, x)
        window = np.abs(x / runtime["velocity_peak_radius0"]) <= 5.5
        errors = [
            float(np.linalg.norm((numerical - exact)[window]) / np.linalg.norm(exact[window]))
            for numerical, exact in (
                (velocity, exact_velocity),
                (vorticity, exact_vorticity),
                (numerical_gradient, exact_gradient),
            )
        ]

        field_path = samples_dir / f"vortex_{scheme}" / "field_diagnostics.csv"
        fields = pd.read_csv(field_path, on_bad_lines="skip").dropna(subset=["time"])
        fields = fields.sort_values("time")
        fields = fields[fields["time"] <= EXPECTED_END_TIME + EXPECTED_DT]
        time = fields["time"].to_numpy(float)
        measured_core = fields["mean_core_radius"].to_numpy(float)
        exact_core = BETA_RMAX * np.sqrt(
            GAUSSIAN_CORE_RADIUS**2 + 4.0 * runtime["kinematic_viscosity"] * time
        )
        relative_core_error = (measured_core - exact_core) / exact_core
        finite = np.isfinite(relative_core_error)
        if not finite.any():
            raise ValueError(f"vortex_{scheme}: no finite core-radius history")
        output[scheme] = {
            "profile_time": float(selected_time),
            "relative_l2_velocity": errors[0],
            "relative_l2_vorticity": errors[1],
            "relative_l2_velocity_gradient": errors[2],
            "core_radius_comparable_samples": int(finite.sum()),
            "core_radius_relative_rmse": float(np.sqrt(np.mean(relative_core_error[finite] ** 2))),
            "core_radius_relative_mean_bias": float(np.mean(relative_core_error[finite])),
            "core_radius_relative_maximum_absolute_error": float(
                np.max(np.abs(relative_core_error[finite]))
            ),
            "core_radius_final_relative_error": float(relative_core_error[finite][-1]),
        }
    return {
        "common_profile_target_time": float(latest),
        "profile_window": "abs(x/a_c0) <= 5.5",
        "runs": output,
    }


def _single_vortex_errors(
    schemes: tuple[str, ...] = SCHEMES,
) -> dict[str, tuple[float, float, float]]:
    """Compatibility view of the final common single-vortex profile errors."""
    report = single_vortex_error_audit(SAMPLES_DIR, schemes)
    return {
        scheme: (
            values["relative_l2_velocity"],
            values["relative_l2_vorticity"],
            values["relative_l2_velocity_gradient"],
        )
        for scheme, values in report["runs"].items()
    }


def dipole_error_audit(
    samples_dir: Path = SAMPLES_DIR,
    schemes: tuple[str, ...] = SCHEMES,
) -> dict:
    """Compare translating-pair motion on one common physical-time window."""
    runtime = resolve_runtime_physics(
        samples_dir, 1.0, 1.0 / 530.0, 1.0, 0.125 / 1.12, prefix="dipole"
    )
    frames = {}
    for scheme in schemes:
        path = samples_dir / f"dipole_{scheme}" / "field_diagnostics.csv"
        frame = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["time"])
        frame = frame.sort_values("time").drop_duplicates("time", keep="last")
        if frame.empty:
            raise ValueError(f"dipole_{scheme}: no field diagnostics")
        frame = frame[uniform_cadence_mask(frame["step"].to_numpy(int))]
        frames[scheme] = frame
    common_end = min(
        EXPECTED_END_TIME,
        *(float(frame["time"].max()) for frame in frames.values()),
    )
    exact_core_end = BETA_RMAX * np.sqrt(
        GAUSSIAN_CORE_RADIUS**2 + 4.0 * runtime["kinematic_viscosity"] * common_end
    )
    reference_end = theoretical_dipole_trajectory(
        np.asarray([common_end]),
        runtime["circulation"],
        runtime["vortex_separation"],
        runtime["kinematic_viscosity"],
        runtime["t0"],
        runtime["column_length"],
    )[0]
    runs = {}
    for scheme, frame in frames.items():
        time = frame["time"].to_numpy(float)
        trajectory = frame["vortex_centre_0_x"].to_numpy(float)
        core_radius = frame["mean_core_radius"].to_numpy(float)
        separation = frame["vortex_separation"].to_numpy(float)
        comparable = (time <= common_end + 1.0e-12) & np.isfinite(trajectory)
        reference = theoretical_dipole_trajectory(
            time[comparable],
            runtime["circulation"],
            runtime["vortex_separation"],
            runtime["kinematic_viscosity"],
            runtime["t0"],
            runtime["column_length"],
        )
        reference_range = float(np.ptp(reference))
        trajectory_nrmse = float(
            np.sqrt(np.mean((trajectory[comparable] - reference) ** 2)) / reference_range
        )
        end_trajectory = float(np.interp(common_end, time, trajectory))
        end_core = float(np.interp(common_end, time, core_radius))
        end_separation = float(np.interp(common_end, time, separation))
        runs[scheme] = {
            "comparable_samples": int(comparable.sum()),
            "trajectory_rmse_over_reference_range": trajectory_nrmse,
            "end_trajectory_over_a_c0": end_trajectory / runtime["velocity_peak_radius0"],
            "end_trajectory_relative_error": (end_trajectory - reference_end) / reference_end,
            "end_separation_over_b0": end_separation / runtime["vortex_separation"],
            "end_core_radius_over_a_c0": end_core / runtime["velocity_peak_radius0"],
            "end_core_radius_relative_to_isolated_exact": (end_core - exact_core_end)
            / exact_core_end,
        }
    return {
        "common_end_time": float(common_end),
        "reference": "fixed-separation finite Lamb--Oseen filaments",
        "runs": runs,
    }


def energy_balance_audit(
    samples_dir: Path = SAMPLES_DIR,
    schemes: tuple[str, ...] = SCHEMES,
) -> dict:
    """Separate comparable direct-energy rates from changing-box estimates."""
    runs = {}
    for case_id in CASES:
        for scheme in schemes:
            name = f"{case_id}_{scheme}"
            path = samples_dir / name / "flow_integrals.csv"
            frame = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["time"])
            direct = _direct_energy_rate_mask(frame)
            energy_rate = frame["kinetic_energy_rate"].to_numpy(float)
            viscous_rate = frame["viscous_kinetic_energy_rate"].to_numpy(float)
            nonzero = np.isfinite(energy_rate) & (energy_rate != 0.0)
            comparable = direct & nonzero & np.isfinite(viscous_rate)
            raw_positive = nonzero & (energy_rate > 1.0e-7)
            comparable_positive = comparable & (energy_rate > 1.0e-7)
            residual = energy_rate[comparable] - viscous_rate[comparable]
            denominator = float(np.sqrt(np.mean(viscous_rate[comparable] ** 2)))
            runs[name] = {
                "total_rate_samples": int(nonzero.sum()),
                "direct_comparable_samples": int(comparable.sum()),
                "raw_positive_samples": int(raw_positive.sum()),
                "direct_positive_samples": int(comparable_positive.sum()),
                "all_positive_samples_outside_direct_definition": bool(
                    raw_positive.any() and not np.any(raw_positive & direct)
                ),
                "relative_rms_balance_residual": (
                    float(np.sqrt(np.mean(residual**2)) / denominator)
                    if comparable.any() and denominator > 0.0
                    else None
                ),
                "direct_comparable_end_time": (
                    float(frame.loc[comparable, "time"].max()) if comparable.any() else None
                ),
            }
    return {
        "finite_difference_definition": (
            "backward difference of consecutive direct unbounded kinetic-energy integrals"
        ),
        "dynamic_fourier_box_mode": (
            "instantaneous energy and -nu*Omega remain reportable; dE/dt is undefined"
        ),
        "legacy_backend_inference_particle_limit": DIRECT_ENERGY_PARTICLE_LIMIT,
        "runs": runs,
    }


def _solver_log_record(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    cumulative = [
        float(value)
        for value in re.findall(
            r"^\s*cumulative\s+([0-9.]+(?:e[+-]?\d+)?)\s+s\s*$",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    ]
    if not cumulative:
        return None

    def first(pattern: str) -> str | None:
        match = re.search(pattern, text, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    return {
        "solver_cumulative_seconds": cumulative[-1],
        "backend": first(r"^\s*backend\s+(\S+)\s*$"),
        "host": first(r"^\s*host\s+(.+?)\s*$"),
        "platform": first(r"^\s*platform\s+(.+?)\s*$"),
    }


def runtime_audit(solution_dir: Path = SOLUTION_DIR) -> dict:
    """Summarize recorded solver time without conflating different hardware."""
    runs = {}
    for case_id in CASES:
        for scheme in ("cs", "gbd", "dvh"):
            name = f"{case_id}_{scheme}"
            record = _solver_log_record(solution_dir / name / f"vpm_{name}.log")
            if record is not None:
                runs[name] = record

        name = f"{case_id}_rwm"
        member_records = [
            record
            for path in sorted((solution_dir).glob(f"{name}_*/vpm_*.log"))
            if (record := _solver_log_record(path)) is not None
        ]
        if member_records:
            seconds = np.asarray(
                [record["solver_cumulative_seconds"] for record in member_records],
                dtype=float,
            )
            runs[name] = {
                "ensemble_members": int(seconds.size),
                "ensemble_total_solver_seconds": float(seconds.sum()),
                "member_mean_solver_seconds": float(seconds.mean()),
                "member_median_solver_seconds": float(np.median(seconds)),
                "member_minimum_solver_seconds": float(seconds.min()),
                "member_maximum_solver_seconds": float(seconds.max()),
                "backend": member_records[0]["backend"],
                "host": member_records[0]["host"],
                "platform": member_records[0]["platform"],
            }
    environments = {(record.get("backend"), record.get("host")) for record in runs.values()}
    return {
        "definition": (
            "last cumulative solver-step time recorded in each solver log; "
            "external post-processing is excluded"
        ),
        "cross_scheme_wall_time_comparable": len(environments) == 1,
        "comparison_note": (
            "CS, GBD, and DVH are mutually comparable on the recorded Metal host; "
            "RWM was run on a different CPU host and may only be compared by member "
            "and ensemble cost within that environment."
        ),
        "runs": runs,
    }


def validate(pre_plot: bool, schemes: tuple[str, ...] = SCHEMES) -> int:
    failures: list[str] = []
    for physics_id in CASES:
        for scheme in schemes:
            name = f"{physics_id}_{scheme}"
            folder = SAMPLES_DIR / name
            metadata_path = folder / "run_metadata.json"
            if not metadata_path.is_file():
                failures.append(f"{name}: missing run_metadata.json")
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                failures.append(f"{name}: unreadable metadata ({error})")
                continue
            if metadata.get("status") != "complete" or metadata.get("completed") is not True:
                failures.append(f"{name}: metadata is not complete")
            final_time = float(metadata.get("final_time", np.nan))
            if final_time < EXPECTED_END_TIME - EXPECTED_DT:
                failures.append(
                    f"{name}: final time {final_time:.9g} does not cover {EXPECTED_END_TIME:.9g}"
                )

            integrals = _read_csv(folder / "flow_integrals.csv", failures)
            if integrals is not None:
                if "time" not in integrals:
                    failures.append(f"{name}: flow-integral history has no time column")
                    continue
                integral_time = integrals["time"].to_numpy(float)
                positive_cadence = np.diff(integral_time)
                positive_cadence = positive_cadence[positive_cadence > 0.0]
                cadence = (
                    float(np.median(positive_cadence)) if positive_cadence.size else EXPECTED_DT
                )
                if integrals["time"].iloc[-1] < EXPECTED_END_TIME - cadence - EXPECTED_DT:
                    failures.append(f"{name}: flow-integral history is incomplete")
                if np.any(np.diff(integral_time) <= 0.0):
                    failures.append(f"{name}: flow-integral time is not strictly increasing")
                for column in ("kinetic_energy_rate", "viscous_kinetic_energy_rate"):
                    if column not in integrals:
                        continue
                    tested_column = column
                    if scheme == "rwm" and f"{column}_ci_lower" in integrals:
                        # A positive sample mean is not a physical violation when
                        # its confidence interval still includes zero.  Fail only
                        # when the modeled rate is significantly positive.
                        tested_column = f"{column}_ci_lower"
                    tested_values = integrals[tested_column].to_numpy(float)
                    if column == "kinetic_energy_rate":
                        tested_values = tested_values[_direct_energy_rate_mask(integrals)]
                    if np.any(tested_values > 1.0e-7):
                        failures.append(
                            f"{name}: significantly positive modeled energy rate in {column}"
                        )

            fields = _read_csv(folder / "field_diagnostics.csv", failures, allow_nonfinite=True)
            if fields is not None:
                required = {"time", "step", "core_radius_0", "mean_core_radius"}
                if not required.issubset(fields.columns):
                    failures.append(
                        f"{name}: missing field columns {sorted(required - set(fields.columns))}"
                    )
                else:
                    if np.any(fields["core_radius_0"].to_numpy(float) <= 0.0):
                        failures.append(f"{name}: non-positive extracted core radius")
                    field_time = fields["time"].to_numpy(float)
                    field_cadence = np.diff(field_time)
                    field_cadence = field_cadence[field_cadence > 0.0]
                    allowed_gap = (
                        float(np.median(field_cadence)) if field_cadence.size else EXPECTED_DT
                    )
                    if fields["time"].iloc[-1] < EXPECTED_END_TIME - allowed_gap - EXPECTED_DT:
                        failures.append(f"{name}: field diagnostics are incomplete")
                boundary_columns = [column for column in fields if "boundary_limited" in column]
                if any(bool(fields[column].astype(bool).any()) for column in boundary_columns):
                    failures.append(f"{name}: extracted core radius is boundary limited")

            if scheme == "rwm":
                ensemble_size = int(metadata.get("ensemble_size", 0))
                seeds = metadata.get("random_seeds", [])
                if ensemble_size < 4 or len(set(seeds)) != ensemble_size:
                    failures.append(f"{name}: invalid or non-independent RWM ensemble metadata")
                if metadata.get("statistical_estimator") != (
                    "fixed_time_seed_ensemble_mean_of_column_projected_fields"
                ):
                    failures.append(f"{name}: missing column-projected ensemble estimator")
                convergence = _read_csv(folder / "rwm_convergence.csv", failures)
                if convergence is not None:
                    for column in (
                        "relative_standard_error_l2_velocity",
                        "relative_standard_error_l2_vorticity",
                    ):
                        if column not in convergence:
                            failures.append(f"{name}: missing convergence column {column}")
                        elif float(convergence[column].max()) > 0.075:
                            failures.append(
                                f"{name}: {column} exceeds the 7.5% Monte Carlo uncertainty limit"
                            )
                    capture = convergence.get("minimum_absolute_circulation_capture_fraction")
                    if capture is None or float(capture.min()) < 0.995:
                        failures.append(
                            f"{name}: column projection loses more than 0.5% circulation"
                        )
                if fields is not None:
                    uncertainty_columns = {
                        "core_radius_0_standard_error",
                        "vortex_centre_0_x_standard_error",
                    }
                    if not uncertainty_columns.issubset(fields.columns):
                        failures.append(
                            f"{name}: missing feature uncertainty columns "
                            f"{sorted(uncertainty_columns - set(fields.columns))}"
                        )
                    if name == "merging_rwm" and "is_pair_unresolved" in fields:
                        unresolved = (
                            fields["is_pair_unresolved"].astype(str).str.lower().isin(("true", "1"))
                        )
                        if not unresolved.any():
                            failures.append(f"{name}: statistically resolved pair never merges")
                        elif np.any(np.diff(unresolved.to_numpy(int)) < 0):
                            failures.append(f"{name}: pair resolution resurrects after merger")

            if not list(folder.glob("*_zq.pvd")):
                failures.append(f"{name}: missing sampled surface-field PVD")

    normalization, normalization_failures = merging_normalization_audit(SAMPLES_DIR, schemes)
    failures.extend(normalization_failures)
    for scheme, values in normalization["runs"].items():
        print(
            f"merging_{scheme}: Re_Gamma={values['circulation_reynolds_number']:.6g}, "
            f"a_c0/b0={values['initial_velocity_core_to_separation_ratio']:.6g}, "
            f"t(3)={values['physical_time_at_normalized_3']:.6g}s, "
            f"coverage={values['available_normalized_horizon']:.6g}"
        )
        for feature, agreement in values["reference_agreement"].items():
            print(
                f"  {feature}: RMSE/reference-range="
                f"{agreement['rmse_over_reference_range']:.3%}, "
                f"bias={agreement['mean_bias']:.6g}"
            )

    try:
        errors = _single_vortex_errors(schemes)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        failures.append(f"single-vortex analytic comparison failed: {error}")
    else:
        for scheme, (velocity, vorticity, gradient) in errors.items():
            if scheme not in schemes:
                continue
            print(
                f"vortex_{scheme}: analytic relative L2 "
                f"velocity={velocity:.3%}, vorticity={vorticity:.3%}, gradient={gradient:.3%}"
            )
            if scheme in {"cs", "rwm"} and max(velocity, vorticity, gradient) > 0.20:
                failures.append(f"vortex_{scheme}: analytic profile error exceeds 20%")
            if scheme != "cs" and max(velocity, vorticity, gradient) > 2.0:
                failures.append(f"vortex_{scheme}: analytic profile error exceeds 200%")

    if not pre_plot:
        for fig_name in (
            "vortex_comparison",
            "dipole_comparison",
            "merging_comparison",
            "vortex_surface_fields",
            "lamboseen_energy",
        ):
            figure = FIGURES_DIR / f"{fig_name}.png"
            if not figure.is_file() or figure.stat().st_size == 0:
                failures.append(f"missing or empty figure {figure.name}")

    if failures:
        print("\n".join(f"[FAIL] {failure}" for failure in failures))
        return 1
    print("[OK] lamb_oseen_vortex certification passed")
    return 0


# =============================================================
# Manifest generation
# =============================================================


def _metadata(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _last_time(path: Path, column: str) -> tuple[int, float | None]:
    try:
        frame = pd.read_csv(path, on_bad_lines="skip")
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return 0, None
    return len(frame), (float(values.max()) if not values.empty else None)


def _observed_time_step(path: Path) -> float | None:
    try:
        frame = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["time", "step"])
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None
    delta_time = np.diff(frame["time"].to_numpy(float))
    delta_step = np.diff(frame["step"].to_numpy(float))
    valid = (delta_time > 0.0) & (delta_step > 0.0)
    return float(np.median(delta_time[valid] / delta_step[valid])) if valid.any() else None


def _quality_warnings(
    scheme: str,
    metadata: dict,
    max_particles: float | None,
    observed_time_step: float | None,
) -> list[str]:
    warnings = []
    if scheme == "rwm" and metadata:
        if int(metadata.get("ensemble_size", 0)) < 8:
            warnings.append("RWM ensemble has fewer than eight independent members.")
    cap = MAX_PARTICLES if scheme in ("dvh", "gbd") else None
    if cap and max_particles is not None and max_particles >= float(cap):
        warnings.append(
            f"{scheme.upper()} reached its particle-count guard; inspect late-time sensitivity."
        )
    requested_end = float(metadata.get("end_time", np.nan))
    final_time = float(metadata.get("final_time", np.nan))
    tolerance = observed_time_step if observed_time_step is not None else EXPECTED_DT
    if (
        np.isfinite(requested_end)
        and np.isfinite(final_time)
        and final_time > requested_end + tolerance
    ):
        warnings.append(
            "Archival run exceeds the requested end time; all comparisons are truncated "
            "to the declared common physical-time window."
        )
    return warnings


def build_manifest(samples_dir: Path, figures_dir: Path) -> dict:
    runs = {}
    for case_id in CASES:
        for scheme in SCHEMES:
            name = f"{case_id}_{scheme}"
            folder = samples_dir / name
            metadata = _metadata(folder / "run_metadata.json")
            field_rows, field_time = _last_time(folder / "field_diagnostics.csv", "time")
            integral_rows, integral_time = _last_time(folder / "flow_integrals.csv", "time")
            _, max_particles = _last_time(folder / "flow_integrals.csv", "n_particles_total")
            observed_time_step = _observed_time_step(folder / "field_diagnostics.csv")
            has_samples = field_rows > 0 or integral_rows > 0 or any(folder.glob("*_zq_*.vts"))
            complete = metadata.get("completed") is True or metadata.get("status") == "complete"
            if complete:
                status = "complete"
            elif metadata or has_samples:
                status = str(metadata.get("status", "partial"))
            else:
                status = "missing"
            runs[name] = {
                "status": status,
                "complete": complete,
                "field_rows": field_rows,
                "last_field_time": field_time,
                "integral_rows": integral_rows,
                "last_integral_time": integral_time,
                "requested_end_time": metadata.get("end_time"),
                "final_time": metadata.get("final_time"),
                "core_radius_definition": (
                    "radius_of_maximum_outward_semicircle_mean_azimuthal_velocity"
                ),
                "vortex_centre_definition": (
                    "centre_of_connected_area_inside_80_percent_peak_vorticity_contour"
                ),
                "sample_plane_z": 0.25 * COLUMN_LENGTH,
                "particle_spacing_ratio": SPACING / CORE_RADIUS,
                "field_spacing_ratio": FIELD_SPACING / CORE_RADIUS,
                "max_n_particles_sampled": max_particles,
                "requested_time_step_size": TIME_STEP_SIZE,
                "metadata_time_step_size": metadata.get("time_step_size"),
                "observed_time_step_size": observed_time_step,
                "integrator": metadata.get("integrator"),
                "induction_backend": metadata.get("induction_backend"),
                "strength_rate_formulation": metadata.get("strength_rate_formulation"),
                "particle_kernel": metadata.get("particle_kernel"),
                "precision": metadata.get("precision"),
                "max_particles_capacity": MAX_PARTICLES,
                "initial_n_particles_total": metadata.get("initial_n_particles_total"),
                "ensemble_size": metadata.get("ensemble_size"),
                "compute_device": metadata.get("compute_device"),
                "circulation_normalization": "per_vortex_after_strength_cutoff",
                "quality_warnings": _quality_warnings(
                    scheme, metadata, max_particles, observed_time_step
                ),
            }
    normalization, normalization_failures = merging_normalization_audit(samples_dir, SCHEMES)
    single_vortex = single_vortex_error_audit(samples_dir, SCHEMES)
    dipole = dipole_error_audit(samples_dir, SCHEMES)
    energy = energy_balance_audit(samples_dir, SCHEMES)
    timing = runtime_audit(SOLUTION_DIR)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plotting_notes": (
            "Figures plot every scientifically reportable sample; undefined rates from "
            "changing Fourier audit boxes are retained in raw archives but not plotted."
        ),
        "runs": runs,
        "normalization_audit": normalization,
        "normalization_failures": normalization_failures,
        "single_vortex_error_audit": single_vortex,
        "dipole_error_audit": dipole,
        "energy_rate_audit": energy,
        "runtime_audit": timing,
        "figures": sorted(
            path.name for path in figures_dir.iterdir() if path.suffix.lower() in {".png", ".pdf"}
        ),
    }


def write_manifest() -> int:
    manifest = build_manifest(SAMPLES_DIR, FIGURES_DIR)
    output = FIGURES_DIR / "postprocessing_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output)
    counts = {}
    for run in manifest["runs"].values():
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    print(f"  [status] {counts}; wrote {output}")
    return 0


# =============================================================
# CLI
# =============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract-fields",
        action="store_true",
        help="extract deterministic field features from all sampled VTS planes",
    )
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--case", default=None, help="limit --extract-fields to one case")
    parser.add_argument("--pre-plot", action="store_true", help="skip figure existence checks")
    parser.add_argument("--manifest", action="store_true", help="write JSON status manifest")
    parser.add_argument(
        "--aggregate-rwm",
        action="store_true",
        help="build canonical mean fields and uncertainty from raw RWM ensemble backups",
    )
    parser.add_argument(
        "--expected-rwm-members",
        type=int,
        default=None,
        help="require exactly this many complete RWM ensemble members",
    )
    for physics in CASES:
        parser.add_argument(
            f"--expected-rwm-{physics}-members",
            type=int,
            default=None,
            help=f"override the required complete RWM member count for {physics}",
        )
    parser.add_argument(
        "--rwm-only",
        action="store_true",
        help="certify only the RWM products (useful when deterministic baselines are archival)",
    )
    args = parser.parse_args()
    if args.extract_fields:
        extract_field_diagnostics(args.samples_dir, args.case)
        return 0
    if args.aggregate_rwm:
        expected_by_case = {
            physics: getattr(args, f"expected_rwm_{physics}_members")
            if getattr(args, f"expected_rwm_{physics}_members") is not None
            else args.expected_rwm_members
            for physics in CASES
        }
        aggregate_rwm_ensemble(
            CASE_DIR / "solution",
            SAMPLES_DIR,
            expected_members=expected_by_case,
        )
        return 0
    if args.manifest:
        return write_manifest()
    return validate(pre_plot=args.pre_plot, schemes=("rwm",) if args.rwm_only else SCHEMES)


if __name__ == "__main__":
    raise SystemExit(main())
