"""Post-processing for the Lamb--Oseen VPM tutorial.

This module owns sampled-field reading, physical feature definitions, RWM
ensemble reconstruction and uncertainty, theory/reference transformations,
and plot-ready data preparation. Plot scripts are
deliberately presentation-only.

  * vortex centres   — geometric centres of the connected areas enclosed by
    the 80%-of-peak vorticity contours, following Cerretelli--Williamson;
  * core radius a_c  — radius where the azimuthally-averaged tangential
    velocity |u_theta(r)| peaks, measured on the outward semicircle before
    merger and over the full circle after merger;
  * vortex_separation and orbital angle (dipole/merging pair).

Run ``python -m openonda.tutorial_runner . assets.postprocess --extract-fields`` to rebuild deterministic feature CSVs,
or aggregate independent random walks of the same initial field.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats
from scipy.special import expi

from openonda import plotting as theme
from openonda.results import (
    read_csv_table,
    read_history_table,
    read_json,
    read_numeric_table,
    read_pvd_frames,
    read_surface_frame,
    write_csv_table,
    write_text,
)
from source.solution_layout import vpm_backup_files
from source.solvers.vpm.io.ensemble import (
    field_ensemble,
    mean_standard_error,
    realization_metadata,
    table_ensemble,
)
from source.solvers.vpm.io.postprocess import (
    common_frame_time,
    particle_state,
    saved_frame,
)
from source.vtk_output import write_vtk_dataset

ASSETS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ASSETS_DIR.parent
CASE_DIR = SCRIPT_DIR
FIGURES_DIR = SCRIPT_DIR / "figures"
SAMPLES_DIR = SCRIPT_DIR / "samples"
SOLUTION_DIR = SCRIPT_DIR / "solution"
REF_DIR = ASSETS_DIR / "references"
SCHEMES = ("cs", "rwm", "dvh", "gbd")
SCHEME_DRAW_ORDER = tuple((scheme for scheme in SCHEMES if scheme != "cs")) + ("cs",)
_SCHEME_ZORDER = {
    scheme: 200 if scheme == "cs" else 10 + index for index, scheme in enumerate(SCHEME_DRAW_ORDER)
}
CASES = ("vortex", "dipole", "merging")
ENERGY_CASES = (
    ("vortex", "Single vortex", 1),
    ("dipole", "Vortex dipole", 2),
    ("merging", "Co-rotating", 2),
)
BETA_RMAX = 1.12
REFERENCE_CIRCULATION = 1.0
REYNOLDS_NUMBER = 530.0
CORE_RADIUS = 0.125
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX
SEPARATION = 1.0
COLUMN_LENGTH = 40.0 * CORE_RADIUS
FIELD_SPACING = 0.15 * CORE_RADIUS
TOTAL_TIME = 30.0
SPACING = 0.6 * CORE_RADIUS
VTS_STEP_RE = re.compile("_(\\d+)\\.vts$")
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


def _flatten_solver_metadata(metadata: dict) -> dict:
    """Read the current record and derive this column's physical length scales."""
    configuration = metadata["configuration"]
    numerics, state = configuration["numerics"], metadata["state"]
    conditions = configuration["initial_conditions"]
    first = conditions[0]
    distribution = first["distribution"]
    bounds = distribution["bounds"]
    centres = np.asarray([condition["centre"] for condition in conditions])
    spacing = distribution["spacing"]
    gaussian_core = first["vortex_core_radius"]
    step_size, requested_steps = numerics["time_step_size"], configuration["run"]["steps"]
    status = metadata["run_status"]["status"]
    return {
        "status": status,
        "completed": status == "completed",
        "case_name": metadata["case_name"],
        "time_step_size": step_size,
        "number_of_steps": requested_steps,
        "end_time": float(step_size) * int(requested_steps),
        "completed_steps": state["step"],
        "final_time": state["time"],
        "integrator": numerics["integrator"]["name"],
        "induction_backend": numerics["induction"]["method"],
        "stretching_scheme": numerics["induction"]["stretching_scheme"],
        "turbulence_model": numerics["turbulence"]["model"],
        "viscous_scheme": numerics["viscous"]["scheme"],
        "particle_kernel": numerics["particle_kernel"],
        "precision": numerics["precision"],
        "write_precision": numerics["write_precision"],
        "compute_device": numerics["compute_device"],
        "random_seed": numerics["random_seed"],
        "particle_spacing": spacing,
        "particle_core_radius": float(spacing) * float(distribution["core_radius_ratio"]),
        "core_radius": gaussian_core,
        "velocity_peak_radius": BETA_RMAX * float(gaussian_core),
        "kinematic_viscosity": first["kinematic_viscosity"],
        "column_half_length": 0.5 * (float(bounds[2][1]) - float(bounds[2][0])),
        "circulations": [condition["circulation"] for condition in conditions],
        "vortex_separation": float(np.linalg.norm(centres[0] - centres[1]))
        if len(centres) > 1
        else None,
        "initial_n_particles_total": state["initial_n_particles_total"],
        "final_n_particles_total": state["n_particles_total"],
    }


def _metadata(path: Path) -> dict:
    requested = Path(path)
    if requested.name == "vpm_metadata.json":
        native = requested
    else:
        solution = requested.parent.parent / "solution"
        name = requested.name
        native = (
            next(solution.glob(f"{name}_*/vpm_metadata.json"))
            if name.endswith("_rwm")
            else solution / name / "vpm_metadata.json"
        )
    return _flatten_solver_metadata(read_json(native))


def read_solver_metadata(samples_dir: Path, prefix: str = "vortex") -> dict:
    return _metadata(samples_dir / f"{prefix}_cs")


def resolve_runtime_physics(samples_dir: Path, prefix: str = "vortex") -> dict[str, float | None]:
    metadata = read_solver_metadata(samples_dir, prefix)
    core = float(metadata["core_radius"])
    viscosity = float(metadata["kinematic_viscosity"])
    return {
        "kinematic_viscosity": viscosity,
        "t0": core**2 / (4 * viscosity),
        "ac0": core,
        "velocity_peak_radius0": float(metadata["velocity_peak_radius"]),
        "circulation": abs(float(metadata["circulations"][0])),
        "vortex_separation": metadata["vortex_separation"],
        "column_length": 2 * float(metadata["column_half_length"]),
    }


def pvd_time_map(samples_dir: Path, prefix: str, scheme: str) -> dict[int, float]:
    index = samples_dir / f"{prefix}_{scheme}" / f"{prefix}_{scheme}_zq.pvd"
    return {
        int(VTS_STEP_RE.search(path.name).group(1)): time for time, path in read_pvd_frames(index)
    }


def read_surface_field(path: Path) -> dict:
    data = read_surface_frame(path)
    return {
        name: values.T if isinstance(values, np.ndarray) and values.ndim == 2 else values
        for name, values in data.items()
    }


def _subgrid_peak_centre(
    x: np.ndarray, y: np.ndarray, signed_vorticity: np.ndarray, peak_index: tuple[int, int]
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
    contour_fraction: float = 0.8,
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


def _peak_candidates(values: np.ndarray, min_relative_peak: float = 0.2):
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
    field: dict, candidates: list[tuple[int, int]], previous_centres: list[np.ndarray] | None
) -> list[tuple[int, int]]:
    """Select the physical two-peak branch using strength and continuity."""
    x, y, values = (field["x"], field["y"], field["vorticity_z"])
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
    field: dict, physics: str, previous_centres: list[np.ndarray] | None = None
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Vortex centres from connected 80%-of-peak vorticity regions."""
    x, y, wz = (field["x"], field["y"], field["vorticity_z"])
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
        return (centres, peaks)
    signed = np.abs(wz) if physics == "vortex" else wz
    candidates = _peak_candidates(signed, 0.05 if physics == "merging" else 0.2)
    if not candidates:
        return ([np.array([np.nan, np.nan])], [])
    if physics == "merging":
        peaks = _merging_peak_pair(field, candidates, previous_centres)
        return ([_high_vorticity_region_centre(x, y, signed, peak) for peak in peaks], peaks)
    centre = _high_vorticity_region_centre(x, y, signed, candidates[0])
    centres = [centre] if np.isfinite(centre).all() else [np.array([np.nan, np.nan])]
    return (centres, [candidates[0]])


def _pair_resolution_diagnostic(
    field: dict, peaks: list[tuple[int, int]], centres: list[np.ndarray]
) -> tuple[bool, float, float, float]:
    """Test whether two same-sign peaks are resolved above Monte Carlo noise.

    A pair is resolved only when it has two spatially distinct 80%-contour
    regions and the smaller peak rises above the intervening saddle by more
    than the ensemble confidence multiplier times the contrast standard error.
    Deterministic fields use a one-percent contrast floor in place of a Monte
    Carlo uncertainty estimate.
    """
    if len(peaks) != 2 or len(centres) != 2 or (not np.isfinite(centres).all()):
        return (False, float("nan"), float("nan"), float("nan"))
    wz = np.asarray(field["vorticity_z"], dtype=float)
    first, second = peaks
    if first == second or min(*first, *second) < 0:
        return (False, float("nan"), float("nan"), float("nan"))
    dx_values = np.diff(field["x"][:, 0])
    dy_values = np.diff(field["y"][0, :])
    grid_spacing = float(
        np.median(np.abs(np.concatenate((dx_values[dx_values != 0], dy_values[dy_values != 0]))))
    )
    centre_separation = float(np.linalg.norm(centres[0] - centres[1]))
    if centre_separation < 2.0 * grid_spacing:
        return (False, float("nan"), float("nan"), float("nan"))
    n_line = max(16, int(np.ceil(np.linalg.norm(np.subtract(first, second)))) * 4 + 1)
    line_i = np.linspace(first[0], second[0], n_line)
    line_j = np.linspace(first[1], second[1], n_line)
    line = ndimage.map_coordinates(wz, (line_i, line_j), order=1, mode="nearest")
    if line.size < 5 or not np.isfinite(line).all():
        return (False, float("nan"), float("nan"), float("nan"))
    interior = line[1:-1]
    saddle_offset = int(np.argmin(interior)) + 1
    saddle = float(line[saddle_offset])
    peak_values = np.array([float(wz[first]), float(wz[second])])
    contrast = float(np.min(peak_values) - saddle)
    se_field = field.get("vorticity_standard_error_z")
    if se_field is None:
        threshold = 0.01 * max(float(np.max(peak_values)), np.finfo(float).tiny)
        return (contrast > threshold, contrast, float("nan"), float("inf"))
    se_field = np.asarray(se_field, dtype=float)
    line_se = ndimage.map_coordinates(se_field, (line_i, line_j), order=1, mode="nearest")
    lower_peak = first if peak_values[0] <= peak_values[1] else second
    contrast_se = float(np.hypot(se_field[lower_peak], line_se[saddle_offset]))
    multiplier = float(field["confidence_multiplier"])
    peak_significant = all(
        (
            value > multiplier * se_field[index]
            for value, index in zip(peak_values, (first, second), strict=True)
        )
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
        return (float("nan"), float("nan"))
    wz = np.asarray(field["vorticity_z"], dtype=float)
    positive = np.clip(wz, 0.0, None)
    peak = float(np.nanmax(positive))
    if not np.isfinite(peak) or peak <= 0.0:
        return (float("nan"), float("nan"))
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
        return (float("nan"), float("nan"))
    dx = np.asarray(field["x"], dtype=float) - centre[0]
    dy = np.asarray(field["y"], dtype=float) - centre[1]
    qxx = float(np.sum(weights * dx * dx) / weight_sum)
    qyy = float(np.sum(weights * dy * dy) / weight_sum)
    qxy = float(np.sum(weights * dx * dy) / weight_sum)
    discriminant = float(np.hypot(qxx - qyy, 2.0 * qxy))
    trace = qxx + qyy
    if trace <= np.finfo(float).tiny:
        return (float("nan"), float("nan"))
    return (0.5 * float(np.arctan2(2.0 * qxy, qxx - qyy)), discriminant / trace)


def _match_centres_to_previous(
    centres: list[np.ndarray], previous_centres: list[np.ndarray] | None
) -> list[np.ndarray]:
    """Keep centre identities continuous without changing pair geometry."""
    if len(centres) != 2:
        return centres
    if previous_centres is None or len(previous_centres) != 2:
        return sorted(centres, key=lambda centre: (centre[1], centre[0]), reverse=True)
    direct = sum(
        (np.linalg.norm(a - b) ** 2 for a, b in zip(centres, previous_centres, strict=True))
    )
    swapped = sum(
        (np.linalg.norm(a - b) ** 2 for a, b in zip(centres[::-1], previous_centres, strict=True))
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
        return (float("nan"), False)
    x, y = (field["x"], field["y"])
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
        return (float("nan"), False)
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
    core_radius, magnitudes = ([], [])
    for b in np.unique(bin_index):
        selected = bin_index == b
        if np.count_nonzero(selected) < 3:
            continue
        core_radius.append(0.5 * (edges[b] + edges[min(b + 1, len(edges) - 1)]))
        magnitudes.append(float(u_theta[keep][selected].mean()))
    core_radius = np.asarray(core_radius)
    magnitudes = np.asarray(magnitudes)
    if core_radius.size < 3:
        return (float("nan"), False)
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
    return (float(r_peak), bool(boundary_limited))


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
    if physics == "merging" and (not force_merged):
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
    pair_unresolved = physics == "merging" and (not pair_resolved)
    if pair_unresolved and (not peak_coalesced):
        vortex_separation = float("nan")
    elif peak_coalesced:
        vortex_separation = 0.0
    r_max = _search_radius(physics, vortex_separation)
    support0 = support1 = None
    if not pair_unresolved and np.isfinite(c0).all() and np.isfinite(c1).all():
        x, y = (field["x"], field["y"])
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
    if pair_unresolved and (not peak_coalesced):
        core_radius_0 = core_radius_1 = float("nan")
    mean_core_radius = (
        float(np.mean([a for a in (core_radius_0, core_radius_1) if np.isfinite(a)]))
        if any((np.isfinite(a) for a in (core_radius_0, core_radius_1)))
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
        vts = sorted(case_dir.glob("*_zq_*.vts"))
        if vts:
            cases[case_dir.name] = vts
    return cases


def extract_field_diagnostics(samples_dir: Path, case: str | None = None) -> None:
    """Write diagnostics for every sampled directory, optionally by physical case."""
    samples_dir = Path(samples_dir)
    cases = _find_cases(samples_dir)
    if case is not None:
        cases = {
            name: vts for name, vts in cases.items() if name == case or name.startswith(f"{case}_")
        }
    for case_name, vts in sorted(cases.items()):
        physics, scheme = case_name.split("_", 1)
        if scheme == "rwm":
            continue
        timeline = pvd_time_map(samples_dir, physics, scheme)
        rows = []
        previous_centres = None
        coalesced_phase = False
        pair_lost = False
        for path in vts:
            step = int(VTS_STEP_RE.search(path.name).group(1))
            field = read_surface_field(path)
            field["step"] = step
            field["time"] = timeline[step]
            if physics == "merging" and coalesced_phase:
                row = _diagnostics_row(field, "merging", force_merged=True)
            else:
                row = _diagnostics_row(field, physics, previous_centres)
                if (
                    physics == "merging"
                    and pair_lost
                    and (not bool(row[FIELD_CSV_COLUMNS.index("is_peak_coalesced")]))
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
        write_csv_table(out, rows, columns=FIELD_CSV_COLUMNS)
        print(f"  [field] {case_name}: wrote field_diagnostics.csv ({len(rows)} samples)")


def _theme():
    return theme


def load_theme() -> tuple[dict[str, str], object | None]:
    """Load the OpenONDA matplotlib theme and return (COLORS dict, theme module)."""
    theme = _theme()
    theme.set_thesis_style()
    return (dict(theme.COLORS), theme)


def build_style_map(colors: dict[str, str]) -> dict[str, dict]:
    """Map scheme names to plot style dicts (color, marker, label)."""
    return {name: dict(style) for name, style in _theme().LAMB_OSEEN_SCHEME_STYLE.items()}


def scheme_zorder(scheme: str, offset: int = 0) -> int:
    """Layer one scheme consistently, with CS above every comparison curve."""
    return _SCHEME_ZORDER[scheme] + offset


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a named figure size in inches from the shared theme."""
    return _theme().figure_size(name)


def centered_subplots_adjust(fig, *, outer: float, **kwargs) -> None:
    """Apply the shared horizontally-centred thesis layout."""
    _theme().centered_subplots_adjust(fig, outer=outer, **kwargs)


def save_fig(fig, path: Path, dpi: int) -> None:
    """Export the authored canvas, including both formats for a .both request."""
    _theme().export_figure(fig, path, dpi=dpi)


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse as _argparse

    p = _argparse.ArgumentParser(description=description)
    p.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Raster resolution.")
    p.add_argument(
        "--format", choices=("png", "pdf", "both"), default="both", help="Output figure format."
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


MERGING_NORMALIZED_END_TIME = 3.0
THETA_REFERENCE = REF_DIR / "theta_vs_tau.csv"
CORE_REFERENCE = REF_DIR / "a2_over_b02.csv"
SEPARATION_DIMENSIONAL_REFERENCE = REF_DIR / "b_over_b0_time.csv"
REFERENCE_FINAL_TIME_SECONDS = 33.6
REFERENCE_FINAL_VISCOUS_TIME = 0.04744
REFERENCE_VISCOUS_TIME_PER_SECOND = REFERENCE_FINAL_VISCOUS_TIME / REFERENCE_FINAL_TIME_SECONDS


def lamb_oseen_profile(
    radius: np.ndarray, time: float, circulation: float, kinematic_viscosity: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact Lamb--Oseen velocity and vorticity at physical vortex age ``time``."""
    core_squared = 4.0 * kinematic_viscosity * time
    gaussian_core = float(np.sqrt(core_squared))
    vorticity = circulation / (np.pi * core_squared) * np.exp(-(radius**2) / core_squared)
    velocity = np.zeros_like(radius)
    nonzero = np.abs(radius) > 1e-12
    velocity[nonzero] = (
        circulation
        / (2.0 * np.pi * radius[nonzero])
        * (1.0 - np.exp(-(radius[nonzero] ** 2) / core_squared))
    )
    return (velocity, vorticity, gaussian_core)


def lamb_oseen_gradient(
    radius: np.ndarray, time: float, circulation: float, kinematic_viscosity: float
) -> np.ndarray:
    """Exact radial derivative of Lamb--Oseen azimuthal velocity."""
    core_squared = 4.0 * kinematic_viscosity * time
    gradient = np.zeros_like(radius)
    nonzero = np.abs(radius) > 1e-12
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
) -> tuple:
    """Return the y=0 single-vortex profile at a common physical time."""
    timeline = pvd_time_map(samples_dir, "vortex", scheme)
    _, selected_step = saved_frame([(time, step) for step, time in timeline.items()], target_time)
    path = samples_dir / f"vortex_{scheme}" / f"vortex_{scheme}_zq_{selected_step:06d}.vts"
    field = read_surface_field(path)
    row = int(np.argmin(np.abs(field["y"][0, :])))
    x = field["x"][:, row]
    velocity = field["velocity_y"][:, row]
    vorticity = field["vorticity_z"][:, row]
    if not include_uncertainty:
        return (x, velocity, vorticity, timeline[selected_step])
    velocity_se = field["velocity_standard_error_y"][:, row]
    vorticity_se = field["vorticity_standard_error_z"][:, row]
    gradient_se = field["velocity_gradient_yx_standard_error"][:, row]
    return (
        x,
        velocity,
        vorticity,
        timeline[selected_step],
        velocity_se,
        vorticity_se,
        gradient_se,
        float(field["confidence_multiplier"]),
    )


def latest_common_time(samples_dir: Path, prefix: str = "vortex") -> float:
    """Latest saved physical time shared by every method for ``prefix``."""
    clocks = [
        sorted(pvd_time_map(samples_dir, prefix, scheme).values()) for scheme in SCHEME_DRAW_ORDER
    ]
    return common_frame_time(*clocks)


def load_dipole_feature_frame(samples_dir: Path, scheme: str) -> pd.DataFrame:
    """Return one scheme's field diagnostics on the dominant sample cadence."""
    path = samples_dir / f"dipole_{scheme}" / "field_diagnostics.csv"
    data = pd.DataFrame(read_history_table(path))
    data = data[uniform_cadence_mask(data["step"].to_numpy(int))].copy()
    data["step"] = data["step"].astype(int)
    return data.set_index("step", drop=False)


def extract_dipole_timeseries(samples_dir: Path, scheme: str) -> dict | None:
    data = load_dipole_feature_frame(samples_dir, scheme)
    output = {
        "t": data["time"].to_numpy(float),
        "x_core": data["vortex_centre_0_x"].to_numpy(float),
        "a_c": data["core_radius_0"].to_numpy(float),
    }
    if scheme == "rwm":
        for column, key in (
            ("vortex_centre_0_x_ci_lower", "x_core_ci_lower"),
            ("vortex_centre_0_x_ci_upper", "x_core_ci_upper"),
            ("core_radius_0_ci_lower", "a_c_ci_lower"),
            ("core_radius_0_ci_upper", "a_c_ci_upper"),
        ):
            output[key] = data[column].to_numpy(float)
    return output


def connected_core_aspect(field: dict, fraction: float = 0.5) -> float:
    """Return the mean aspect ratio of both connected high-vorticity cores.

    For each sign, the support is the connected region above ``fraction`` of
    that sign's peak.  The reported value is
    ``sqrt(lambda_max / lambda_min)`` of its vorticity-weighted covariance,
    averaged over the two signs.  A circular core therefore has value one.
    """
    x = np.asarray(field["x"], dtype=float)
    y = np.asarray(field["y"], dtype=float)
    vorticity = np.asarray(field["vorticity_z"], dtype=float)
    aspects: list[float] = []
    for sign in (1.0, -1.0):
        signed = sign * vorticity
        candidates = _peak_candidates(signed)
        if not candidates:
            aspects.append(float("nan"))
            continue
        peak = candidates[0]
        support = np.isfinite(signed) & (signed >= fraction * float(signed[peak]))
        labels, _ = ndimage.label(support, structure=np.ones((3, 3), dtype=np.int8))
        region = labels == int(labels[peak])
        weights = np.where(region, signed, 0.0)
        total = float(np.sum(weights))
        if total <= np.finfo(float).tiny:
            aspects.append(float("nan"))
            continue
        centre_x = float(np.sum(weights * x) / total)
        centre_y = float(np.sum(weights * y) / total)
        dx = x - centre_x
        dy = y - centre_y
        covariance = (
            np.array(
                [
                    [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
                    [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
                ],
                dtype=float,
            )
            / total
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        aspects.append(
            float(np.sqrt(eigenvalues[1] / eigenvalues[0]))
            if eigenvalues[0] > 0.0
            else float("nan")
        )
    return float(np.nanmean(aspects))


def diffusion_only_dipole_features(
    template: dict,
    times: np.ndarray,
    circulation: float,
    gaussian_radius0: float,
    kinematic_viscosity: float,
    separation0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the plotted feature definitions to two fixed diffusing Gaussians."""
    x = np.asarray(template["x"], dtype=float)
    y = np.asarray(template["y"], dtype=float)
    separation: list[float] = []
    aspect: list[float] = []
    for time in np.asarray(times, dtype=float):
        sigma2 = gaussian_radius0**2 + 4.0 * kinematic_viscosity * time
        positive = np.exp(-(x * x + (y - 0.5 * separation0) ** 2) / sigma2)
        negative = np.exp(-(x * x + (y + 0.5 * separation0) ** 2) / sigma2)
        field = {
            "x": x,
            "y": y,
            "vorticity_z": circulation * (positive - negative) / (np.pi * sigma2),
        }
        centres, _ = _vorticity_peak_centres(field, "dipole")
        separation.append(float(np.linalg.norm(centres[0] - centres[1])))
        aspect.append(connected_core_aspect(field))
    return (np.asarray(separation), np.asarray(aspect))


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
) -> dict:
    """Return merger features on ``nu*t/a_c0^2`` through the literature horizon."""
    path = samples_dir / f"merging_{scheme}" / "field_diagnostics.csv"
    data = pd.DataFrame(read_history_table(path))
    data = data[uniform_cadence_mask(data["step"].to_numpy(int))]
    tau = kinematic_viscosity * data["time"].to_numpy(float) / core_radius**2
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
        "orientation_anisotropy": data["orientation_anisotropy"].to_numpy(float),
    }
    if scheme == "rwm":
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
                output[f"{key}_ci_{bound}"] = transform(data[f"{base}_ci_{bound}"].to_numpy(float))
    if scheme == "rwm":
        theta_half_width = 0.5 * (output["theta_ci_upper"] - output["theta_ci_lower"])
        theta_reliable = np.isfinite(theta_half_width) & (theta_half_width <= 45.0)
        core_half_width = 0.5 * (
            output["a_c2_over_b02_ci_upper"] - output["a_c2_over_b02_ci_lower"]
        )
        core_reliable = (
            np.isfinite(core_half_width)
            & (output["a_c2_over_b02_ci_lower"] >= 0.0)
            & (core_half_width <= 0.5 * np.maximum(output["a_c2_over_b02"], 1e-12))
        )
        separation_half_width = 0.5 * (output["b_over_b0_ci_upper"] - output["b_over_b0_ci_lower"])
        separation_reliable = output["is_pair_unresolved"] | np.isfinite(separation_half_width) & (
            output["b_over_b0_ci_lower"] >= 0.0
        ) & (separation_half_width <= 0.5 * np.maximum(output["b_over_b0"], 1e-12))
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
        values = read_numeric_table(path, delimiter=",")
        output[name] = np.column_stack((values[:, 0] / scale, values[:, 1]))
    values = read_numeric_table(SEPARATION_DIMENSIONAL_REFERENCE, delimiter=",")
    paper_viscous_time = values[:, 0] * REFERENCE_VISCOUS_TIME_PER_SECOND
    output["separation"] = np.column_stack((paper_viscous_time / scale, values[:, 1]))
    return output


def _reportable_energy_rate_mask(data_frame: pd.DataFrame) -> np.ndarray:
    """Identify finite dE/dt samples with a documented energy definition."""
    finite = np.isfinite(data_frame["kinetic_energy_rate"].to_numpy(float))
    if "kinetic_energy_rate_source" in data_frame:
        source = data_frame["kinetic_energy_rate_source"].astype(str).to_numpy()
        documented = np.asarray([not value.startswith("undefined_") for value in source])
        return finite & documented
    return finite


def read_flow_integrals(csv_path: Path) -> dict:
    """Load a validated, non-initialized flow-integral history."""
    data_frame = pd.DataFrame(read_history_table(csv_path))
    kinetic_energy_rate = data_frame["kinetic_energy_rate"].to_numpy(float)
    reportable_energy_rate = _reportable_energy_rate_mask(data_frame)
    kinetic_energy_rate = np.where(reportable_energy_rate, kinetic_energy_rate, np.nan)
    data = {
        "time": data_frame["time"].to_numpy(float),
        "kinetic_energy_rate": kinetic_energy_rate,
        "viscous_kinetic_energy_rate": data_frame["viscous_kinetic_energy_rate"].to_numpy(float),
    }
    if "total_kinetic_energy" in data_frame:
        data["total_kinetic_energy"] = data_frame["total_kinetic_energy"].to_numpy(float)
    for measure in ("total_kinetic_energy", "kinetic_energy_rate", "viscous_kinetic_energy_rate"):
        for bound in ("lower", "upper"):
            column = f"{measure}_ci_{bound}"
            if column in data_frame:
                data[column] = data_frame[column].to_numpy(float)
    return data


def _quadrant_split(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    return (int(np.searchsorted(y[:, 0], 0.0)), int(np.searchsorted(x[0], 0.0)))


def _boundary_edges(field: np.ndarray, row: int, column: int, tx: float, ty: float):
    boundary_column = field[:, column - 1] * (1.0 - tx) + field[:, column] * tx
    boundary_row = field[row - 1, :] * (1.0 - ty) + field[row, :] * ty
    corner = boundary_row[column - 1] * (1.0 - tx) + boundary_row[column] * tx
    return (boundary_column, boundary_row, corner)


def _tile_coordinates(
    x: np.ndarray, y: np.ndarray, quadrant: str, column: int, row: int
) -> tuple[np.ndarray, np.ndarray]:
    if quadrant == "TL":
        return (np.append(x[:column], 0.0), np.insert(y[row:], 0, 0.0))
    if quadrant == "TR":
        return (np.insert(x[column:], 0, 0.0), np.insert(y[row:], 0, 0.0))
    if quadrant == "BL":
        return (np.append(x[:column], 0.0), np.append(y[:row], 0.0))
    if quadrant == "BR":
        return (np.insert(x[column:], 0, 0.0), np.append(y[:row], 0.0))


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


def surface_plot_tiles(
    samples_dir: Path,
    layout: list[tuple],
    core_radius: float,
    velocity_scale: float,
    vorticity_scale: float,
    kinematic_viscosity: float,
) -> tuple[list[dict], float | None]:
    """Return normalized, seam-free quadrant tiles at one common physical time."""
    comparison_time = latest_common_time(samples_dir, "vortex")
    tiles = []
    selected_times = []
    for scheme, quadrant, *_rest in layout:
        timeline = pvd_time_map(samples_dir, "vortex", scheme)
        _, step = saved_frame([(time, step) for step, time in timeline.items()], comparison_time)
        path = samples_dir / f"vortex_{scheme}" / f"vortex_{scheme}_zq_{step:06d}.vts"
        field = read_surface_field(path)
        x = field["x"].T / core_radius
        y = field["y"].T / core_radius
        velocity = np.hypot(field["velocity_x"], field["velocity_y"]).T / velocity_scale
        vorticity = np.clip(field["vorticity_z"].T, 0.0, None) / vorticity_scale
        row, column = _quadrant_split(x, y)
        x_axis, y_axis = (x[0], y[:, 0])
        tx = -x_axis[column - 1] / (x_axis[column] - x_axis[column - 1])
        ty = -y_axis[row - 1] / (y_axis[row] - y_axis[row - 1])
        grid_x, grid_y = np.meshgrid(*_tile_coordinates(x_axis, y_axis, quadrant, column, row))
        tiled = {}
        for name, values in (("velocity", velocity), ("vorticity", vorticity)):
            tiled[name] = _tile_field(
                values, quadrant, column, row, *_boundary_edges(values, row, column, tx, ty)
            )
        selected_time = timeline[int(path.stem[-6:])]
        selected_times.append(selected_time)
        tiles.append({"scheme": scheme, "quadrant": quadrant, "x": grid_x, "y": grid_y, **tiled})
    if selected_times:
        time_scale = kinematic_viscosity / core_radius**2
        print(
            f"  [surface] plotting {len(tiles)}/{len(SCHEMES)} methods at common nu*t/a_c0^2={comparison_time * time_scale:.2g} (selected samples {min(selected_times) * time_scale:.2g}-{max(selected_times) * time_scale:.2g})"
        )
    return (tiles, comparison_time)


STEP_RE = re.compile("_(\\d{6,})\\.h5$")
PHYSICS_CASES = ("vortex", "dipole", "merging")
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
Member = SimpleNamespace


class DiagnosticState:
    def __init__(self):
        self.previous_centres = None
        self.pair_lost = False
        self.coalesced = False


def _backup_map(folder: Path) -> dict[int, Path]:
    result = {}
    for path in vpm_backup_files(folder):
        match = STEP_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def discover_members(solution_root: Path, samples_root: Path, case_name: str) -> list[Member]:
    member_re = re.compile(f"^{re.escape(case_name)}_(\\d+)$")
    members: list[Member] = []
    records = []
    for solution_dir in sorted(solution_root.glob(f"{case_name}_*")):
        match = member_re.fullmatch(solution_dir.name)
        index = int(match.group(1))
        member_samples = Path(samples_root) / solution_dir.name
        metadata_path = solution_dir / "vpm_metadata.json"
        record = read_json(metadata_path)
        records.append(record)
        metadata = _flatten_solver_metadata(record)
        backups = _backup_map(solution_dir)
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
    realization_metadata(records)
    return members


def _template_grid(samples_root: Path, physics: str, metadata: dict) -> dict:
    folder = Path(samples_root) / f"{physics}_cs"
    candidates = sorted(folder.glob(f"{physics}_cs_zq_*.vts"))
    template = read_surface_field(candidates[0])
    if physics != "vortex":
        return template
    dx = float(np.median(np.diff(template["x"][:, 0])))
    core_radius = float(metadata["core_radius"])
    kinematic_viscosity = float(metadata["kinematic_viscosity"])
    end_time = float(metadata["end_time"])
    final_gaussian_radius = math.sqrt(core_radius**2 + 4.0 * kinematic_viscosity * end_time)
    brownian_standard_deviation = math.sqrt(2.0 * kinematic_viscosity * end_time)
    old_half_width = max(float(np.max(np.abs(template["x"]))), float(np.max(np.abs(template["y"]))))
    half_width = max(
        4.0 * final_gaussian_radius, old_half_width + 4.0 * brownian_standard_deviation
    )
    half_cells = int(math.ceil(half_width / dx))
    axis = np.arange(-half_cells, half_cells + 1, dtype=float) * dx
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return {"x": x, "y": y}


def _deposit_circulation_cic(
    x: np.ndarray, y: np.ndarray, position: np.ndarray, circulation_per_length: np.ndarray
) -> tuple[np.ndarray, float]:
    """Conservative cloud-in-cell deposition on the shared uniform grid."""
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
    return (deposited / (dx * dy), capture_fraction)


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
    return (velocity_x, velocity_y)


def project_backup(path: Path, template: dict, column_length: float) -> tuple[dict, dict]:
    """Return the column-projected 2-D field represented by one backup."""
    state = particle_state(path)
    position, strength, core_radius = (
        state[name] for name in ("position", "vortex_strength", "core_radius")
    )
    step, time = state["step"], state["time"]
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
    filter_options = {
        "sigma": (blob_radius / (math.sqrt(2.0) * dx), blob_radius / (math.sqrt(2.0) * dy)),
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
    arrays = field_ensemble(
        fields,
        coordinates=("x", "y"),
        fields=("velocity_x", "velocity_y", "vorticity_z", "velocity_gradient_yx"),
    )
    velocity = np.stack((arrays["velocity_x"], arrays["velocity_y"]), axis=-1)
    vorticity = arrays["vorticity_z"]
    gradient = arrays["velocity_gradient_yx"]
    n_members = len(fields)
    mean_velocity, velocity_se = mean_standard_error(velocity)
    mean_vorticity, vorticity_se = mean_standard_error(vorticity)
    mean_gradient, gradient_se = mean_standard_error(gradient)
    field = {
        "x": fields[0]["x"],
        "y": fields[0]["y"],
        "velocity_x": mean_velocity[..., 0],
        "velocity_y": mean_velocity[..., 1],
        "vorticity_z": mean_vorticity,
        "velocity_standard_error_x": velocity_se[..., 0],
        "velocity_standard_error_y": velocity_se[..., 1],
        "vorticity_standard_error_z": vorticity_se,
        "velocity_gradient_yx": mean_gradient,
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
    return (field, diagnostics)


def _leave_one_out_field(field: dict, omitted: int, confidence_multiplier: float) -> dict:
    velocity = np.delete(field["_member_velocity"], omitted, axis=0)
    vorticity = np.delete(field["_member_vorticity"], omitted, axis=0)
    gradient = np.delete(field["_member_gradient"], omitted, axis=0)
    n_members = len(velocity)
    _, velocity_se = mean_standard_error(velocity)
    _, vorticity_se = mean_standard_error(vorticity)
    mean_standard_error(gradient)
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
    if physics == "merging" and state.pair_lost and (not bool(row[coalesced_index])):
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
    estimate: list, leave_one_out: list[list], confidence_multiplier: float
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
    write_vtk_dataset(grid, path)


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
    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_flow_integrals(members: list[Member], output: Path, multiplier: float) -> None:
    frames = [
        pd.DataFrame(read_csv_table(member.samples_dir / "flow_integrals.csv"))
        for member in members
    ]
    clock, stacks = table_ensemble(frames)
    result_columns = {column: clock[column].to_numpy() for column in clock.columns}
    for column, values in stacks.items():
        mean, standard_error = mean_standard_error(values)
        result_columns[column] = mean
        result_columns[f"{column}_standard_error"] = standard_error
        result_columns[f"{column}_ci_lower"] = mean - multiplier * standard_error
        result_columns[f"{column}_ci_upper"] = mean + multiplier * standard_error
    result = pd.DataFrame(result_columns)
    write_csv_table(output, result.itertuples(index=False, name=None), columns=result.columns)


def aggregate_case(solution_root: Path, samples_root: Path, physics: str) -> dict:
    case_name = f"{physics}_rwm"
    members = discover_members(solution_root, samples_root, case_name)
    metadata0 = members[0].metadata
    template = _template_grid(samples_root, physics, metadata0)
    column_length = 2.0 * float(metadata0["column_half_length"])
    sample_z = 0.25 * column_length
    n_members = len(members)
    multiplier = float(stats.t.ppf(0.5 + 0.5 * CONFIDENCE_LEVEL, n_members - 1))
    loo_multiplier = float(stats.t.ppf(0.5 + 0.5 * CONFIDENCE_LEVEL, n_members - 2))
    output_dir = Path(samples_root) / case_name
    diagnostic_state = DiagnosticState()
    loo_states = [DiagnosticState() for _ in members]
    feature_records = []
    convergence_records = []
    pvd_entries = []
    minimum_capture = 1.0
    steps = sorted(members[0].backups)
    for step in steps:
        member_fields = []
        projection_qa = []
        for member in members:
            field, qa = project_backup(member.backups[step], template, column_length)
            member_fields.append(field)
            projection_qa.append(qa)
        aggregate, convergence = _stack_mean_field(member_fields, multiplier)
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
            f"  [RWM] {case_name}: step {step:06d}, t={aggregate['time']:.4g}, relative MCSE(omega)={convergence['relative_standard_error_l2_vorticity']:.3%}"
        )
    _write_pvd(output_dir / f"{case_name}_zq.pvd", pvd_entries)
    feature_table = pd.DataFrame(feature_records)
    write_csv_table(
        output_dir / "field_diagnostics.csv",
        feature_table.itertuples(index=False, name=None),
        columns=feature_table.columns,
    )
    convergence_table = pd.DataFrame(convergence_records)
    write_csv_table(
        output_dir / "rwm_convergence.csv",
        convergence_table.itertuples(index=False, name=None),
        columns=convergence_table.columns,
    )
    _aggregate_flow_integrals(members, output_dir / "flow_integrals.csv", multiplier)
    return {
        "case": physics,
        "ensemble_size": n_members,
        "random_seeds": [member.seed for member in members],
        "confidence_level": CONFIDENCE_LEVEL,
        "minimum_absolute_circulation_capture_fraction": minimum_capture,
    }


def aggregate_rwm_ensemble(solution_root: Path, samples_root: Path) -> dict[str, dict]:
    return {
        physics: aggregate_case(solution_root, samples_root, physics) for physics in PHYSICS_CASES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-fields", action="store_true")
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--aggregate-rwm", action="store_true")
    args = parser.parse_args()
    if args.aggregate_rwm:
        if args.case:
            aggregate_case(SOLUTION_DIR, args.samples_dir, args.case)
        else:
            aggregate_rwm_ensemble(SOLUTION_DIR, args.samples_dir)
    if args.extract_fields:
        extract_field_diagnostics(args.samples_dir, args.case)


if __name__ == "__main__":
    main()
