#!/usr/bin/env python3
"""Vortex diagnostics for the Lamb-Oseen tutorial: paths, physics constants,
sampled-field readers, and the field-based centre/core-radius extraction
algorithm.

The centre/core-radius/vortex_separation/merged status for all three physics cases
(single vortex, dipole, merging) are derived from the *sampled velocity
field* itself (the ``*_zq_*.vts`` planes written by the ``SurfaceSampler``
at z = +L/4) — one consistent method, independent of the viscous diffusion
scheme and the physics case:

  * vortex centres   — sub-grid locations of peak vorticity, matching the
    Cerretelli--Williamson vortex_separation/orientation definition;
  * core radius a_c  — radius where the azimuthally-averaged tangential
    velocity |u_theta(r)| peaks, measured on the outward semicircle before
    merger and over the full circle after merger;
  * vortex_separation and orbital angle (dipole/merging pair).

Run directly to extract ``<case>/field_diagnostics.csv`` next to the sampled
planes for every case under ``samples/``.
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

# -- Directory layout --------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent  # …/assets/
SCRIPT_DIR = ASSETS_DIR.parent  # …/lamb_oseen_vortex/
FIGURES_DIR = SCRIPT_DIR / "figures"
SAMPLES_DIR = SCRIPT_DIR / "samples"
REF_DIR = ASSETS_DIR / "references"

# -- Tutorial constants -------------------------------------------------------
SCHEMES = ("cs", "rwm", "dvh", "gbd")
CASES = ("vortex", "dipole", "merging")
ENERGY_CASES = (
    ("vortex", "Single vortex", 1),
    ("dipole", "Vortex dipole", 2),
    ("merging", "Co-rotating merger", 2),
)

BETA_RMAX = 1.12
REFERENCE_CIRCULATION = 1.0
REYNOLDS_NUMBER = 530.0
CORE_RADIUS = 0.125  # paper's radius of maximum azimuthal velocity
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX
SEPARATION = 1.0
COLUMN_LENGTH = 50.0 * CORE_RADIUS  # mirrors lamb_oseen_setup.py::COLUMN_LENGTH
FIELD_SPACING = 0.16 * CORE_RADIUS  # mirrors lamb_oseen_setup.py::FIELD_SPACING
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
    "is_merged",
    "is_core_radius_0_boundary_limited",
    "is_core_radius_1_boundary_limited",
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

    Plotting must not depend on a dense particle checkpoint: those files are sparse
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
    velocity = np.asarray(grid.point_data["velocity"], dtype=np.float64)
    vorticity = np.asarray(grid.point_data["vorticity"], dtype=np.float64)

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
        "velocity_x": _f(velocity[:, 0]),
        "velocity_y": _f(velocity[:, 1]),
        "vorticity_z": _f(vorticity[:, 2]),
    }


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
) -> list[np.ndarray]:
    """Vortex centres from sub-grid peak-vorticity locations."""
    x, y, wz = field["x"], field["y"], field["vorticity_z"]

    if physics == "dipole":
        centres = []
        for sign in (1.0, -1.0):
            signed = sign * wz
            candidates = _peak_candidates(signed)
            if not candidates:
                centres.append(np.array([np.nan, np.nan]))
                continue
            centres.append(_subgrid_peak_centre(x, y, signed, candidates[0]))
        return centres

    signed = np.abs(wz) if physics == "vortex" else wz
    candidates = _peak_candidates(signed, 0.35 if physics == "merging" else 0.20)
    if not candidates:
        return [np.array([np.nan, np.nan])]

    if physics == "merging":
        peaks = _merging_peak_pair(field, candidates, previous_centres)
        return [_subgrid_peak_centre(x, y, signed, peak) for peak in peaks]

    centre = _subgrid_peak_centre(x, y, signed, candidates[0])
    return [centre] if np.isfinite(centre).all() else [np.array([np.nan, np.nan])]


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
    """u_theta-peak search window. Dipole/merging scale off the nominal pair
    vortex_separation; a lone vortex has no vortex_separation to scale off, so use a
    generous multiple of the *expected* final core radius instead (the core
    can diffuse well past a fixed fraction of SEPARATION by t=TOTAL_TIME)."""
    if physics == "vortex":
        expected_final_gaussian_radius = np.sqrt(
            GAUSSIAN_CORE_RADIUS**2 + 4.0 * (REFERENCE_CIRCULATION / REYNOLDS_NUMBER) * TOTAL_TIME
        )
        return 2.0 * BETA_RMAX * expected_final_gaussian_radius
    return 0.5


def _diagnostics_row(
    field: dict,
    physics: str,
    previous_centres: list[np.ndarray] | None = None,
) -> list:
    centres = _match_centres_to_previous(
        _vorticity_peak_centres(field, physics, previous_centres), previous_centres
    )
    c0 = centres[0] if len(centres) >= 1 else np.array([np.nan, np.nan])
    c1 = centres[1] if len(centres) >= 2 else np.array([np.nan, np.nan])

    vortex_separation = (
        float(np.linalg.norm(c0 - c1))
        if np.isfinite(c0).all() and np.isfinite(c1).all()
        else float("nan")
    )
    merged = physics == "merging" and not np.isfinite(vortex_separation)

    r_max = _search_radius(physics, vortex_separation)
    support0 = support1 = None
    if np.isfinite(c0).all() and np.isfinite(c1).all():
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

    mean_core_radius = (
        float(np.mean([a for a in (core_radius_0, core_radius_1) if np.isfinite(a)]))
        if any(np.isfinite(a) for a in (core_radius_0, core_radius_1))
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
        vortex_separation,
        core_radius_0,
        core_radius_1,
        mean_core_radius,
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
        previous_centres = None
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
                # pair and create a fictitious vortex_separation or angle history.
                row = _diagnostics_row(field, "vortex")
                row[11] = True
            else:
                row = _diagnostics_row(field, physics, previous_centres)
            rows.append(row)
            if physics == "merging" and bool(row[11]):
                merged_phase = True
                previous_centres = None
            elif np.isfinite(row[2:6]).all():
                previous_centres = [np.asarray(row[2:4]), np.asarray(row[4:6])]

        out = samples_dir / case_name / "field_diagnostics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(FIELD_CSV_COLUMNS)
            writer.writerows(rows)
        print(f"  [field] {case_name}: wrote field_diagnostics.csv ({len(rows)} steps)")


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


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a named figure size in inches from the shared theme."""
    return _theme().figure_size(name)


def save_fig(fig, path: Path, dpi: int) -> None:
    """Save without tight layout or cropping; manual subplots_adjust() takes precedence."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"  Saved: {out}")


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse as _argparse

    p = _argparse.ArgumentParser(description=description)
    p.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Figure DPI (PNG only).")
    p.add_argument(
        "--format",
        choices=_theme().EXPORT_FORMATS,
        default="png",
        help="Output figure format (default: png).",
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
