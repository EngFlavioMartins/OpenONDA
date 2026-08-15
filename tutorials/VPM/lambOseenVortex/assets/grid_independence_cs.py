#!/usr/bin/env python3
"""Resource-bounded CS-only spatial convergence study for the single vortex.

The three default levels use a constant refinement ratio of 4/3.  Particle
spacing and axial layer spacing are refined together, while the field sampler
uses one fixed, finer grid so sampling error is not confused with solver-grid
error.  Existing completed levels are reused and incomplete directories are
left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .plot_vortex_comparison import lamb_oseen_gradient, lamb_oseen_profile, load_profile
    from .vortex_diagnostics import BETA_RMAX
else:
    from plot_vortex_comparison import lamb_oseen_gradient, lamb_oseen_profile, load_profile
    from vortex_diagnostics import BETA_RMAX


SCRIPT_DIR = Path(__file__).resolve().parent.parent
SETUP = SCRIPT_DIR / "lambossen_setup.py"
DEFAULT_ROOT = SCRIPT_DIR / "grid_study" / "cs_single_rk3_p3"
DEFAULT_LEVELS = (0.60, 0.45, 0.3375)
FIELDS = ("velocity_l2", "vorticity_l2", "velocity_gradient_l2")
SELF_FIELDS = ("velocity", "vorticity", "velocity_gradient")
SELF_CONVERGENCE_TOLERANCE = 0.005


def level_name(spacing_ratio: float) -> str:
    return f"h_{spacing_ratio:.5f}".rstrip("0").replace(".", "p")


def relative_l2(numerical: np.ndarray, exact: np.ndarray) -> float:
    denominator = float(np.linalg.norm(exact))
    return float(np.linalg.norm(numerical - exact) / denominator) if denominator > 0.0 else np.nan


def json_compatible(value):
    """Replace non-finite numerical values so the report is strict JSON."""
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def completed_metadata(
    root: Path,
    spacing_ratio: float | None = None,
    args: argparse.Namespace | None = None,
) -> dict | None:
    path = root / "samples" / "vortex_cs" / "run_metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if metadata.get("completed") is not True:
        return None
    if spacing_ratio is None or args is None:
        return metadata
    a0 = float(metadata.get("velocity_peak_radius", np.nan))
    expected = (
        metadata.get("case") == "vortex",
        metadata.get("scheme") == "cs",
        metadata.get("core_radius_definition") == "gaussian_1_over_e_vorticity_radius",
        metadata.get("circulation_normalization") == "per_vortex_after_strength_cutoff",
        metadata.get("processing_unit") == "CPU",
        metadata.get("advection_scheme") == "RK3",
        np.isclose(float(metadata.get("time_step", np.nan)), 0.01),
        np.isclose(float(metadata.get("treecode_theta", np.nan)), 0.30),
        int(metadata.get("treecode_multipole_order", -1)) == 3,
        np.isclose(float(metadata.get("in_plane_spacing", np.nan)) / a0, spacing_ratio),
        np.isclose(float(metadata.get("field_spacing", np.nan)) / a0, args.field_spacing_ratio),
        np.isclose(float(metadata.get("total_time", np.nan)), args.total_time),
        np.isclose(
            float(metadata.get("sample_plane_fraction", np.nan)), args.sample_plane_fraction
        ),
    )
    return metadata if all(expected) else None


def run_level(root: Path, spacing_ratio: float, args: argparse.Namespace) -> dict:
    existing = completed_metadata(root, spacing_ratio, args)
    if existing is not None:
        print(f"  [grid] reuse completed {level_name(spacing_ratio)}")
        return existing
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite incomplete grid level: {root}. "
            "Move it aside or choose a different --output-root."
        )

    command = [
        sys.executable,
        "-u",
        str(SETUP),
        "--gamma1",
        "+1",
        "--schemes",
        "cs",
        "--spacing-ratio",
        str(spacing_ratio),
        "--field-spacing-ratio",
        str(args.field_spacing_ratio),
        "--total-time",
        str(args.total_time),
        "--sample-plane-fraction",
        str(args.sample_plane_fraction),
        "--output-root",
        str(root),
        "--processing-unit",
        "CPU",
    ]
    print(f"  [grid] run {level_name(spacing_ratio)} (CS only)")
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)  # noqa: S603
    metadata = completed_metadata(root, spacing_ratio, args)
    if metadata is None:
        raise RuntimeError(f"Grid level did not complete cleanly: {root}")
    return metadata


def analyze_level(root: Path, spacing_ratio: float, metadata: dict) -> dict[str, float]:
    profile = load_profile(root / "samples", "cs")
    if profile is None:
        raise RuntimeError(f"No final CS profile found below {root}")
    x, velocity, vorticity, time = profile
    gamma = abs(float(metadata.get("circulations", [1.0])[0]))
    nu = float(metadata["viscosity"])
    gaussian_a0 = float(metadata["core_radius"])
    velocity_peak_a0 = float(metadata["velocity_peak_radius"])
    t0 = gaussian_a0**2 / (4.0 * nu)
    exact_velocity, exact_vorticity, _ = lamb_oseen_profile(x, t0 + time, gamma, nu)
    exact_gradient = lamb_oseen_gradient(x, t0 + time, gamma, nu)
    numerical_gradient = np.gradient(velocity, x)
    window = np.abs(x / velocity_peak_a0) <= 5.5
    return {
        "spacing_ratio": spacing_ratio,
        "spacing": float(metadata["in_plane_spacing"]),
        "field_spacing_ratio": float(metadata["field_spacing"]) / velocity_peak_a0,
        "sample_plane_fraction": float(metadata.get("sample_plane_fraction", np.nan)),
        "column_length_over_a0": 2.0 * float(metadata["column_half_length"]) / velocity_peak_a0,
        "final_time": float(time),
        "initial_particles": int(metadata.get("initial_particle_count", -1)),
        "wall_time_seconds": float(metadata.get("wall_time_seconds", np.nan)),
        "velocity_l2": relative_l2(velocity[window], exact_velocity[window]),
        "vorticity_l2": relative_l2(vorticity[window], exact_vorticity[window]),
        "velocity_gradient_l2": relative_l2(numerical_gradient[window], exact_gradient[window]),
    }


def add_convergence_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    rows.sort(key=lambda row: row["spacing_ratio"], reverse=True)
    h = np.asarray([row["spacing"] for row in rows])
    orders = {}
    for field in FIELDS:
        errors = np.asarray([row[field] for row in rows])
        finite = np.isfinite(errors) & (errors > 0.0)
        orders[field] = (
            float(np.polyfit(np.log(h[finite]), np.log(errors[finite]), 1)[0])
            if np.count_nonzero(finite) >= 3
            else np.nan
        )
        for index, row in enumerate(rows):
            row[f"{field}_change_to_finer"] = (
                abs(errors[index] - errors[index + 1]) / errors[index + 1]
                if index + 1 < len(rows) and errors[index + 1] > 0.0
                else np.nan
            )
    return orders


def self_convergence_metrics(
    root: Path, rows: list[dict[str, float]], refinement_ratio: float
) -> dict[str, dict[str, float | bool]]:
    """Compare successive numerical solutions on their common fixed sample grid."""
    if len(rows) != 3:
        return {}

    profiles = []
    for row in rows:
        profile = load_profile(root / level_name(row["spacing_ratio"]) / "samples", "cs")
        if profile is None:
            return {}
        x, velocity, vorticity, _ = profile
        profiles.append((x, velocity, vorticity, np.gradient(velocity, x)))

    common_x = profiles[-1][0]
    a0 = rows[-1]["spacing"] / rows[-1]["spacing_ratio"]
    window = np.abs(common_x / a0) <= 5.5
    metrics: dict[str, dict[str, float | bool]] = {}
    for field_index, field in enumerate(SELF_FIELDS, start=1):
        values = [np.interp(common_x, profile[0], profile[field_index]) for profile in profiles]
        denominator = float(np.linalg.norm(values[-1][window]))
        coarse_medium = (
            float(np.linalg.norm((values[0] - values[1])[window]) / denominator)
            if denominator > 0.0
            else np.nan
        )
        medium_fine = (
            float(np.linalg.norm((values[1] - values[2])[window]) / denominator)
            if denominator > 0.0
            else np.nan
        )
        decreasing = bool(
            np.isfinite(coarse_medium) and np.isfinite(medium_fine) and medium_fine < coarse_medium
        )
        apparent_order = (
            float(np.log(coarse_medium / medium_fine) / np.log(refinement_ratio))
            if decreasing and medium_fine > 0.0
            else np.nan
        )
        metrics[field] = {
            "coarse_to_medium_relative_difference": coarse_medium,
            "medium_to_fine_relative_difference": medium_fine,
            "successive_difference_decreases": decreasing,
            "three_grid_apparent_order": apparent_order,
        }
    return metrics


def write_results(
    root: Path,
    rows: list[dict[str, float]],
    orders: dict[str, float],
    self_metrics: dict[str, dict[str, float | bool]],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "grid_independence_cs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    exact_errors_decrease = {
        field: bool(
            len(rows) == 3
            and all(rows[index + 1][field] < rows[index][field] for index in range(2))
        )
        for field in FIELDS
    }
    self_differences_decrease = {
        field: bool(metrics["successive_difference_decreases"])
        for field, metrics in self_metrics.items()
    }
    complete_three_grid_study = len(rows) == 3 and len(self_metrics) == len(SELF_FIELDS)
    numerical_grid_independent = bool(
        complete_three_grid_study
        and all(self_differences_decrease.values())
        and all(
            float(metrics["medium_to_fine_relative_difference"]) <= SELF_CONVERGENCE_TOLERANCE
            for metrics in self_metrics.values()
        )
    )
    analytical_validation_supported = bool(
        complete_three_grid_study and all(exact_errors_decrease.values())
    )
    report = {
        "scheme": "cs",
        "physics": "single_vortex",
        "levels_coarse_to_fine": [row["spacing_ratio"] for row in rows],
        "refinement_strategy": "in-plane and axial particle spacings refined together",
        "sampling_strategy": "fixed field grid at every level",
        "sample_plane_fraction": rows[0]["sample_plane_fraction"],
        "column_length_over_a0": rows[0]["column_length_over_a0"],
        "processing_unit": "CPU",
        "complete_three_grid_study": complete_three_grid_study,
        "observed_exact_solution_error_orders": orders,
        "exact_solution_errors_decrease_under_refinement": exact_errors_decrease,
        "successive_solution_comparisons": self_metrics,
        "successive_differences_decrease_under_refinement": self_differences_decrease,
        "self_convergence_tolerance": SELF_CONVERGENCE_TOLERANCE,
        "grid_independence_verdict": (
            "supported_at_stated_tolerance" if numerical_grid_independent else "not_supported"
        ),
        "analytical_reference_convergence_verdict": (
            "supported" if analytical_validation_supported else "not_supported"
        ),
        "interpretation": (
            "Spatial grid independence is supported only when every successive-solution "
            "difference decreases and every medium-to-fine difference is no larger than "
            f"{SELF_CONVERGENCE_TOLERANCE:.3g}. Analytical validation is reported separately: "
            "the exact reference is an infinite two-dimensional Lamb-Oseen vortex, whereas "
            "these simulations use a finite vortex column, so model-form error can impose "
            "an error floor even after the numerical solution is grid independent."
        ),
    }
    (root / "grid_independence_cs.json").write_text(
        json.dumps(json_compatible(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"  [grid] wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs=3, type=float, default=DEFAULT_LEVELS)
    parser.add_argument("--field-spacing-ratio", type=float, default=0.15)
    parser.add_argument("--total-time", type=float, default=30.0)
    parser.add_argument(
        "--sample-plane-fraction",
        type=float,
        default=0.25,
        help="sampling plane z/L (default: 0.25, matching the production figures)",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="do not run simulations; analyze three already-completed levels",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    levels = tuple(float(level) for level in args.levels)
    if any(level < 0.30 for level in levels):
        raise ValueError("resource guard: spacing ratios below 0.30 are not permitted")
    ratios = np.asarray(levels[:-1]) / np.asarray(levels[1:])
    if not np.allclose(ratios, ratios[0], rtol=0.02):
        raise ValueError("the three levels must use an approximately constant refinement ratio")
    if not 0.0 < args.field_spacing_ratio <= min(levels):
        raise ValueError("field spacing must be positive and no coarser than the finest particles")

    output_root = args.output_root.resolve()
    rows = []
    for spacing_ratio in levels:
        root = output_root / level_name(spacing_ratio)
        metadata = completed_metadata(root, spacing_ratio, args)
        if metadata is None:
            if args.analyze_only:
                print(f"  [grid] skip incomplete or missing {level_name(spacing_ratio)}")
                continue
            metadata = run_level(root, spacing_ratio, args)
        rows.append(analyze_level(root, spacing_ratio, metadata))
    if not rows:
        print("  [grid] no completed levels available; no report written")
        return 0
    orders = add_convergence_metrics(rows)
    self_metrics = self_convergence_metrics(output_root, rows, float(ratios[0]))
    write_results(output_root, rows, orders, self_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
