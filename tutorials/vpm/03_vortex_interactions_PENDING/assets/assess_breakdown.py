"""Summarize native seeded-breakdown evidence without reconstructing fields.

Particle mode metrics come from native HDF5 backups. Cross-plane asymmetry
comes from native ``cross_section`` SurfaceSampler VTS files. The two cadences
are kept separate deliberately: at the default dt, backups are 0.75 s apart
while cross sections are sampled every 0.30 s.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pyvista as pv

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from . import legacy_les as setup_les


ROOT = Path(__file__).resolve().parents[1]
MODE = 8
DEFAULT_BIN_COUNT = 128
REFERENCE_RADIUS = 1.0
DEFAULT_RUNS = (
    "cs_breakdown_baseline",
    "cs_breakdown_stretching_viscosity",
    "cs_breakdown_p_moments",
    "cs_breakdown_splitting",
)


def _centered_mode_estimate(
    position: np.ndarray,
    strength: np.ndarray,
    *,
    mode: int = MODE,
    reference_radius: float = REFERENCE_RADIUS,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> dict[str, float]:
    """Estimate centered axial/radial modes from native particle geometry.

    Particles are first reduced to strength-weighted conditional means in
    uniform azimuth bins.  The Fourier fit then gives every occupied azimuth
    bin equal weight, so nonuniform particle strength density does not itself
    become a geometric mode.  The intercept in the fit makes the result
    invariant to rigid axial translation and uniform radial expansion.  The
    estimator still measures a strength-weighted centerline proxy: it does not
    recover sub-bin structure or distinguish tube-shape modes from
    centerline modes when the native particle spacing is too coarse.
    """
    position = np.asarray(position, dtype=np.float64)
    strength = np.asarray(strength, dtype=np.float64)
    theta = np.mod(np.arctan2(position[:, 2], position[:, 1]), 2.0 * np.pi)
    radius = np.hypot(position[:, 1], position[:, 2])
    weight = np.linalg.norm(strength, axis=1)
    finite = np.isfinite(theta) & np.isfinite(radius) & np.isfinite(weight)
    finite &= weight > 0.0
    if not np.any(finite):
        return {
            "axial_mode8_amplitude": math.nan,
            "axial_mode8_phase": math.nan,
            "axial_mode8_real": math.nan,
            "axial_mode8_imag": math.nan,
            "radial_mode8_amplitude": math.nan,
            "radial_mode8_phase": math.nan,
            "radial_mode8_real": math.nan,
            "radial_mode8_imag": math.nan,
            "axial_center": math.nan,
            "radial_center": math.nan,
            "mode8_azimuth_coverage": 0.0,
            "mode8_weight_concentration": math.nan,
        }

    theta, radius, weight, position = (
        theta[finite],
        radius[finite],
        weight[finite],
        position[finite],
    )
    total_weight = float(np.sum(weight))
    axial_center = float(np.sum(weight * position[:, 0]) / total_weight)
    radial_center = float(np.sum(weight * radius) / total_weight)
    bins = np.minimum((theta * bin_count / (2.0 * np.pi)).astype(int), bin_count - 1)
    weight_by_bin = np.bincount(bins, weights=weight, minlength=bin_count)
    occupied = weight_by_bin > np.finfo(float).tiny
    coverage = float(np.count_nonzero(occupied) / bin_count)
    if np.count_nonzero(occupied) < max(2 * mode + 1, 16):
        return {
            "axial_mode8_amplitude": math.nan,
            "axial_mode8_phase": math.nan,
            "axial_mode8_real": math.nan,
            "axial_mode8_imag": math.nan,
            "radial_mode8_amplitude": math.nan,
            "radial_mode8_phase": math.nan,
            "radial_mode8_real": math.nan,
            "radial_mode8_imag": math.nan,
            "axial_center": axial_center / reference_radius,
            "radial_center": radial_center / reference_radius,
            "mode8_azimuth_coverage": coverage,
            "mode8_weight_concentration": math.nan,
        }

    axial_by_bin = np.bincount(bins, weights=weight * position[:, 0], minlength=bin_count)
    radial_by_bin = np.bincount(bins, weights=weight * radius, minlength=bin_count)
    axial_by_bin = axial_by_bin[occupied] / weight_by_bin[occupied]
    radial_by_bin = radial_by_bin[occupied] / weight_by_bin[occupied]
    # Use the same within-bin quadrature for the Fourier basis as for the
    # geometry. This avoids attenuating a known mode merely because a bin has
    # finite angular width.
    cosine_by_bin = (
        np.bincount(bins, weights=weight * np.cos(mode * theta), minlength=bin_count)[occupied]
        / weight_by_bin[occupied]
    )
    sine_by_bin = (
        np.bincount(bins, weights=weight * np.sin(mode * theta), minlength=bin_count)[occupied]
        / weight_by_bin[occupied]
    )
    design = np.column_stack((np.ones_like(cosine_by_bin), cosine_by_bin, sine_by_bin))

    def fit(values: np.ndarray) -> tuple[float, float, float, float]:
        values = values / reference_radius
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        # q = a cos(m theta) + b sin(m theta); the complex Fourier coefficient
        # is (a - i b)/2, so amplitude is sqrt(a²+b²).
        complex_coefficient = 0.5 * (coefficients[1] - 1j * coefficients[2])
        return (
            float(2.0 * abs(complex_coefficient)),
            float(np.angle(complex_coefficient)),
            float(complex_coefficient.real),
            float(complex_coefficient.imag),
        )

    # The binned arrays above are already centered by the fitted intercept; do
    # not subtract a particle-weighted mean before fitting, because the
    # unweighted azimuthal fit must remove rigid offsets itself.
    axial = fit(axial_by_bin)
    radial = fit(radial_by_bin)
    return {
        "axial_mode8_amplitude": axial[0],
        "axial_mode8_phase": axial[1],
        "axial_mode8_real": axial[2],
        "axial_mode8_imag": axial[3],
        "radial_mode8_amplitude": radial[0],
        "radial_mode8_phase": radial[1],
        "radial_mode8_real": radial[2],
        "radial_mode8_imag": radial[3],
        "axial_center": axial_center / reference_radius,
        "radial_center": radial_center / reference_radius,
        "mode8_azimuth_coverage": coverage,
        "mode8_weight_concentration": float(
            np.max(weight_by_bin[occupied]) / np.mean(weight_by_bin[occupied])
        ),
    }


def _legacy_mode_estimate(
    position: np.ndarray,
    strength: np.ndarray,
    *,
    mode: int = MODE,
    reference_radius: float = REFERENCE_RADIUS,
) -> tuple[float, float]:
    """Return the former absolute-coordinate estimator for control comparison."""
    theta = np.mod(np.arctan2(position[:, 2], position[:, 1]), 2.0 * np.pi)
    radius = np.hypot(position[:, 1], position[:, 2])
    weight = np.linalg.norm(strength, axis=1)
    denominator = max(float(np.sum(weight)), np.finfo(float).tiny)
    factor = np.exp(-1j * mode * theta)
    axial = np.sum(weight * position[:, 0] * factor) / denominator / reference_radius
    radial = np.sum(weight * radius * factor) / denominator / reference_radius
    return float(2.0 * abs(axial)), float(2.0 * abs(radial))


def _reflection_odd_fraction(values: np.ndarray) -> float:
    """Fraction odd under theta -> -theta; zero does not mean axisymmetry."""
    values = np.asarray(values, dtype=np.float64)
    mirrored = values[(-np.arange(len(values))) % len(values)]
    odd = 0.5 * (values - mirrored)
    denominator = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    return float(np.linalg.norm(odd) / denominator)


def _mirror_asymmetry(magnitude: np.ndarray) -> float:
    """Return the native cross-plane metric: odd reflection / even norm."""
    magnitude = np.asarray(magnitude, dtype=np.float64)
    even = 0.5 * (magnitude + magnitude[:, ::-1])
    odd = 0.5 * (magnitude - magnitude[:, ::-1])
    denominator = max(float(np.linalg.norm(even.ravel())), np.finfo(float).tiny)
    return float(np.linalg.norm(odd.ravel()) / denominator)


def _mode_rows(path: Path, *, reference_radius: float) -> list[dict[str, float | int | str]]:
    with h5py.File(path, "r") as file:
        particles = file["particles"]
        position = np.asarray(particles["position"][:], dtype=np.float64)
        strength = np.asarray(particles["vortex_strength"][:], dtype=np.float64)
        groups = np.asarray(particles["group_id"][:], dtype=np.int32)
        solver = file["solver"].attrs
        time = float(solver["time"])
        step = int(solver["step"])
        n_particles = int(solver["n_particles_total"])

    rows = []
    for group in sorted(np.unique(groups)):
        selected = groups == group
        estimate = _centered_mode_estimate(
            position[selected], strength[selected], reference_radius=reference_radius
        )
        rows.append(
            {
                "source": "particle_backup",
                "step": step,
                "time": time,
                "group_id": int(group),
                "n_particles": n_particles,
                **estimate,
                "field_mirror_asymmetry": math.nan,
            }
        )
    return rows


def _field_rows(path: Path, time: float) -> list[dict[str, float | int | str]]:
    grid = pv.read(path)
    dimensions = grid.dimensions
    raw = np.asarray(grid.point_data["vorticity"], dtype=np.float64)
    field = raw.reshape((dimensions[1], dimensions[0], 3)).transpose(1, 0, 2)
    magnitude = np.linalg.norm(field, axis=2)
    asymmetry = _mirror_asymmetry(magnitude)
    step = int(path.stem.rsplit("_", 1)[1])
    return [
        {
            "source": "cross_section_field",
            "step": step,
            "time": time,
            "group_id": -1,
            "n_particles": math.nan,
            "axial_mode8_amplitude": math.nan,
            "axial_mode8_phase": math.nan,
            "radial_mode8_amplitude": math.nan,
            "field_mirror_asymmetry": asymmetry,
        }
    ]


def _pvd_times(path: Path) -> dict[str, float]:
    """Read native sampler times rather than reconstructing them from dt."""
    if not path.exists():
        return {}
    root = ET.parse(path).getroot()
    return {
        dataset.attrib["file"]: float(dataset.attrib["timestep"])
        for dataset in root.findall(".//DataSet")
        if "file" in dataset.attrib and "timestep" in dataset.attrib
    }


def _analytic_controls() -> list[dict[str, float | int | str]]:
    """Exercise centering, weight-density rejection and reflection semantics."""
    count = 2048
    theta = 2.0 * np.pi * np.arange(count) / count
    weights = 1.0 + 0.45 * np.cos(MODE * theta) + 0.15 * np.sin(3.0 * theta)
    position = np.column_stack((np.zeros(count), np.cos(theta), np.sin(theta)))
    strength = np.column_stack((weights, np.zeros((count, 2))))

    cases = (
        ("rigid_translation", 2.3 + 0.0 * theta, 1.0 + 0.0 * theta, "axisymmetric"),
        (
            "known_axial_mode_plus_translation",
            2.3 + 0.05 * np.cos(MODE * theta),
            1.0 + 0.0 * theta,
            "mode",
        ),
        ("rigid_expansion", 0.0 * theta, 1.25 + 0.0 * theta, "axisymmetric"),
        (
            "known_radial_mode_plus_expansion",
            0.0 * theta,
            1.25 + 0.05 * np.cos(MODE * theta),
            "mode",
        ),
        (
            "mirror_even_cos8",
            0.05 * np.cos(MODE * theta),
            1.0 + 0.0 * theta,
            "non_axisymmetric_reflection_even",
        ),
        (
            "mirror_odd_sin8",
            0.05 * np.sin(MODE * theta),
            1.0 + 0.0 * theta,
            "non_axisymmetric_reflection_odd",
        ),
    )
    rows: list[dict[str, float | int | str]] = []
    for name, axial, radial, interpretation in cases:
        current = position.copy()
        current[:, 0] = axial
        current[:, 1] = radial * np.cos(theta)
        current[:, 2] = radial * np.sin(theta)
        estimate = _centered_mode_estimate(current, strength)
        legacy_axial, legacy_radial = _legacy_mode_estimate(current, strength)
        rows.append(
            {
                "control": name,
                "interpretation": interpretation,
                **estimate,
                "legacy_axial_mode8_amplitude": legacy_axial,
                "legacy_radial_mode8_amplitude": legacy_radial,
                "reflection_odd_fraction": _reflection_odd_fraction(axial),
                "field_mirror_asymmetry": math.nan,
            }
        )
    field_axis = np.linspace(-1.0, 1.0, 256)
    xx, zz = np.meshgrid(field_axis, field_axis, indexing="ij")
    field_theta = np.arctan2(zz, xx)
    envelope = np.exp(-2.0 * (xx**2 + zz**2))
    for name, field, interpretation in (
        ("field_axisymmetric", envelope, "axisymmetric"),
        (
            "field_non_axisymmetric_mirror_even_cos8",
            envelope * (1.0 + 0.35 * np.cos(MODE * field_theta)),
            "non_axisymmetric_reflection_even",
        ),
        (
            "field_non_axisymmetric_mirror_odd_sin8",
            envelope * (1.0 + 0.35 * np.sin(MODE * field_theta)),
            "non_axisymmetric_reflection_odd",
        ),
    ):
        rows.append(
            {
                "control": name,
                "interpretation": interpretation,
                "axial_mode8_amplitude": math.nan,
                "axial_mode8_phase": math.nan,
                "axial_mode8_real": math.nan,
                "axial_mode8_imag": math.nan,
                "radial_mode8_amplitude": math.nan,
                "radial_mode8_phase": math.nan,
                "radial_mode8_real": math.nan,
                "radial_mode8_imag": math.nan,
                "legacy_axial_mode8_amplitude": math.nan,
                "legacy_radial_mode8_amplitude": math.nan,
                "reflection_odd_fraction": math.nan,
                "field_mirror_asymmetry": _mirror_asymmetry(field),
            }
        )
    return rows


def _native_initialization_controls() -> list[dict[str, float | int | str]]:
    """Apply the estimator to the native seeded geometry without advancing it."""
    case = setup_les.build_case(
        "baseline",
        scenario="seeded_breakdown",
        compute_device="CPU",
        steps=1,
        wall_minutes=1,
        case_name="assessment_native_initialization",
    )
    rows: list[dict[str, float | int | str]] = []
    for group, ring in enumerate(case.initial_conditions):
        particles = ring.build()
        estimate = _centered_mode_estimate(
            particles.position,
            particles.vortex_strength,
            reference_radius=setup_les.RING_RADIUS,
        )
        rows.append(
            {
                "control": f"native_seed_initial_group_{group}",
                "interpretation": "native_seed",
                **estimate,
                "legacy_axial_mode8_amplitude": math.nan,
                "legacy_radial_mode8_amplitude": math.nan,
                "reflection_odd_fraction": math.nan,
                "field_mirror_asymmetry": math.nan,
            }
        )
    return rows


def _write_report(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "breakdown_metrics.csv"
    fields = (
        "run",
        "source",
        "step",
        "time",
        "group_id",
        "n_particles",
        "axial_mode8_amplitude",
        "axial_mode8_phase",
        "axial_mode8_real",
        "axial_mode8_imag",
        "radial_mode8_amplitude",
        "radial_mode8_phase",
        "radial_mode8_real",
        "radial_mode8_imag",
        "axial_center",
        "radial_center",
        "mode8_azimuth_coverage",
        "mode8_weight_concentration",
        "field_mirror_asymmetry",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Native seeded-breakdown metrics",
        "",
        "Particle mode values and field-plane asymmetry retain their native cadences; no interpolation or particle-field reconstruction is used.",
        "The centered estimator fits the complex mode to strength-weighted conditional means in uniform azimuth bins, removing rigid axial translation and uniform radial expansion. Its values are normalized by the native R0 from metadata.",
        "Particle backups are nominally 0.75 s apart. This report's mirror metric uses the native `cross_section` PVD timestamps (nominally 0.30 s); the separate `core_section` fields are nominally 0.15 s and are not substituted for cross-section samples.",
        "",
        "| run | max axial mode-8 / R0 | final axial mode-8 / R0 | max field mirror asymmetry | final N |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in sorted({str(row["run"]) for row in rows}):
        particle = [row for row in rows if row["run"] == run and row["source"] == "particle_backup"]
        field = [
            row for row in rows if row["run"] == run and row["source"] == "cross_section_field"
        ]
        amplitudes = [float(row["axial_mode8_amplitude"]) for row in particle]
        final_time = max(float(row["time"]) for row in particle)
        final = [row for row in particle if float(row["time"]) == final_time]
        asymmetries = [float(row["field_mirror_asymmetry"]) for row in field]
        count = int(max(float(row["n_particles"]) for row in final))
        lines.append(
            f"| {run} | {max(amplitudes, default=math.nan):.5g} | "
            f"{max((float(row['axial_mode8_amplitude']) for row in final), default=math.nan):.5g} | "
            f"{max(asymmetries, default=math.nan):.5g} | {count} |"
        )
    (output / "breakdown_metrics.md").write_text("\n".join(lines) + "\n")

    history = [
        row
        for row in rows
        if row["source"] == "particle_backup"
        and not math.isnan(float(row["axial_mode8_amplitude"]))
    ]
    history_lines = [
        "",
        "## Full complex particle-mode history",
        "",
        "The axial and radial complex mode histories are both retained; an axial-only decrease is not interpreted as modal damping.",
        "",
        "| run | time (s) | group | axial Re | axial Im | |axial|/R0 | radial Re | radial Im | |radial|/R0 | coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        history, key=lambda item: (str(item["run"]), float(item["time"]), int(item["group_id"]))
    ):
        history_lines.append(
            f"| {row['run']} | {float(row['time']):.6g} | {int(row['group_id'])} | "
            f"{float(row['axial_mode8_real']):.6g} | {float(row['axial_mode8_imag']):.6g} | "
            f"{float(row['axial_mode8_amplitude']):.6g} | {float(row['radial_mode8_real']):.6g} | "
            f"{float(row['radial_mode8_imag']):.6g} | {float(row['radial_mode8_amplitude']):.6g} | "
            f"{float(row['mode8_azimuth_coverage']):.4f} |"
        )
    with (output / "breakdown_metrics.md").open("a") as stream:
        stream.write("\n".join(history_lines) + "\n")


def _write_controls(controls: list[dict[str, float | int | str]], output: Path) -> None:
    fields = (
        "control",
        "interpretation",
        "axial_mode8_amplitude",
        "axial_mode8_phase",
        "axial_mode8_real",
        "axial_mode8_imag",
        "radial_mode8_amplitude",
        "radial_mode8_phase",
        "radial_mode8_real",
        "radial_mode8_imag",
        "legacy_axial_mode8_amplitude",
        "legacy_radial_mode8_amplitude",
        "reflection_odd_fraction",
        "field_mirror_asymmetry",
        "axial_center",
        "radial_center",
        "mode8_azimuth_coverage",
        "mode8_weight_concentration",
    )
    with (output / "assessment_controls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(controls)
    lines = [
        "# Assessment positive controls",
        "",
        "The centered estimator is tested against rigid offsets, known modes and nonuniform positive strength weights. The legacy columns show why absolute-coordinate weighted integration was not retained.",
        "",
        "| control | interpretation | centered axial / R0 | centered radial / R0 | legacy axial / R0 | legacy radial / R0 | reflection-odd fraction | field mirror asymmetry |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in controls:
        lines.append(
            f"| {row['control']} | {row['interpretation']} | "
            f"{float(row['axial_mode8_amplitude']):.6g} | {float(row['radial_mode8_amplitude']):.6g} | "
            f"{float(row['legacy_axial_mode8_amplitude']):.6g} | "
            f"{float(row['legacy_radial_mode8_amplitude']):.6g} | "
            f"{float(row['reflection_odd_fraction']):.6g} | "
            f"{float(row['field_mirror_asymmetry']):.6g} |"
        )
    lines.extend(
        [
            "",
            "A reflection-even cos(8 theta) mode is non-axisymmetric but has zero reflection-odd fraction. Therefore the cross-section mirror metric is a diagnostic of reflection-odd content, not a necessary condition for non-axisymmetry or breakdown.",
        ]
    )
    (output / "assessment_controls.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "cs_breakdown")
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for run in args.runs:
        solution = ROOT / "solution" / run
        samples = ROOT / "samples" / run
        metadata = solution / "vpm_metadata.json"
        if not metadata.exists():
            raise SystemExit(f"missing native metadata: {metadata}")
        import json

        configuration = json.loads(metadata.read_text())["configuration"]
        reference_radius = float(configuration["initial_conditions"][0]["radius"])
        backups = sorted(solution.glob("vpm_*.h5"))
        fields = sorted(samples.glob("cross_section_*.vts"))
        field_times = _pvd_times(samples / "cross_section.pvd")
        if not backups:
            raise SystemExit(f"missing native backups: {solution}")
        rows.extend(
            {"run": run, **row}
            for path in backups
            for row in _mode_rows(path, reference_radius=reference_radius)
        )
        rows.extend(
            {"run": run, **row}
            for path in fields
            for row in _field_rows(
                path,
                field_times.get(path.name, math.nan),
            )
        )

    _write_report(rows, args.output)
    _write_controls(_analytic_controls() + _native_initialization_controls(), args.output)
    print(f"wrote {args.output / 'breakdown_metrics.csv'}")


if __name__ == "__main__":
    main()
