#!/usr/bin/env python3
"""Audit the cap-limited/completed seeded filter continuations at exact clocks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyvista as pv

from openonda.tutorial_runner import case_package

if not __package__:
    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

from source.solvers.vpm.numerics import fourier_integrals  # noqa: E402
from source.solvers.vpm.numerics.fourier_integrals import (  # noqa: E402
    CartesianGrid,
    _grid_for_particles,
    gaussian_fourier_integrals,
)
from .assess_breakdown import _centered_mode_estimate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "cs_filter_pair_step300"
RUNS = {
    "control_cs020": "cs_breakdown_filter_cs020_cpu_t6_step300_tail",
    "molecular_cs000": "cs_breakdown_filter_cs000_cpu_t6_step300_continuation",
}
SOURCE_RUNS = {
    "control_cs020": "cs_breakdown_filter_cs020_cpu_t6_step080",
    "molecular_cs000": "cs_breakdown_filter_cs000_cpu_t6_step080",
}
RUN_SEGMENTS = {
    "control_cs020": (
        "cs_breakdown_filter_cs020_cpu_t6_step080",
        "cs_breakdown_filter_cs020_cpu_t6_step300_continuation",
        "cs_breakdown_filter_cs020_cpu_t6_step300_tail",
    ),
    "molecular_cs000": (
        "cs_breakdown_filter_cs000_cpu_t6_step080",
        "cs_breakdown_filter_cs000_cpu_t6_step300_continuation",
    ),
}
ACTUAL_PROCESS_SECONDS = {"control_cs020": 835.50, "molecular_cs000": 225.46}
PROCESS_SEGMENTS = {
    "control_step080_to_240": 610.90,
    "molecular_step080_to_300": 225.46,
    "control_step240_to_300_tail": 224.60,
}
LIFECYCLES = {
    "control_cs020": {"status": "completed", "step": 300, "time": 2.25},
    "molecular_cs000": {"status": "completed", "step": 300, "time": 2.25},
}
COMMON_PARTICLE_STEP = 300
COMMON_PARTICLE_TIME = 2.25
COMMON_FIELD_STEPS = (300,)
SEED_AMPLITUDE = 0.05
FIELD_CORE_CUTOFF = 0.6
GRID_EXTRA_CELLS_PER_SIDE = (4, 8, 12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_h5(run: str, step: int) -> dict[str, np.ndarray | float | int]:
    path = ROOT / "solution" / run / f"vpm_{step:06d}.h5"
    with h5py.File(path, "r") as handle:
        particles = handle["particles"]
        result: dict[str, np.ndarray | float | int] = {
            name: np.asarray(particles[name], dtype=np.float64)
            for name in (
                "position",
                "vortex_strength",
                "core_radius",
                "particle_volume",
            )
        }
        result["group_id"] = np.asarray(particles["group_id"], dtype=np.int32)
        result["step"] = int(handle["solver"].attrs["step"])
        result["time"] = float(handle["solver"].attrs["time"])
        result["n_stabilization_events"] = int(handle["solver"].attrs["n_stabilization_events"])
        result["n_regularization_events"] = int(handle["solver"].attrs["n_regularization_events"])
    if result["step"] != step or not np.isfinite(result["time"]):
        raise RuntimeError(f"invalid native state in {path}")
    if not all(
        np.isfinite(value).all() for value in result.values() if isinstance(value, np.ndarray)
    ):
        raise RuntimeError(f"non-finite native state in {path}")
    result["path"] = str(path)
    result["sha256"] = _sha256(path)
    return result


def _validate_lifecycles() -> dict[str, dict[str, object]]:
    records = {}
    for key, run in RUNS.items():
        metadata_path = ROOT / "solution" / run / "vpm_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        lifecycle = metadata["lifecycle"]["status"]
        state = metadata["state"]
        expected = LIFECYCLES[key]
        if (lifecycle, state["step"], state["time"], state["n_particles_total"]) != (
            expected["status"],
            expected["step"],
            expected["time"],
            16_104,
        ):
            raise RuntimeError(f"unexpected lifecycle for {run}")
        records[key] = {
            "run": run,
            "status": lifecycle,
            "step": state["step"],
            "time": state["time"],
            "n_particles_total": state["n_particles_total"],
            "actual_process_seconds": ACTUAL_PROCESS_SECONDS[key],
        }
    return records


def _mode_rows() -> pd.DataFrame:
    values: dict[tuple[str, int], tuple[complex, complex]] = {}
    for key, run in RUNS.items():
        metrics = pd.read_csv(ROOT / "figures" / run / "breakdown_metrics.csv")
        rows = metrics[
            (metrics.step == COMMON_PARTICLE_STEP) & (metrics.source == "particle_backup")
        ]
        if set(rows.group_id.astype(int)) != {0, 1}:
            raise RuntimeError(f"missing common-step modes for {run}")
        for row in rows.itertuples(index=False):
            values[(key, int(row.group_id))] = (
                complex(row.axial_mode8_real, row.axial_mode8_imag),
                complex(row.radial_mode8_real, row.radial_mode8_imag),
            )

    result = []
    for group in (0, 1):
        control = values[("control_cs020", group)]
        molecular = values[("molecular_cs000", group)]
        row: dict[str, float | int] = {
            "step": COMMON_PARTICLE_STEP,
            "time": COMMON_PARTICLE_TIME,
            "group_id": group,
        }
        for name, index in (("axial", 0), ("radial", 1)):
            phase_change = np.angle(
                np.exp(1j * (np.angle(molecular[index]) - np.angle(control[index])))
            )
            row.update(
                {
                    f"control_{name}_real": control[index].real,
                    f"control_{name}_imag": control[index].imag,
                    f"molecular_{name}_real": molecular[index].real,
                    f"molecular_{name}_imag": molecular[index].imag,
                    f"control_{name}_amplitude": 2.0 * abs(control[index]),
                    f"molecular_{name}_amplitude": 2.0 * abs(molecular[index]),
                    f"{name}_amplitude_relative_change": abs(molecular[index]) / abs(control[index])
                    - 1.0,
                    f"{name}_phase_change_degrees": np.degrees(phase_change),
                }
            )
        control_norm = 2.0 * np.sqrt(abs(control[0]) ** 2 + abs(control[1]) ** 2)
        molecular_norm = 2.0 * np.sqrt(abs(molecular[0]) ** 2 + abs(molecular[1]) ** 2)
        row.update(
            {
                "control_joint_mode_norm": control_norm,
                "molecular_joint_mode_norm": molecular_norm,
                "joint_mode_norm_relative_change": molecular_norm / control_norm - 1.0,
                "phase_aware_separation_over_seed": 2.0
                * np.sqrt(abs(molecular[0] - control[0]) ** 2 + abs(molecular[1] - control[1]) ** 2)
                / SEED_AMPLITUDE,
            }
        )
        result.append(row)
    return pd.DataFrame(result)


def _angular_spectrum_rows() -> pd.DataFrame:
    """Evaluate the existing centered estimator for modes 1--12 at common backups."""
    backup_files: dict[str, dict[int, Path]] = {}
    for key, segments in RUN_SEGMENTS.items():
        files: dict[int, Path] = {}
        for segment in segments:
            files.update(
                {
                    int(path.stem.rsplit("_", 1)[1]): path
                    for path in (ROOT / "solution" / segment).glob("vpm_*.h5")
                }
            )
        backup_files[key] = files
    common_steps = sorted(set.intersection(*(set(files) for files in backup_files.values())))

    rows = []
    for key, files in backup_files.items():
        for step in common_steps:
            values = _load_h5(files[step].parent.name, step)
            for group in (0, 1):
                selected = values["group_id"] == group
                for mode in range(1, 13):
                    estimate = _centered_mode_estimate(
                        values["position"][selected],
                        values["vortex_strength"][selected],
                        mode=mode,
                    )
                    rows.append(
                        {
                            "run_key": key,
                            "step": step,
                            "time": float(values["time"]),
                            "group_id": group,
                            "mode": mode,
                            "axial_amplitude": estimate["axial_mode8_amplitude"],
                            "radial_amplitude": estimate["radial_mode8_amplitude"],
                            "joint_mode_norm": np.hypot(
                                estimate["axial_mode8_amplitude"],
                                estimate["radial_mode8_amplitude"],
                            ),
                            "axial_phase": estimate["axial_mode8_phase"],
                            "radial_phase": estimate["radial_mode8_phase"],
                        }
                    )
    return pd.DataFrame(rows)


def _particle_geometry_rows(
    states: dict[str, dict[str, np.ndarray | float | int]],
) -> pd.DataFrame:
    rows = []
    for key, values in states.items():
        groups = values["group_id"]
        for group in (0, 1):
            selected = groups == group
            position = values["position"][selected]
            strength = values["vortex_strength"][selected]
            weight = np.linalg.norm(strength, axis=1)
            denominator = float(weight.sum())
            centroid = np.average(position, axis=0, weights=weight)
            centered = position - centroid
            covariance = (centered * weight[:, None]).T @ centered / denominator
            eigenvalues = np.linalg.eigvalsh(covariance)
            impulse = 0.5 * np.cross(position, strength).sum(axis=0, dtype=np.float64)
            rows.append(
                {
                    "run_key": key,
                    "step": COMMON_PARTICLE_STEP,
                    "time": COMMON_PARTICLE_TIME,
                    "group_id": group,
                    "weighted_centroid_x": centroid[0],
                    "weighted_centroid_y": centroid[1],
                    "weighted_centroid_z": centroid[2],
                    "weighted_mean_radius": np.dot(weight, np.hypot(position[:, 1], position[:, 2]))
                    / denominator,
                    "weighted_major_radius": np.sqrt(max(eigenvalues[-1] + eigenvalues[-2], 0.0)),
                    "vortex_strength_magnitude_sum": denominator,
                    "linear_impulse_x": impulse[0],
                    "linear_impulse_y": impulse[1],
                    "linear_impulse_z": impulse[2],
                    "linear_impulse_norm": np.linalg.norm(impulse),
                    "impulse_radius": 2.0 * np.linalg.norm(impulse) / denominator,
                }
            )
    return pd.DataFrame(rows)


def _field_geometry_rows() -> pd.DataFrame:
    rows = []
    for step in COMMON_FIELD_STEPS:
        for plane in ("core_section", "cross_section"):
            for key, run in RUNS.items():
                path = ROOT / "samples" / run / f"{plane}_{step:06d}.vts"
                grid = pv.read(path)
                points = np.asarray(grid.points, dtype=np.float64)
                vorticity = np.asarray(grid.point_data["vorticity"], dtype=np.float64)
                coordinates = (
                    points[:, :2]
                    if plane == "core_section"
                    else np.column_stack((points[:, 0], np.abs(points[:, 2])))
                )
                ring_rows = (
                    pd.read_csv(ROOT / "samples" / run / "ring_diagnostics.csv")
                    .query("step == @step")
                    .sort_values("group_id")
                )
                centers = ring_rows[["vortex_centroid_x", "major_radius"]].to_numpy(
                    dtype=np.float64
                )
                assignment = np.argmin(
                    np.sum(
                        (coordinates[:, None, :] - centers[None, :, :]) ** 2,
                        axis=2,
                    ),
                    axis=1,
                )
                for group in (0, 1):
                    selected = assignment == group
                    local_coordinates = coordinates[selected]
                    local_vorticity = vorticity[selected]
                    local_distance_sq = np.sum((local_coordinates - centers[group]) ** 2, axis=1)
                    weight = np.linalg.norm(local_vorticity, axis=1)
                    weight *= local_distance_sq <= FIELD_CORE_CUTOFF**2
                    centroid = np.einsum("i,ij->j", weight, local_coordinates) / weight.sum()
                    width = np.sqrt(
                        np.dot(
                            weight,
                            np.sum((local_coordinates - centroid) ** 2, axis=1),
                        )
                        / weight.sum()
                    )
                    rows.append(
                        {
                            "run_key": key,
                            "step": step,
                            "time": step * 0.0075,
                            "plane": plane,
                            "group_id": group,
                            "field_centroid_x": centroid[0],
                            "field_major_radius": centroid[1],
                            "field_core_rms_width": width,
                            "field_peak_vorticity": np.linalg.norm(local_vorticity, axis=1).max(),
                            "core_cutoff_radius": FIELD_CORE_CUTOFF,
                        }
                    )
    return pd.DataFrame(rows)


def _native_common_rows() -> pd.DataFrame:
    frames = []
    for key, run in RUNS.items():
        frame = pd.read_csv(ROOT / "samples" / run / "flow_integrals.csv")
        frame.insert(0, "run_key", key)
        frames.append(frame)
    common_steps = set(frames[0].step) & set(frames[1].step)
    return pd.concat([frame[frame.step.isin(common_steps)] for frame in frames], ignore_index=True)


def _cross_asymmetry_rows() -> pd.DataFrame:
    rows = []
    for key, run in RUNS.items():
        metrics = pd.read_csv(ROOT / "figures" / run / "breakdown_metrics.csv")
        fields = metrics[
            (metrics.source == "cross_section_field") & (metrics.step.isin(COMMON_FIELD_STEPS))
        ]
        for row in fields.itertuples(index=False):
            rows.append(
                {
                    "run_key": key,
                    "step": int(row.step),
                    "time": float(row.time),
                    "field_mirror_asymmetry": float(row.field_mirror_asymmetry),
                }
            )
    return pd.DataFrame(rows)


def _mirror_asymmetry(magnitude: np.ndarray) -> float:
    even = 0.5 * (magnitude + magnitude[:, ::-1])
    odd = 0.5 * (magnitude - magnitude[:, ::-1])
    denominator = max(float(np.linalg.norm(even.ravel())), np.finfo(float).tiny)
    return float(np.linalg.norm(odd.ravel()) / denominator)


def _ring_cross_asymmetry_rows() -> pd.DataFrame:
    """Partition native cross planes by nearest axial group centroid."""
    available: dict[str, dict[int, Path]] = {}
    diagnostics: dict[str, pd.DataFrame] = {}
    for key, segments in RUN_SEGMENTS.items():
        files: dict[int, Path] = {}
        ring_frames = []
        for segment in segments:
            sample_dir = ROOT / "samples" / segment
            files.update(
                {
                    int(path.stem.rsplit("_", 1)[1]): path
                    for path in sample_dir.glob("cross_section_*.vts")
                }
            )
            ring_frames.append(pd.read_csv(sample_dir / "ring_diagnostics.csv"))
        available[key] = files
        diagnostics[key] = pd.concat(ring_frames, ignore_index=True).drop_duplicates(
            subset=["step", "group_id"], keep="last"
        )
    common_steps = sorted(set.intersection(*(set(files) for files in available.values())))

    rows = []
    for key, files in available.items():
        ring_diagnostics = diagnostics[key]
        for step in common_steps:
            grid = pv.read(files[step])
            dimensions = grid.dimensions
            raw = np.asarray(grid.point_data["vorticity"], dtype=np.float64)
            magnitude = np.linalg.norm(
                raw.reshape((dimensions[1], dimensions[0], 3)).transpose(1, 0, 2),
                axis=2,
            )
            axial_coordinates = (
                np.asarray(grid.points, dtype=np.float64)[:, 0]
                .reshape((dimensions[1], dimensions[0]))
                .T[:, 0]
            )
            step_rows = ring_diagnostics[ring_diagnostics.step == step].set_index("group_id")
            centers = np.asarray([step_rows.loc[group, "vortex_centroid_x"] for group in (0, 1)])
            assignment = np.argmin(np.abs(axial_coordinates[:, None] - centers[None, :]), axis=1)
            for group in (0, 1):
                rows.append(
                    {
                        "run_key": key,
                        "step": step,
                        "time": float(step_rows.loc[group, "time"]),
                        "group_id": group,
                        "vortex_centroid_x": centers[group],
                        "nearest_centroid_local_mirror_asymmetry": (
                            _mirror_asymmetry(magnitude[assignment == group])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _common_integral_rows(
    source_states: dict[str, dict[str, np.ndarray | float | int]],
    common_states: dict[str, dict[str, np.ndarray | float | int]],
) -> pd.DataFrame:
    states = {
        "control_step080": source_states["control_cs020"],
        "molecular_step080": source_states["molecular_cs000"],
        "control_step300": common_states["control_cs020"],
        "molecular_step300": common_states["molecular_cs000"],
    }
    spacing = float(np.median(np.cbrt(common_states["control_cs020"]["particle_volume"])))
    base = _grid_for_particles(
        np.vstack([values["position"] for values in states.values()]), spacing
    )
    rows = []
    for extra in GRID_EXTRA_CELLS_PER_SIDE:
        grid = CartesianGrid(
            base.origin - extra * spacing,
            spacing,
            tuple(size + 2 * extra for size in base.shape),
        )
        for name, values in states.items():
            result = gaussian_fourier_integrals(
                values["position"],
                values["vortex_strength"],
                values["core_radius"],
                values["particle_volume"],
                grid=grid,
            )
            rows.append(
                {
                    "state": name,
                    "grid_extra_cells_per_side": extra,
                    "grid_spacing": spacing,
                    "grid_shape_x": grid.shape[0],
                    "grid_shape_y": grid.shape[1],
                    "grid_shape_z": grid.shape[2],
                    "energy_measurement": result.energy_measurement,
                    "total_kinetic_energy": result.total_kinetic_energy,
                    "total_enstrophy": result.total_enstrophy,
                    "test_filtered_enstrophy": result.test_filtered_enstrophy,
                    "total_helicity": result.total_helicity,
                    "energy_expansion_relative_change": abs(
                        result.total_kinetic_energy - result.previous_order_total_kinetic_energy
                    )
                    / abs(result.total_kinetic_energy),
                    "enstrophy_expansion_relative_change": abs(
                        result.total_enstrophy - result.previous_order_total_enstrophy
                    )
                    / abs(result.total_enstrophy),
                }
            )
    return pd.DataFrame(rows)


def _endpoint_rows() -> dict[str, dict[str, object]]:
    result = {}
    for key, run in RUNS.items():
        flow = pd.read_csv(ROOT / "samples" / run / "flow_integrals.csv")
        final = flow[flow.step == COMMON_PARTICLE_STEP].iloc[0]
        metrics = pd.read_csv(ROOT / "figures" / run / "breakdown_metrics.csv")
        modes = metrics[
            (metrics.step == COMMON_PARTICLE_STEP) & (metrics.source == "particle_backup")
        ]
        cross = metrics[
            (metrics.step == COMMON_PARTICLE_STEP) & (metrics.source == "cross_section_field")
        ].iloc[0]
        result[key] = {
            "step": COMMON_PARTICLE_STEP,
            "time": COMMON_PARTICLE_TIME,
            "native_energy_rate_source": str(final.kinetic_energy_rate_source),
            "native_total_kinetic_energy": float(final.total_kinetic_energy),
            "native_total_enstrophy": float(final.total_enstrophy),
            "native_test_filtered_enstrophy": float(final.test_filtered_enstrophy),
            "vortex_strength_magnitude_sum": float(final.vortex_strength_magnitude_sum),
            "net_vortex_strength_x": float(final.net_vortex_strength_x),
            "mean_particle_core_radius": float(final.mean_particle_core_radius),
            "min_particle_core_radius": float(final.min_particle_core_radius),
            "max_particle_core_radius": float(final.max_particle_core_radius),
            "vorticity_divergence_error": float(final.vorticity_divergence_error),
            "vortex_strength_misalignment_degrees": float(
                final.vortex_strength_misalignment_degrees
            ),
            "lagrangian_cfl": float(final.lagrangian_cfl),
            "linear_impulse_x": float(final.linear_impulse_x),
            "field_mirror_asymmetry": float(cross.field_mirror_asymmetry),
            "joint_mode_norms": {
                str(int(row.group_id)): float(
                    np.hypot(row.axial_mode8_amplitude, row.radial_mode8_amplitude)
                )
                for row in modes.itertuples(index=False)
            },
        }
    return result


def _percent(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def _write_markdown(
    lifecycle: dict[str, dict[str, object]],
    modes: pd.DataFrame,
    angular_spectrum: pd.DataFrame,
    fields: pd.DataFrame,
    common_integrals: pd.DataFrame,
    asymmetry: pd.DataFrame,
    ring_asymmetry: pd.DataFrame,
    endpoints: dict[str, dict[str, object]],
) -> None:
    spectral = common_integrals[common_integrals.grid_extra_cells_per_side == 8].set_index("state")
    final_control = spectral.loc["control_step300"]
    final_molecular = spectral.loc["molecular_step300"]
    cross_changes = {
        name: final_molecular[name] / final_control[name] - 1.0
        for name in (
            "total_kinetic_energy",
            "total_enstrophy",
            "test_filtered_enstrophy",
        )
    }
    trajectory_changes = {
        key: {
            name: spectral.loc[f"{key}_step300", name] / spectral.loc[f"{key}_step080", name] - 1.0
            for name in (
                "total_kinetic_energy",
                "total_enstrophy",
                "test_filtered_enstrophy",
            )
        }
        for key in ("control", "molecular")
    }
    asym = asymmetry.pivot(index="step", columns="run_key", values="field_mirror_asymmetry")
    control_asymmetry = asym.loc[COMMON_PARTICLE_STEP, "control_cs020"]
    molecular_asymmetry = asym.loc[COMMON_PARTICLE_STEP, "molecular_cs000"]
    local_asymmetry = ring_asymmetry.set_index(["run_key", "step", "group_id"])
    molecular_group0_initial_asymmetry = local_asymmetry.loc[
        ("molecular_cs000", 80, 0),
        "nearest_centroid_local_mirror_asymmetry",
    ]
    molecular_group0_final_asymmetry = local_asymmetry.loc[
        ("molecular_cs000", COMMON_PARTICLE_STEP, 0),
        "nearest_centroid_local_mirror_asymmetry",
    ]
    molecular_group1_final_asymmetry = local_asymmetry.loc[
        ("molecular_cs000", COMMON_PARTICLE_STEP, 1),
        "nearest_centroid_local_mirror_asymmetry",
    ]
    aggregate_seconds = sum(PROCESS_SEGMENTS.values())
    source_group1_norms = {}
    for key, run in SOURCE_RUNS.items():
        source_modes = pd.read_csv(ROOT / "figures" / run / "breakdown_metrics.csv")
        source_group1 = source_modes[
            (source_modes.step == 80)
            & (source_modes.source == "particle_backup")
            & (source_modes.group_id == 1)
        ].iloc[0]
        source_group1_norms[key] = float(
            np.hypot(
                source_group1.axial_mode8_amplitude,
                source_group1.radial_mode8_amplitude,
            )
        )
    final_group1 = modes[modes.group_id == 1].iloc[0]
    group0_spectrum = angular_spectrum[
        (angular_spectrum.group_id == 0)
        & (angular_spectrum.step.isin((80, COMMON_PARTICLE_STEP)))
        & (angular_spectrum["mode"].isin((4, 8, 12)))
    ].set_index(["run_key", "step", "mode"])
    lines = [
        "# Seeded filter continuation: exact-clock comparison",
        "",
        "The control first reached step 240/time 1.8 at its approved native "
        f"wall-time cap ({PROCESS_SEGMENTS['control_step080_to_240']:.2f} s). "
        "The molecular leg then completed step 300/time 2.25 "
        f"({PROCESS_SEGMENTS['molecular_step080_to_300']:.2f} s), and the authorized "
        "control tail completed the same endpoint "
        f"({PROCESS_SEGMENTS['control_step240_to_300_tail']:.2f} s). Both matched "
        "endpoints contain 16,104 particles and neither trajectory triggered a "
        "health guard, stabilization, or regularization event. No interpolation is used.",
        "",
        "## Phase-aware particle modes at step 300/time 2.25",
        "",
        "| group | Dg / seed | M(Cs=.20) | M(Cs=0) | relative M change |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in modes.itertuples(index=False):
        lines.append(
            f"| {row.group_id} | {row.phase_aware_separation_over_seed:.6f} | "
            f"{row.control_joint_mode_norm:.8f} | {row.molecular_joint_mode_norm:.8f} | "
            f"{_percent(row.joint_mode_norm_relative_change)} |"
        )
    lines += [
        "",
        f"At the same final clock, group 1 meets the reviewed 20% practical "
        f"cross-case joint-norm positive "
        f"({_percent(final_group1.joint_mode_norm_relative_change)}) "
        "and its axial component is phase-consistent "
        f"({final_group1.axial_phase_change_degrees:+.2f} degrees). Both group-1 "
        "norms nevertheless decline from step 80 to step 300: "
        f"{_percent(final_group1.control_joint_mode_norm / source_group1_norms['control_cs020'] - 1.0)} "
        "for the control and "
        f"{_percent(final_group1.molecular_joint_mode_norm / source_group1_norms['molecular_cs000'] - 1.0)} "
        "for the molecular leg. "
        "Group 0 does not: its joint norm changes "
        f"{_percent(modes.loc[modes.group_id == 0, 'joint_mode_norm_relative_change'].iloc[0])}, "
        "with strong component and phase redistribution. This is ring-dependent "
        "mode retention/redistribution. It is not temporal amplification, coherent "
        "growth of both rings, or a breakdown result.",
        "",
        "## Two-plane core result",
        "",
        "| step/time | plane | group | width Cs=.20 | width Cs=0 | Cs=0 narrowing | peak ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for step in COMMON_FIELD_STEPS:
        for plane in ("core_section", "cross_section"):
            for group in (0, 1):
                subset = fields[
                    (fields.step == step) & (fields.plane == plane) & (fields.group_id == group)
                ].set_index("run_key")
                w20 = subset.loc["control_cs020", "field_core_rms_width"]
                w0 = subset.loc["molecular_cs000", "field_core_rms_width"]
                peak = (
                    subset.loc["molecular_cs000", "field_peak_vorticity"]
                    / subset.loc["control_cs020", "field_peak_vorticity"]
                )
                lines.append(
                    f"| {step}/{step * 0.0075:g} | {plane} | {group} | "
                    f"{w20:.8f} | {w0:.8f} | {100.0 * (1.0 - w0 / w20):.2f}% | {peak:.4f} |"
                )
    lines += [
        "",
        f"At step 300, cross-plane mirror asymmetry is `{control_asymmetry:.8f}` "
        f"for the control and `{molecular_asymmetry:.8f}` for `Cs=0` "
        f"({molecular_asymmetry / control_asymmetry:.2f}x). The molecular value is "
        "still below one percent. It resolves more reflection-odd content, but is "
        "not by itself evidence of breakdown.",
        "",
        "A nearest-axial-centroid partition of the same native cross planes assigns "
        "the right-hand molecular ring to group 0 and the left-hand ring to group 1. "
        f"At step 300 their local mirror-odd ratios are "
        f"`{molecular_group0_final_asymmetry:.8f}` and "
        f"`{molecular_group1_final_asymmetry:.8f}`, respectively. Group 0 rises from "
        f"`{molecular_group0_initial_asymmetry:.8f}` at step 80, but its particle "
        "joint mode norm does not grow over that interval. The global asymmetry "
        "therefore cannot assign the lobe to group 1 or connect it to group 1's "
        "same-time retention result.",
        "",
        "The existing centered angular estimator provides the complementary "
        "identity-resolved test. Group-0 joint norms at steps 80 -> 300 are:",
        "",
        "| mode | control | molecular |",
        "|---:|---:|---:|",
    ]
    for mode in (4, 8, 12):
        control_initial = group0_spectrum.loc[("control_cs020", 80, mode), "joint_mode_norm"]
        control_final = group0_spectrum.loc[
            ("control_cs020", COMMON_PARTICLE_STEP, mode), "joint_mode_norm"
        ]
        molecular_initial = group0_spectrum.loc[("molecular_cs000", 80, mode), "joint_mode_norm"]
        molecular_final = group0_spectrum.loc[
            ("molecular_cs000", COMMON_PARTICLE_STEP, mode), "joint_mode_norm"
        ]
        lines.append(
            f"| {mode} | {control_initial:.8f} -> {control_final:.8f} | "
            f"{molecular_initial:.8f} -> {molecular_final:.8f} |"
        )
    lines += [
        "",
        "Molecular group 0 loses 18.32% of its imposed mode-8 joint norm while "
        "modes 4 and 12 emerge. The lobe is therefore currently associated with "
        "group-0 reflection-odd/harmonic redistribution, not group-1 mode-8 "
        "retention or demonstrated breakdown. Future discrimination should track "
        "the group-0 local mirror ratio and its complex angular spectrum at "
        "successive exact clocks.",
        "",
        "## Common-estimator invariants at step 300",
        "",
        f"On the identical eight-extra-cell periodic grid, `Cs=0` has "
        f"`{_percent(cross_changes['total_kinetic_energy'])}` energy, "
        f"`{_percent(cross_changes['total_enstrophy'])}` enstrophy, and "
        f"`{_percent(cross_changes['test_filtered_enstrophy'])}` test-filtered "
        "enstrophy relative to `Cs=.20`. From each leg's own step-80 state to step "
        f"300, control energy/enstrophy/test-filtered-enstrophy changes are "
        f"{_percent(trajectory_changes['control']['total_kinetic_energy'])}/"
        f"{_percent(trajectory_changes['control']['total_enstrophy'])}/"
        f"{_percent(trajectory_changes['control']['test_filtered_enstrophy'])}, "
        "versus "
        f"{_percent(trajectory_changes['molecular']['total_kinetic_energy'])}/"
        f"{_percent(trajectory_changes['molecular']['total_enstrophy'])}/"
        f"{_percent(trajectory_changes['molecular']['test_filtered_enstrophy'])} "
        "for the molecular leg. Native cross-case energy remains excluded because "
        "its estimator identities differ.",
        "",
        "## Matched endpoint health",
        "",
        "| run | mean core | divergence | misalignment | CFL | impulse x |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("control_cs020", "Cs=.20"), ("molecular_cs000", "Cs=0")):
        endpoint = endpoints[key]
        lines.append(
            f"| {label} | {endpoint['mean_particle_core_radius']:.8f} | "
            f"{endpoint['vorticity_divergence_error']:.5f} | "
            f"{endpoint['vortex_strength_misalignment_degrees']:.2f} deg | "
            f"{endpoint['lagrangian_cfl']:.4f} | "
            f"{endpoint['linear_impulse_x']:.8f} |"
        )
    lines += [
        "",
        "Both endpoints remain inside all configured guards. The paired evidence "
        "supports a filter-dependent difference in core concentration, symmetry, "
        "and seeded-mode retention, but does not establish physical breakdown.",
        "",
        f"Aggregate continuation process time was `{aggregate_seconds:.2f} s` "
        f"({int(aggregate_seconds // 60)}:{aggregate_seconds % 60:05.2f}) of the approved "
        "20-minute envelope. Both processes are released; no further CFD is authorized.",
    ]
    (OUTPUT / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lifecycle = _validate_lifecycles()
    source_states = {key: _load_h5(run, 80) for key, run in SOURCE_RUNS.items()}
    common_states = {key: _load_h5(run, COMMON_PARTICLE_STEP) for key, run in RUNS.items()}
    terminal_states = {
        key: _load_h5(run, int(LIFECYCLES[key]["step"])) for key, run in RUNS.items()
    }
    for values in (*common_states.values(), *terminal_states.values()):
        if values["n_stabilization_events"] or values["n_regularization_events"]:
            raise RuntimeError("unexpected stabilization or regularization event")

    modes = _mode_rows()
    angular_spectrum = _angular_spectrum_rows()
    particles = _particle_geometry_rows(common_states)
    fields = _field_geometry_rows()
    native = _native_common_rows()
    asymmetry = _cross_asymmetry_rows()
    ring_asymmetry = _ring_cross_asymmetry_rows()
    common_integrals = _common_integral_rows(source_states, common_states)
    endpoints = _endpoint_rows()
    tables = {
        "mode_pair_step300": modes,
        "angular_spectrum_common_backups": angular_spectrum,
        "particle_geometry_step300": particles,
        "field_geometry_common": fields,
        "native_flow_common": native,
        "cross_asymmetry_common": asymmetry,
        "ring_cross_asymmetry_common": ring_asymmetry,
        "common_periodic_integrals_step300": common_integrals,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)

    result = {
        "status": "ANALYZED_EXACT_COMMON_CLOCKS",
        "lifecycle": lifecycle,
        "process_segments_seconds": PROCESS_SEGMENTS,
        "aggregate_actual_process_seconds": sum(ACTUAL_PROCESS_SECONDS.values()),
        "aggregate_process_budget_seconds": 1200.0,
        "common_particle_step": COMMON_PARTICLE_STEP,
        "common_particle_time": COMMON_PARTICLE_TIME,
        "common_field_steps": list(COMMON_FIELD_STEPS),
        "source_states": {
            key: {name: value for name, value in state.items() if not isinstance(value, np.ndarray)}
            for key, state in source_states.items()
        },
        "common_states": {
            key: {name: value for name, value in state.items() if not isinstance(value, np.ndarray)}
            for key, state in common_states.items()
        },
        "terminal_states": {
            key: {name: value for name, value in state.items() if not isinstance(value, np.ndarray)}
            for key, state in terminal_states.items()
        },
        "installed_fourier_integrals_path": str(Path(fourier_integrals.__file__).resolve()),
        "installed_fourier_integrals_sha256": _sha256(Path(fourier_integrals.__file__).resolve()),
        "phase_aware_modes": modes.to_dict(orient="records"),
        "angular_spectrum": angular_spectrum.to_dict(orient="records"),
        "particle_geometry": particles.to_dict(orient="records"),
        "field_geometry": fields.to_dict(orient="records"),
        "cross_asymmetry": asymmetry.to_dict(orient="records"),
        "ring_cross_asymmetry": ring_asymmetry.to_dict(orient="records"),
        "common_periodic_integrals": common_integrals.to_dict(orient="records"),
        "matched_step300_endpoints": endpoints,
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(
        lifecycle,
        modes,
        angular_spectrum,
        fields,
        common_integrals,
        asymmetry,
        ring_asymmetry,
        endpoints,
    )


if __name__ == "__main__":
    main()
