#!/usr/bin/env python3
"""Compare the completed seeded ``Cs=.20``/``Cs=0`` qualification pair.

The native histories use different kinetic-energy estimators when one state
has variable core/viscosity and the other is uniform.  This read-only audit
therefore also evaluates the common initial state and both final states on
identical periodic Fourier grids.  Native energy values are retained with
their estimator labels but are never differenced across the pair.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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

from .check_filter_pair_preflight import _build_case, _cloud  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "cs_filter_pair"
RUNS = {
    "control_cs020": "cs_breakdown_filter_cs020_cpu_t6_step080",
    "molecular_cs000": "cs_breakdown_filter_cs000_cpu_t6_step080",
}
RUN_LABELS = {
    "control_cs020": r"$C_s=.20$",
    "molecular_cs000": r"$C_s=0$",
}
FINAL_STEP = 80
FINAL_TIME = 0.6
SEED_AMPLITUDE = 0.05
FIELD_CORE_CUTOFF = 0.6
GRID_EXTRA_CELLS_PER_SIDE = (4, 8, 12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_particles(run: str) -> dict[str, np.ndarray | float | int]:
    path = ROOT / "solution" / run / f"vpm_{FINAL_STEP:06d}.h5"
    with h5py.File(path, "r") as file:
        particles = file["particles"]
        result: dict[str, np.ndarray | float | int] = {
            name: np.asarray(particles[name], dtype=np.float64)
            for name in (
                "position",
                "vortex_strength",
                "core_radius",
                "particle_volume",
                "effective_viscosity",
            )
        }
        result["group_id"] = np.asarray(particles["group_id"], dtype=np.int32)
        result["time"] = float(file["solver"].attrs["time"])
        result["step"] = int(file["solver"].attrs["step"])
    if result["step"] != FINAL_STEP or result["time"] != FINAL_TIME:
        raise RuntimeError(f"unexpected endpoint for {run}: {result['step']}, {result['time']}")
    if not all(
        np.isfinite(value).all() for value in result.values() if isinstance(value, np.ndarray)
    ):
        raise RuntimeError(f"non-finite final particle array in {run}")
    result["h5_sha256"] = _sha256(path)
    return result


def _native_rows() -> pd.DataFrame:
    rows = []
    for key, run in RUNS.items():
        flow = pd.read_csv(ROOT / "samples" / run / "flow_integrals.csv")
        if not np.array_equal(flow.step.to_numpy(), np.arange(0, FINAL_STEP + 1, 10)):
            raise RuntimeError(f"unexpected native FlowIntegrals cadence for {run}")
        for endpoint, row in (("initial", flow.iloc[0]), ("final", flow.iloc[-1])):
            rows.append(
                {
                    "run_key": key,
                    "run": run,
                    "run_label": RUN_LABELS[key],
                    "endpoint": endpoint,
                    **{
                        name: row[name]
                        for name in (
                            "time",
                            "step",
                            "total_kinetic_energy",
                            "total_enstrophy",
                            "test_filtered_enstrophy",
                            "kinetic_energy_rate_source",
                            "total_helicity",
                            "vortex_strength_magnitude_sum",
                            "linear_impulse_x",
                            "mean_particle_core_radius",
                            "mean_eddy_viscosity",
                            "max_eddy_viscosity",
                            "vorticity_divergence_error",
                            "vortex_strength_misalignment_degrees",
                            "lagrangian_cfl",
                            "n_particles_total",
                            "n_stabilization_events",
                            "n_regularization_events",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def _mode_rows() -> pd.DataFrame:
    raw: dict[tuple[str, int], tuple[complex, complex]] = {}
    for key, run in RUNS.items():
        metrics = pd.read_csv(ROOT / "figures" / run / "breakdown_metrics.csv")
        metrics = metrics[(metrics.step == FINAL_STEP) & (metrics.source == "particle_backup")]
        if set(metrics.group_id.astype(int)) != {0, 1}:
            raise RuntimeError(f"missing endpoint particle modes for {run}")
        for row in metrics.itertuples(index=False):
            raw[(key, int(row.group_id))] = (
                complex(row.axial_mode8_real, row.axial_mode8_imag),
                complex(row.radial_mode8_real, row.radial_mode8_imag),
            )

    rows = []
    for group in (0, 1):
        control = raw[("control_cs020", group)]
        molecular = raw[("molecular_cs000", group)]
        control_norm = 2.0 * np.sqrt(abs(control[0]) ** 2 + abs(control[1]) ** 2)
        molecular_norm = 2.0 * np.sqrt(abs(molecular[0]) ** 2 + abs(molecular[1]) ** 2)
        separation = (
            2.0
            * np.sqrt(abs(molecular[0] - control[0]) ** 2 + abs(molecular[1] - control[1]) ** 2)
            / SEED_AMPLITUDE
        )
        axial_phase_change = np.angle(np.exp(1j * (np.angle(molecular[0]) - np.angle(control[0]))))
        radial_phase_change = np.angle(np.exp(1j * (np.angle(molecular[1]) - np.angle(control[1]))))
        rows.append(
            {
                "group_id": group,
                "control_axial_real": control[0].real,
                "control_axial_imag": control[0].imag,
                "control_radial_real": control[1].real,
                "control_radial_imag": control[1].imag,
                "molecular_axial_real": molecular[0].real,
                "molecular_axial_imag": molecular[0].imag,
                "molecular_radial_real": molecular[1].real,
                "molecular_radial_imag": molecular[1].imag,
                "control_axial_amplitude": 2.0 * abs(control[0]),
                "molecular_axial_amplitude": 2.0 * abs(molecular[0]),
                "axial_amplitude_relative_change": abs(molecular[0]) / abs(control[0]) - 1.0,
                "control_axial_phase_radians": np.angle(control[0]),
                "molecular_axial_phase_radians": np.angle(molecular[0]),
                "axial_phase_change_degrees": np.degrees(axial_phase_change),
                "control_radial_amplitude": 2.0 * abs(control[1]),
                "molecular_radial_amplitude": 2.0 * abs(molecular[1]),
                "radial_amplitude_relative_change": abs(molecular[1]) / abs(control[1]) - 1.0,
                "control_radial_phase_radians": np.angle(control[1]),
                "molecular_radial_phase_radians": np.angle(molecular[1]),
                "radial_phase_change_degrees": np.degrees(radial_phase_change),
                "control_joint_mode_norm": control_norm,
                "molecular_joint_mode_norm": molecular_norm,
                "joint_mode_norm_relative_change": molecular_norm / control_norm - 1.0,
                "phase_aware_separation_over_seed": separation,
            }
        )
    return pd.DataFrame(rows)


def _particle_geometry_rows(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for key, values in data.items():
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
                    "run": RUNS[key],
                    "run_label": RUN_LABELS[key],
                    "group_id": group,
                    "weighted_centroid_x": centroid[0],
                    "weighted_centroid_y": centroid[1],
                    "weighted_centroid_z": centroid[2],
                    "weighted_mean_radius": np.dot(weight, np.hypot(position[:, 1], position[:, 2]))
                    / denominator,
                    "weighted_major_radius": np.sqrt(max(eigenvalues[-1] + eigenvalues[-2], 0.0)),
                    "vortex_strength_magnitude_sum": denominator,
                    "linear_impulse_x": impulse[0],
                    "linear_impulse_norm": np.linalg.norm(impulse),
                    "impulse_radius": 2.0 * np.linalg.norm(impulse) / denominator,
                }
            )
    return pd.DataFrame(rows)


def _field_geometry_rows() -> pd.DataFrame:
    rows = []
    for plane in ("core_section", "cross_section"):
        for key, run in RUNS.items():
            path = ROOT / "samples" / run / f"{plane}_{FINAL_STEP:06d}.vts"
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
                .query("step == @FINAL_STEP")
                .sort_values("group_id")
            )
            centers = ring_rows[["vortex_centroid_x", "major_radius"]].to_numpy(dtype=np.float64)
            distance_sq = np.sum((coordinates[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            assignment = np.argmin(distance_sq, axis=1)
            for group in (0, 1):
                selected = assignment == group
                local_coordinates = coordinates[selected]
                local_vorticity = vorticity[selected]
                local_distance_sq = np.sum((local_coordinates - centers[group]) ** 2, axis=1)
                weight = np.linalg.norm(local_vorticity, axis=1)
                weight *= local_distance_sq <= FIELD_CORE_CUTOFF**2
                denominator = float(weight.sum())
                centroid = np.einsum("i,ij->j", weight, local_coordinates) / denominator
                width = np.sqrt(
                    np.dot(
                        weight,
                        np.sum((local_coordinates - centroid) ** 2, axis=1),
                    )
                    / denominator
                )
                rows.append(
                    {
                        "run_key": key,
                        "run": run,
                        "run_label": RUN_LABELS[key],
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


def _common_integral_rows(
    data: dict[str, dict[str, Any]], initial: dict[str, np.ndarray]
) -> pd.DataFrame:
    spacing = float(np.median(np.cbrt(data["control_cs020"]["particle_volume"])))
    all_positions = np.vstack(
        [initial["position"], *(values["position"] for values in data.values())]
    )
    base = _grid_for_particles(all_positions, spacing)
    states = {"common_initial": initial, **data}
    rows = []
    for extra in GRID_EXTRA_CELLS_PER_SIDE:
        grid = CartesianGrid(
            origin=base.origin - extra * spacing,
            spacing=spacing,
            shape=tuple(size + 2 * extra for size in base.shape),
        )
        for key, values in states.items():
            result = gaussian_fourier_integrals(
                values["position"],
                values["vortex_strength"],
                values["core_radius"],
                values["particle_volume"],
                effective_viscosity=None,
                grid=grid,
            )
            rows.append(
                {
                    "state": key,
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
                    / max(abs(result.total_kinetic_energy), np.finfo(float).tiny),
                    "enstrophy_expansion_relative_change": abs(
                        result.total_enstrophy - result.previous_order_total_enstrophy
                    )
                    / max(abs(result.total_enstrophy), np.finfo(float).tiny),
                }
            )
    return pd.DataFrame(rows)


def _percent(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def _write_markdown(
    native: pd.DataFrame,
    modes: pd.DataFrame,
    particle: pd.DataFrame,
    field: pd.DataFrame,
    common: pd.DataFrame,
    hashes: dict[str, str],
) -> None:
    final_native = native[native.endpoint == "final"].set_index("run_key")
    common8 = common[common.grid_extra_cells_per_side == 8].set_index("state")
    initial_energy = common8.loc["common_initial", "total_kinetic_energy"]
    initial_enstrophy = common8.loc["common_initial", "total_enstrophy"]
    energy_changes = {
        key: common8.loc[key, "total_kinetic_energy"] / initial_energy - 1.0 for key in RUNS
    }
    enstrophy_changes = {
        key: common8.loc[key, "total_enstrophy"] / initial_enstrophy - 1.0 for key in RUNS
    }
    impulse_difference = (
        final_native.loc["molecular_cs000", "linear_impulse_x"]
        / final_native.loc["control_cs020", "linear_impulse_x"]
        - 1.0
    )
    lines = [
        "# Completed seeded filter-pair comparison",
        "",
        "Both runs reached step 80 (`t=.6 s`) with 16,104 finite particles and no "
        "stabilization or regularization events. Initial primary arrays and mutual "
        "fields were exact in the paired preflight.",
        "",
        "## Phase-aware mode result",
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
        "Here `Dg = 2 sqrt(|c_ax,Cs0-c_ax,Cs.20|^2 + "
        "|c_rad,Cs0-c_rad,Cs.20|^2) / .05`; the complex coefficients are "
        "equal-azimuth-bin fits and `.05 R0` is the imposed seed. The `.05` "
        "distance rubric was documented before execution but is exploratory, "
        "not an independently calibrated or accepted breakdown gate. Both values "
        "exceed it, while neither joint norm meets the reviewed 20% strong-signal "
        "marker. This is an early filter response, not growth or breakdown.",
        "",
        "| group | component | amplitude Cs=.20 | amplitude Cs=0 | amplitude change | wrapped phase change |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in modes.itertuples(index=False):
        for component in ("axial", "radial"):
            lines.append(
                f"| {row.group_id} | {component} | "
                f"{getattr(row, f'control_{component}_amplitude'):.8f} | "
                f"{getattr(row, f'molecular_{component}_amplitude'):.8f} | "
                f"{_percent(getattr(row, f'{component}_amplitude_relative_change'))} | "
                f"{getattr(row, f'{component}_phase_change_degrees'):+.2f} deg |"
            )
    lines += [
        "",
        "The group-1 radial phase is not independently interpretable: its control "
        "amplitude is only `.000716 R0`, so the apparent 174-degree change occurs "
        "near a component zero. `Dg` keeps that component in its complex plane and "
        "the joint norm prevents calling the phase rotation growth.",
        "",
        "## Core and geometry result",
        "",
        "| plane | group | width Cs=.20 | width Cs=0 | Cs=0 narrowing | peak ratio Cs=0/Cs=.20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for plane in ("core_section", "cross_section"):
        for group in (0, 1):
            subset = field[(field.plane == plane) & (field.group_id == group)].set_index("run_key")
            w20 = subset.loc["control_cs020", "field_core_rms_width"]
            w0 = subset.loc["molecular_cs000", "field_core_rms_width"]
            peak_ratio = (
                subset.loc["molecular_cs000", "field_peak_vorticity"]
                / subset.loc["control_cs020", "field_peak_vorticity"]
            )
            lines.append(
                f"| {plane} | {group} | {w20:.8f} | {w0:.8f} | "
                f"{100.0 * (1.0 - w0 / w20):.2f}% | {peak_ratio:.4f} |"
            )
    lines += [
        "",
        "Weighted axial centroids advance by `+.01195/+.01132 R0` for groups "
        "0/1 when `Cs` is removed (computed in `particle_geometry.csv`). Global "
        f"axial impulse differs by only `{abs(100.0 * impulse_difference):.5f}%` "
        "at the endpoint; groupwise "
        "differences largely cancel.",
        "",
        "## Energy-estimator identity",
        "",
        "Native energy is not directly comparable across this pair: the control "
        f"ends with `{final_native.loc['control_cs020', 'kinetic_energy_rate_source']}`, "
        "while the molecular run uses "
        f"`{final_native.loc['molecular_cs000', 'kinetic_energy_rate_source']}`. "
        "The native CSV omits the explicit `energy_measurement` label.",
        "",
        "On one identical periodic Fourier grid (eight extra cells per side), the "
        f"control energy/enstrophy changes are `{_percent(energy_changes['control_cs020'])}`/"
        f"`{_percent(enstrophy_changes['control_cs020'])}` and the molecular changes are "
        f"`{_percent(energy_changes['molecular_cs000'])}`/"
        f"`{_percent(enstrophy_changes['molecular_cs000'])}` from the common initial state. "
        "At the endpoint the molecular case retains `4.753%` more energy, `26.05%` "
        "more enstrophy and `18.10%` more test-filtered enstrophy than the control on "
        "that common grid. Changing the common envelope from four to twelve extra "
        "cells per side changes the absolute final cross-case energy difference by "
        "less than `0.001%` relative.",
        "",
        "## Immutable endpoints",
        "",
        f"- `Cs=.20`: `{hashes['control_cs020']}`",
        f"- `Cs=0`: `{hashes['molecular_cs000']}`",
        "",
        "Machine-readable values are in `comparison.json` and the adjacent CSV files.",
    ]
    OUTPUT.joinpath("comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = {key: _load_particles(run) for key, run in RUNS.items()}
    initial_cases = {
        "control_cs020": _build_case(RUNS["control_cs020"], 0.20),
        "molecular_cs000": _build_case(RUNS["molecular_cs000"], 0.0),
    }
    initial_clouds = {key: _cloud(case) for key, case in initial_cases.items()}
    for field in (
        "position",
        "vortex_strength",
        "core_radius",
        "particle_volume",
        "group_id",
    ):
        if not np.array_equal(
            initial_clouds["control_cs020"][field],
            initial_clouds["molecular_cs000"][field],
        ):
            raise RuntimeError(f"paired initial field changed since preflight: {field}")

    native = _native_rows()
    final_native = native[native.endpoint == "final"]
    if bool((final_native[["n_stabilization_events", "n_regularization_events"]] != 0).any().any()):
        raise RuntimeError("unexpected stabilization or regularization event in filter pair")
    modes = _mode_rows()
    particle = _particle_geometry_rows(data)
    field = _field_geometry_rows()
    common = _common_integral_rows(data, initial_clouds["control_cs020"])
    tables = {
        "native_endpoints": native,
        "mode_pair": modes,
        "particle_geometry": particle,
        "field_geometry": field,
        "common_periodic_integrals": common,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)

    hashes = {key: str(values["h5_sha256"]) for key, values in data.items()}
    result = {
        "status": "COMPLETED",
        "comparison_time": FINAL_TIME,
        "comparison_step": FINAL_STEP,
        "seed_amplitude": SEED_AMPLITUDE,
        "runs": RUNS,
        "offline_fourier_integrals_path": str(Path(fourier_integrals.__file__).resolve()),
        "offline_fourier_integrals_sha256": _sha256(Path(fourier_integrals.__file__).resolve()),
        "final_h5_sha256": hashes,
        "phase_aware_modes": modes.to_dict(orient="records"),
        "native_endpoints": native.to_dict(orient="records"),
        "particle_geometry": particle.to_dict(orient="records"),
        "field_geometry": field.to_dict(orient="records"),
        "common_periodic_integrals": common.to_dict(orient="records"),
        "energy_comparison_warning": (
            "native cross-case energy values have different measurement identities; "
            "use common_periodic_integrals for the matched comparison"
        ),
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(native, modes, particle, field, common, hashes)


if __name__ == "__main__":
    main()
