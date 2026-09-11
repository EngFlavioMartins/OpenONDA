#!/usr/bin/env python3
"""Compare the exact step-300/320 filter-screening states offline."""

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

from .assess_breakdown import _centered_mode_estimate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "cs_filter_pair_step320"
RUNS = {
    "control_cs020": {
        300: "cs_breakdown_filter_cs020_cpu_t6_step300_tail",
        320: "cs_breakdown_filter_cs020_cpu_t6_step320_discriminator",
    },
    "molecular_cs000": {
        300: "cs_breakdown_filter_cs000_cpu_t6_step300_continuation",
        320: "cs_breakdown_filter_cs000_cpu_t6_step320_discriminator",
    },
}
EXPECTED_H5_SHA256 = {
    ("control_cs020", 300): ("51ac824ce35a00a3b48b9ff3c8632e8ab6d20026c7ad009bdb061a637f8dfd10"),
    ("molecular_cs000", 300): ("ac8e1690c8b85d18af2eb7da59f1a5431fdbb72415bbc4679b5c893b9c7c2201"),
    ("control_cs020", 320): ("6362843dc57cca9640a6aeca86a48a9e03effed4e79136dccf824333e50e0568"),
    ("molecular_cs000", 320): ("c0ac74fbe17fc1289240c387d7a1ed3669b791d181801626ef9b81f2ff492355"),
}
EXPECTED_CONFIG_SHA256 = {
    "control_cs020": ("4b4d03f21a1bb308b25f158385f682dc086a2010fcd5f1e8e172eb24d97753e2"),
    "molecular_cs000": ("0e482b559ea93ca030209ac766fe9507a979996331a1af46d664d7e157bcdcbd"),
}
ACTUAL_PROCESS_SECONDS = {"control_cs020": 82.93, "molecular_cs000": 27.53}
NEW_PROCESS_BUDGET_SECONDS = 300.0
PRIOR_FILTER_PROCESS_SECONDS = 1306.93
EXPECTED_INSTALLED_VPM_SHA256 = "ab4be73c378d83ce6ba169f0253b3c13a6288c22171872fa198dd42c810d956e"
CORE_CUTOFF = 0.6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(key: str, step: int) -> tuple[Path, Path]:
    run = RUNS[key][step]
    return ROOT / "solution" / run, ROOT / "samples" / run


def _load_state(key: str, step: int) -> dict[str, object]:
    solution, _ = _paths(key, step)
    path = solution / f"vpm_{step:06d}.h5"
    observed_hash = _sha256(path)
    if observed_hash != EXPECTED_H5_SHA256[(key, step)]:
        raise RuntimeError(f"HDF5 identity changed for {key} step {step}")
    with h5py.File(path, "r") as handle:
        particles = handle["particles"]
        state: dict[str, object] = {
            "position": np.asarray(particles["position"], dtype=np.float64),
            "vortex_strength": np.asarray(particles["vortex_strength"], dtype=np.float64),
            "group_id": np.asarray(particles["group_id"], dtype=np.int32),
            "step": int(handle["solver"].attrs["step"]),
            "time": float(handle["solver"].attrs["time"]),
            "n_particles_total": int(handle["solver"].attrs["n_particles_total"]),
            "n_stabilization_events": int(handle["solver"].attrs["n_stabilization_events"]),
            "n_regularization_events": int(handle["solver"].attrs["n_regularization_events"]),
            "numerical_configuration_sha256": str(
                handle["solver"].attrs["numerical_configuration_sha256"]
            ),
            "path": str(path),
            "sha256": observed_hash,
        }
    expected_time = step * 0.0075
    observed = (
        state["step"],
        state["time"],
        state["n_particles_total"],
        state["n_stabilization_events"],
        state["n_regularization_events"],
    )
    if observed != (step, expected_time, 16_104, 0, 0):
        raise RuntimeError(f"unexpected native state for {key} step {step}: {observed}")
    if state["numerical_configuration_sha256"] != EXPECTED_CONFIG_SHA256[key]:
        raise RuntimeError(f"configuration identity changed for {key}")
    for name in ("position", "vortex_strength", "group_id"):
        if not np.isfinite(state[name]).all():
            raise RuntimeError(f"non-finite {name} for {key} step {step}")
    return state


def _mode_rows(states: dict[tuple[str, int], dict[str, object]]) -> pd.DataFrame:
    rows = []
    for (key, step), state in states.items():
        groups = state["group_id"]
        for group, modes in ((0, (4, 8, 12)), (1, (8,))):
            selected = groups == group
            for mode in modes:
                estimate = _centered_mode_estimate(
                    state["position"][selected],
                    state["vortex_strength"][selected],
                    mode=mode,
                )
                rows.append(
                    {
                        "run_key": key,
                        "step": step,
                        "time": state["time"],
                        "group_id": group,
                        "mode": mode,
                        "axial_real": estimate["axial_mode8_real"],
                        "axial_imag": estimate["axial_mode8_imag"],
                        "axial_amplitude": estimate["axial_mode8_amplitude"],
                        "axial_phase": estimate["axial_mode8_phase"],
                        "radial_real": estimate["radial_mode8_real"],
                        "radial_imag": estimate["radial_mode8_imag"],
                        "radial_amplitude": estimate["radial_mode8_amplitude"],
                        "radial_phase": estimate["radial_mode8_phase"],
                        "joint_mode_norm": np.hypot(
                            estimate["axial_mode8_amplitude"],
                            estimate["radial_mode8_amplitude"],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _mirror_asymmetry(magnitude: np.ndarray) -> float:
    even = 0.5 * (magnitude + magnitude[:, ::-1])
    odd = 0.5 * (magnitude - magnitude[:, ::-1])
    denominator = max(float(np.linalg.norm(even.ravel())), np.finfo(float).tiny)
    return float(np.linalg.norm(odd.ravel()) / denominator)


def _field_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    geometry_rows = []
    asymmetry_rows = []
    for key in RUNS:
        for step in (300, 320):
            _, samples = _paths(key, step)
            diagnostics = pd.read_csv(samples / "ring_diagnostics.csv")
            rings = diagnostics[diagnostics.step == step].set_index("group_id")
            centers = rings[["vortex_centroid_x", "major_radius"]].to_numpy()
            for plane in ("core_section", "cross_section"):
                grid = pv.read(samples / f"{plane}_{step:06d}.vts")
                points = np.asarray(grid.points, dtype=np.float64)
                vorticity = np.asarray(grid.point_data["vorticity"], dtype=np.float64)
                coordinates = (
                    points[:, :2]
                    if plane == "core_section"
                    else np.column_stack((points[:, 0], np.abs(points[:, 2])))
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
                    distance_sq = np.sum((local_coordinates - centers[group]) ** 2, axis=1)
                    weight = np.linalg.norm(local_vorticity, axis=1)
                    weight *= distance_sq <= CORE_CUTOFF**2
                    centroid = np.einsum("i,ij->j", weight, local_coordinates) / weight.sum()
                    width = np.sqrt(
                        np.dot(
                            weight,
                            np.sum((local_coordinates - centroid) ** 2, axis=1),
                        )
                        / weight.sum()
                    )
                    geometry_rows.append(
                        {
                            "run_key": key,
                            "step": step,
                            "time": float(rings.loc[group, "time"]),
                            "plane": plane,
                            "group_id": group,
                            "field_centroid_x": centroid[0],
                            "field_major_radius": centroid[1],
                            "field_core_rms_width": width,
                            "field_peak_vorticity": np.linalg.norm(local_vorticity, axis=1).max(),
                        }
                    )
                if plane == "cross_section":
                    dimensions = grid.dimensions
                    magnitude = np.linalg.norm(
                        vorticity.reshape((dimensions[1], dimensions[0], 3)).transpose(1, 0, 2),
                        axis=2,
                    )
                    axial_coordinates = points[:, 0].reshape((dimensions[1], dimensions[0])).T[:, 0]
                    axial_assignment = np.argmin(
                        np.abs(axial_coordinates[:, None] - centers[None, :, 0]),
                        axis=1,
                    )
                    for group in (0, 1):
                        asymmetry_rows.append(
                            {
                                "run_key": key,
                                "step": step,
                                "time": float(rings.loc[group, "time"]),
                                "group_id": group,
                                "vortex_centroid_x": centers[group, 0],
                                "local_mirror_asymmetry": _mirror_asymmetry(
                                    magnitude[axial_assignment == group]
                                ),
                                "global_mirror_asymmetry": _mirror_asymmetry(magnitude),
                            }
                        )
    return pd.DataFrame(geometry_rows), pd.DataFrame(asymmetry_rows)


def _health_rows() -> pd.DataFrame:
    rows = []
    for key in RUNS:
        for step in (300, 320):
            _, samples = _paths(key, step)
            flow = pd.read_csv(samples / "flow_integrals.csv")
            row = flow[flow.step == step].iloc[0]
            rows.append(
                {
                    "run_key": key,
                    "step": step,
                    "time": row.time,
                    "energy_measurement": (
                        row.energy_measurement
                        if "energy_measurement" in flow.columns
                        else "not_saved_historically"
                    ),
                    "kinetic_energy_rate_source": row.kinetic_energy_rate_source,
                    "vorticity_divergence_error": row.vorticity_divergence_error,
                    "vortex_strength_misalignment_degrees": (
                        row.vortex_strength_misalignment_degrees
                    ),
                    "lagrangian_cfl": row.lagrangian_cfl,
                    "mean_particle_core_radius": row.mean_particle_core_radius,
                    "n_particles_total": int(row.n_particles_total),
                    "n_stabilization_events": int(row.n_stabilization_events),
                    "n_regularization_events": int(row.n_regularization_events),
                }
            )
    return pd.DataFrame(rows)


def _relative(final: float, initial: float) -> float:
    return final / initial - 1.0


def _write_markdown(
    modes: pd.DataFrame,
    fields: pd.DataFrame,
    asymmetry: pd.DataFrame,
    health: pd.DataFrame,
) -> None:
    mode_index = modes.set_index(["run_key", "step", "group_id", "mode"])
    asym_index = asymmetry.set_index(["run_key", "step", "group_id"])
    health_index = health.set_index(["run_key", "step"])

    def phase_change(key: str, group: int, mode: int, component: str) -> float:
        initial = mode_index.loc[(key, 300, group, mode), f"{component}_phase"]
        final = mode_index.loc[(key, 320, group, mode), f"{component}_phase"]
        return float(np.degrees(np.angle(np.exp(1j * (final - initial)))))

    lines = [
        "# Matched step-320 filter discriminator",
        "",
        "Both strict native restarts completed step 320/time 2.4 with 16,104 "
        "particles and no health, stabilization or regularization stop. The control "
        f"used `{ACTUAL_PROCESS_SECONDS['control_cs020']:.2f} s` and the molecular "
        f"leg `{ACTUAL_PROCESS_SECONDS['molecular_cs000']:.2f} s`; aggregate actual "
        f"process time was `{sum(ACTUAL_PROCESS_SECONDS.values()):.2f} s` of the "
        f"`{NEW_PROCESS_BUDGET_SECONDS:.0f} s` grant.",
        "",
        "## Identity-resolved angular screen",
        "",
        "| group/mode | control 300 -> 320 | change | molecular 300 -> 320 | change |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, mode in ((0, 4), (0, 8), (0, 12), (1, 8)):
        values = {}
        for key in RUNS:
            initial = mode_index.loc[(key, 300, group, mode), "joint_mode_norm"]
            final = mode_index.loc[(key, 320, group, mode), "joint_mode_norm"]
            values[key] = (initial, final, _relative(final, initial))
        lines.append(
            f"| g{group}/m{mode} | {values['control_cs020'][0]:.8f} -> "
            f"{values['control_cs020'][1]:.8f} | "
            f"{100 * values['control_cs020'][2]:+.2f}% | "
            f"{values['molecular_cs000'][0]:.8f} -> "
            f"{values['molecular_cs000'][1]:.8f} | "
            f"{100 * values['molecular_cs000'][2]:+.2f}% |"
        )
    molecular_group0_asymmetry_change = _relative(
        asym_index.loc[("molecular_cs000", 320, 0), "local_mirror_asymmetry"],
        asym_index.loc[("molecular_cs000", 300, 0), "local_mirror_asymmetry"],
    )
    group1_final_relative = _relative(
        mode_index.loc[("molecular_cs000", 320, 1, 8), "joint_mode_norm"],
        mode_index.loc[("control_cs020", 320, 1, 8), "joint_mode_norm"],
    )
    lines += [
        "",
        "The right-hand molecular group-0 deformation persists over the short "
        f"screen: its local cross-plane mirror asymmetry changes "
        f"`{asym_index.loc[('molecular_cs000', 300, 0), 'local_mirror_asymmetry']:.8f} "
        f"-> {asym_index.loc[('molecular_cs000', 320, 0), 'local_mirror_asymmetry']:.8f}` "
        f"(`{100 * molecular_group0_asymmetry_change:+.2f}%`). Its mode 8 rebounds, "
        "mode 4 increases slightly and mode 12 decreases, so the angular result is "
        "continued component/harmonic exchange rather than uniform modal growth. "
        "Molecular group-0 wrapped axial/radial phase changes are "
        f"`{phase_change('molecular_cs000', 0, 4, 'axial'):+.2f}/"
        f"{phase_change('molecular_cs000', 0, 4, 'radial'):+.2f} deg` for mode 4, "
        f"`{phase_change('molecular_cs000', 0, 8, 'axial'):+.2f}/"
        f"{phase_change('molecular_cs000', 0, 8, 'radial'):+.2f} deg` for mode 8 and "
        f"`{phase_change('molecular_cs000', 0, 12, 'axial'):+.2f}/"
        f"{phase_change('molecular_cs000', 0, 12, 'radial'):+.2f} deg` for mode 12.",
        "",
        "Group 1 remains separate. Its molecular mode-8 norm falls over the interval "
        "while the control rises; at step 320 the molecular value is "
        f"`{100 * group1_final_relative:+.2f}%` relative to the same-time control. "
        "Its molecular axial/radial phases rotate "
        f"`{phase_change('molecular_cs000', 1, 8, 'axial'):+.2f}/"
        f"{phase_change('molecular_cs000', 1, 8, 'radial'):+.2f} deg`. The earlier "
        "+52.84% retention advantage therefore does not persist.",
        "",
        "## Two-plane morphology",
        "",
        "| plane/group | control width 300 -> 320 | molecular width 300 -> 320 | molecular/control at 320 |",
        "|---|---:|---:|---:|",
    ]
    field_index = fields.set_index(["run_key", "step", "plane", "group_id"])
    for plane in ("core_section", "cross_section"):
        for group in (0, 1):
            control_initial = field_index.loc[
                ("control_cs020", 300, plane, group), "field_core_rms_width"
            ]
            control_final = field_index.loc[
                ("control_cs020", 320, plane, group), "field_core_rms_width"
            ]
            molecular_initial = field_index.loc[
                ("molecular_cs000", 300, plane, group), "field_core_rms_width"
            ]
            molecular_final = field_index.loc[
                ("molecular_cs000", 320, plane, group), "field_core_rms_width"
            ]
            lines.append(
                f"| {plane}/g{group} | {control_initial:.8f} -> "
                f"{control_final:.8f} | {molecular_initial:.8f} -> "
                f"{molecular_final:.8f} | "
                f"{100 * _relative(molecular_final, control_final):+.2f}% |"
            )
    lines += [
        "",
        "Both native planes retain identifiable main cores. The group-0 reflection-"
        "odd deformation persists, but a discrete secondary lobe is not separately "
        "resolved on both final planes. One 0.15-second interval is only a screening "
        "observation; it does not establish sustained physical instability or breakdown.",
        "",
        "## Health and decision",
        "",
        "| run | divergence | misalignment | CFL |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("control_cs020", "Cs=.20"), ("molecular_cs000", "Cs=0")):
        endpoint = health_index.loc[(key, 320)]
        lines.append(
            f"| {label} | {endpoint.vorticity_divergence_error:.5f} | "
            f"{endpoint.vortex_strength_misalignment_degrees:.2f} deg | "
            f"{endpoint.lagrangian_cfl:.4f} |"
        )
    lines += [
        "",
        "Both endpoints remain inside the unchanged limits, but the molecular "
        "divergence and misalignment margins are narrow. The screen supports "
        "persistence of a group-0 filter-dependent deformation, while rejecting a "
        "robust group-1 retention result. It still does not show sustained loss of "
        "coherent ring structure, spatial convergence, or post-breakdown survival. "
        "No additional horizon is warranted automatically.",
        "",
        f"Cumulative actual process time for all filter experiments is "
        f"`{PRIOR_FILTER_PROCESS_SECONDS + sum(ACTUAL_PROCESS_SECONDS.values()):.2f} s`.",
    ]
    (OUTPUT / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    states = {(key, step): _load_state(key, step) for key in RUNS for step in (300, 320)}
    modes = _mode_rows(states)
    fields, asymmetry = _field_rows()
    health = _health_rows()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "identity_resolved_modes": modes,
        "two_plane_geometry": fields,
        "ring_cross_asymmetry": asymmetry,
        "endpoint_health": health,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)
    result = {
        "status": "ANALYZED_EXACT_STEP300_STEP320",
        "run_names": RUNS,
        "states": {
            f"{key}_step{step}": {
                name: value for name, value in state.items() if not isinstance(value, np.ndarray)
            }
            for (key, step), state in states.items()
        },
        "runtime_aggregate_sha256": EXPECTED_INSTALLED_VPM_SHA256,
        "configuration_sha256": EXPECTED_CONFIG_SHA256,
        "actual_process_seconds": ACTUAL_PROCESS_SECONDS,
        "actual_process_budget_seconds": NEW_PROCESS_BUDGET_SECONDS,
        "cumulative_filter_process_seconds": (
            PRIOR_FILTER_PROCESS_SECONDS + sum(ACTUAL_PROCESS_SECONDS.values())
        ),
        "modes": modes.to_dict(orient="records"),
        "two_plane_geometry": fields.to_dict(orient="records"),
        "ring_cross_asymmetry": asymmetry.to_dict(orient="records"),
        "endpoint_health": health.to_dict(orient="records"),
        "claim_boundary": (
            "short-interval persistence is a screening result, not proof of "
            "physical instability or breakdown"
        ),
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(modes, fields, asymmetry, health)


if __name__ == "__main__":
    main()
