#!/usr/bin/env python3
"""Compare finer complete RK/GBD steps at the fixed 3D cube exchange cadence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from source.solvers.fvm.io.backup import decode_state, mesh_hash
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.compare_interface_cadence_3d import FIXED_INPUTS
from studies.coupler_accuracy.compare_inviscid_subcycling_3d import history_metrics, line_metrics
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.verify_accepted_wake_prefix_3d import direct_profile, rms
from studies.coupler_accuracy.verify_sampled_face_trials_3d import independent_cube_force


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources, differences = [], []

    def checked(row):
        assert hash_file(ROOT / row["path"]) == row
        sources.append(row)
        return ROOT / row["path"]

    def report(path):
        sources.append(hash_file(path))
        value = json.loads(path.read_text())
        assert value["status"] == "complete"
        for row in value.get("sources", []):
            checked(row)
        return value

    qualification = report(args.wrapper_verification)
    assert qualification["schema"] == "openonda-profile-observer-verification-3d/1"
    assert qualification["advancing_comparison_intervals"] == 3 and qualification["comparison_histories_bitwise_equal"]
    assert qualification["checkpoint_bitwise_equal_counts"] == {"fvm": 17, "vpm_boundary_condition": 11, "vpm_numeric_datasets": 11}
    control_schedule = report(args.control_schedule)
    assert control_schedule["schema"] == "openonda-split-schedule-verification-3d/1"
    assert control_schedule["vpm_substeps"] == 1 and control_schedule["accepted_intervals"] == 3
    control_observer_path = ROOT / control_schedule["run_directory"] / "profile-observation-3d.json"
    assert hash_file(control_observer_path) in qualification["sources"]
    schedule = report(args.schedule_verification)
    assert schedule["schema"] == "openonda-split-schedule-verification-3d/1"
    assert ROOT / schedule["run_directory"] == args.run and schedule["accepted_intervals"] == 20
    baseline = report(args.baseline_prefix)
    assert baseline["schema"] == "openonda-long-wake-prefix-verification-3d/2"
    assert baseline["through_coupling_step"] >= 20
    candidate = report(args.run / "split-subcycling-3d.json")
    assert hash_file(args.run / "split-subcycling-3d.json") in schedule["sources"]
    assert candidate["spatial_dimensions"] == 3 and candidate["requested_exchanges"] == 20
    assert candidate["outer_dt"] == .05 and candidate["fvm_dt"] == .01
    n = candidate["vpm_substeps"]
    assert n == schedule["vpm_substeps"] == 5 and candidate["substep_dt"] == .05 / n
    for row in candidate["sources"]:
        archive = args.run / "split-sources" / row["path"]
        assert hash_file(archive)["sha256"] == row["sha256"]
        sources.append(hash_file(archive))
    for row in candidate["child_reports"]:
        report(checked(row))
    iteration = report(args.run / "interface-iteration-3d.json")
    trial = report(args.run / "trial/cube-coupled-trial.json")
    profiles = report(args.run / "profile-observation-3d.json")
    checkpoints = report(args.run / "accepted-fvm-checkpoints-3d.json")
    assert iteration["maximum_sweeps"] == 12 and iteration["normal_tolerance"] == iteration["gradient_tolerance"] == 1e-6
    assert iteration["relaxation"] == 1 and iteration["execution_environment"]["TI_CPU_MAX_NUM_THREADS"] == "1"
    assert trial["small_cells"] == 16936 and trial["full_cells"] == 53752
    assert trial["identical_native_shared_cells"] and trial["particle_spacing"] == trial["requested_wall_spacing"] == .0625
    assert trial["fvm_numerics"]["sgs"] == "none" and trial["fvm_numerics"]["kinematic_viscosity"] == .001
    assert trial["source_seed_time"] == .5 and trial["fvm_dt"] == .01 and trial["vpm_dt"] == .05
    history = trial["comparison"]
    assert history == iteration["comparison"] and len(history) == 21
    old_history = baseline["comparison_history"][:21]
    assert history[0] == old_history[0]
    for key in ("coupling_step", "elapsed_flow_time", "physical_time", "full_drag_coefficient"):
        assert [row[key] for row in history] == [row[key] for row in old_history]
    assert len(candidate["outer_advances"]) == 20
    for step, row in enumerate(candidate["outer_advances"], 1):
        assert row["accepted_step"] == step and row["accepted_time"] == history[step]["elapsed_flow_time"]
    geometry = read_arrays(checked(profiles["geometry"]))
    old_geometry = read_arrays(checked(baseline["profile_geometry"]))
    assert geometry.keys() == old_geometry.keys()
    for key in geometry:
        np.testing.assert_array_equal(geometry[key], old_geometry[key])
    shared = geometry["shared_cell_ids"]
    meshes = {key: load_native_mesh(checked(row)) for key, row in checkpoints["meshes"].items()}
    geo = {key: compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False) for key, mesh in meshes.items()}
    cut = next(p for p in meshes["hybrid"]["boundary"] if p["name"] == "numericalBoundary")
    area = np.linalg.norm(geo["hybrid"]["face_area_vector"][cut["start_face"]:cut["start_face"] + cut["n_faces"]], axis=1)
    groups = {}
    for row in iteration["sweeps"]:
        step = row["coupling_step"]
        groups.setdefault(step, []).append(row)
        assert row["accepted_fvm_step"] == 5 * step and row["accepted_transfer_step"] == step + 1
        assert row["runtime_cpu_threads"] == 1 and row["map_evaluations"] == row["sweep"] + 1
        if row["sweep"] == 1:
            assert row["map_replay"]["bitwise_equal"] and row["map_replay"]["numerical_arrays_and_clocks"] == 24
            assert len(row["map_replay"]["sha256"]) == 24
        fields = read_arrays(checked(row["boundary_fields"]))
        for mode, key in (("normal_velocity", "normal_residual_rms"), ("tangential_gradient", "gradient_residual_rms")):
            delta = abs(rms(fields["post_" + mode] - fields["applied_" + mode], area) - row[key])
            assert delta < 2e-14
            differences.append(delta)
        assert row["converged"] == (row["normal_residual_rms"] <= 1e-6 and row["gradient_residual_rms"] <= 1e-6)
    assert sorted(groups) == list(range(1, 21))
    for step, rows in groups.items():
        assert [row["sweep"] for row in rows] == list(range(1, len(rows) + 1))
        assert not any(row["converged"] for row in rows[:-1])
        assert rows[-1]["converged"] or len(rows) == 12
        assert abs(rows[-1]["hybrid_drag_coefficient"] - history[step]["hybrid_drag_coefficient"]) < 2e-13
    assert iteration["converged_intervals"] == sum(rows[-1]["converged"] for rows in groups.values())
    assert [row["fvm_step"] for row in profiles["frames"]] == [0, 25, 50, 75, 100]
    assert [row["fvm_step"] for row in checkpoints["frames"]] == [0, 25, 50, 75, 100]
    old_frames = {row["coupling_step"]: row for row in baseline["profile_frames"]}
    volume = geo["hybrid"]["cell_volume"]
    near = np.max(np.abs(geo["hybrid"]["cell_centre"]), axis=1) < 1
    frame_results = []
    for profile, checkpoint in zip(profiles["frames"], checkpoints["frames"], strict=True):
        step = profile["coupling_step"]
        fields = read_arrays(checked(profile["fields"]))
        old_fields = read_arrays(checked(old_frames[step]["profile_fields"]))
        np.testing.assert_array_equal(fields["full_cell_velocity"], old_fields["full_cell_velocity"])
        if step == 0:
            assert fields.keys() == old_fields.keys()
            for key in fields:
                np.testing.assert_array_equal(fields[key], old_fields[key])
        force = {}
        for key in ("full", "hybrid"):
            raw = read_arrays(checked(checkpoint["checkpoints"][key]))
            assert json.loads(raw["metadata"].item())["mesh_hash"] == mesh_hash(meshes[key])
            state = decode_state(raw)
            assert int(state["step"]) == 5 * step
            assert abs(float(state["time"]) - history[step]["elapsed_flow_time"]) < 1e-12
            field_name = "full_cell_velocity" if key == "full" else "small_cell_velocity"
            np.testing.assert_array_equal(state["velocity"][:meshes[key]["n_cells"]], fields[field_name])
            force[key] = independent_cube_force(state, meshes[key], geo[key])
            delta = abs(force[key]["drag_coefficient"] - history[step][key + "_drag_coefficient"])
            assert delta < 2e-13
            differences.append(delta)
        error = fields["small_cell_velocity"] - fields["full_cell_velocity"][shared]
        for value, key in ((rms(error, volume), "fvm_velocity_rms_over_Uinf"), (rms(error[near], volume[near]), "fvm_near_body_velocity_rms_over_Uinf")):
            delta = abs(value - history[step][key])
            assert delta < 2e-13
            differences.append(delta)
        for name, velocity, prefix in (("full_profile", fields["full_cell_velocity"], "full"), ("small_profile", fields["small_cell_velocity"], "small"), ("reference_on_small_stencil", fields["full_cell_velocity"][shared], "small")):
            weights, values = geometry[prefix + "_weights"], velocity[geometry[prefix + "_indices"]]
            expected = np.einsum("qk,qkj->qj", weights, values)
            bound = 2 * weights.shape[1] * np.finfo(float).eps * np.einsum("qk,qkj->qj", np.abs(weights), np.abs(values))
            assert np.all(np.abs(fields[name] - expected) <= np.maximum(bound, np.finfo(float).tiny))
        metrics, combined = line_metrics(fields, geometry)
        old_metrics, old_combined = line_metrics(old_fields, geometry)
        for key in old_metrics:
            assert old_metrics[key] == old_frames[step]["metrics"][key]
        frame_results.append({"coupling_step": step, "physical_time": profile["physical_time"],
                              "baseline_metrics": old_metrics, "candidate_metrics": metrics, "candidate_independent_forces": force})
    args.output.mkdir(parents=True)
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    direct_velocity, direct_check = direct_profile(fields, geometry["position"][geometry["fluid_mask"]])
    direct_path = args.output / "final-direct-profile.npz"
    np.savez_compressed(direct_path, velocity=direct_velocity)
    sources.append(hash_file(direct_path))
    figures = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    time = np.array([row["physical_time"] for row in history])
    colors = ("#156d91", "#ad572f")
    axes[0, 0].plot(time, [r["full_drag_coefficient"] for r in history], "--", color="#333333", label="Full FVM")
    for rows, label, color in ((old_history, "RK/GBD dt=0.05", colors[0]), (history, f"RK/GBD dt={.05/n:g}", colors[1])):
        axes[0, 0].plot(time, [r["hybrid_drag_coefficient"] for r in rows], label=label, color=color)
        axes[1, 0].plot(time, [100 * r["fvm_velocity_rms_over_Uinf"] for r in rows], label=label, color=color)
    for y, ax in ((0., axes[0, 1]), (.75, axes[1, 1])):
        mask = geometry["position"][:, 1] == y
        points = geometry["position"][mask, 0]
        for values, label, color, style in ((fields["full_profile"], "Full FVM", "#333333", "--"), (old_combined, "Hybrid, RK/GBD dt=0.05", colors[0], "-"), (combined, f"Hybrid, RK/GBD dt={.05/n:g}", colors[1], "-")):
            full = np.full(len(geometry["position"]), np.nan)
            full[geometry["fluid_mask"]] = values[:, 0]
            ax.plot(points, full[mask], style, color=color, label=label)
        for x in (-1.5, 1.5):
            ax.axvline(x, color=".7", lw=.7)
        ax.set(title=f"Physical composite at t=1.5, y={y:g}, z=0", xlabel="x / D", ylabel="Streamwise velocity / U∞")
    axes[0, 0].set(title="Drag history", xlabel="Physical time", ylabel="Cd")
    axes[1, 0].set(title="Whole small-FVM velocity error", xlabel="Physical time", ylabel="Vector RMS / U∞ (%)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Matched fully 3D cube · fixed exchange and renewal interval 0.05")
    for extension in ("png", "svg"):
        path = args.output / ("comparison." + extension)
        fig.savefig(path, dpi=170)
        figures.append(hash_file(path))
    plt.close(fig)
    inviscid = report(args.inviscid_comparison)
    cadence = report(args.cadence_comparison)
    assert inviscid["schema"] == "openonda-inviscid-subcycling-comparison-3d/1"
    assert inviscid["inviscid_substeps"] == 5
    assert inviscid["baseline_history"] == old_history
    assert cadence["schema"] == "openonda-interface-cadence-comparison-3d/1"
    inviscid_child = report(ROOT / inviscid["candidate_directory"] / "trial/cube-coupled-trial.json")
    fine = next(row for row in cadence["experiments"] if row["exchange_dt"] == .01 and row["maximum_sweeps"] == 12)
    fine_child = report(ROOT / fine["directory"] / "trial/cube-coupled-trial.json")
    baseline_row = next(row for row in cadence["experiments"] if row["exchange_dt"] == .05 and row["maximum_sweeps"] == 12)
    baseline_child = report(ROOT / baseline_row["directory"] / "trial/cube-coupled-trial.json")
    assert baseline_child["comparison"] == old_history
    assert inviscid_child["comparison"] == inviscid["candidate_history"]
    assert fine_child["comparison"][-1] == fine["final_comparison"]
    timing_cases = []
    for label, child, micro_rk, micro_gbd, exchange in (
            ("Original time schedule", baseline_child, .05, .05, .05),
            ("Finer RK only", inviscid_child, .01, .05, .05),
            ("Finer RK and GBD", trial, .01, .01, .05),
            ("Finer complete exchange", fine_child, .01, .01, .01)):
        for key in FIXED_INPUTS:
            assert child[key] == trial[key], key
        assert child["sources"] == trial["sources"]
        stride = round(.05 / exchange)
        rows = child["comparison"][::stride]
        assert len(rows) == 21 and rows[0] == history[0]
        assert [row["full_drag_coefficient"] for row in rows] == [row["full_drag_coefficient"] for row in history]
        np.testing.assert_array_equal([row["physical_time"] for row in rows], [row["physical_time"] for row in history])
        timing_cases.append({"label": label, "rk_dt": micro_rk, "gbd_dt": micro_gbd,
                             "exchange_and_outer_stabilization_dt": exchange,
                             "common_endpoint_metrics": history_metrics(rows),
                             "all_endpoint_metrics": history_metrics(child["comparison"])})
    assert timing_cases[1]["common_endpoint_metrics"] == inviscid["candidate"]
    assert timing_cases[3]["common_endpoint_metrics"]["drag_relative_rms_percent"] == fine["common_endpoint_metrics"]["drag_relative_difference_rms_percent"]
    for directory in (ROOT / inviscid["candidate_directory"], ROOT / fine["directory"], ROOT / baseline_row["directory"]):
        path = directory / "trial/latest-comparison-fields.npz"
        other = read_arrays(path)
        np.testing.assert_array_equal(other["full_velocity"], fields["full_cell_velocity"][shared])
        sources.append(hash_file(path))
    sources += [hash_file(Path(__file__).resolve()), hash_file(ROOT / "studies/coupler_accuracy/verify_accepted_wake_prefix_3d.py"),
                hash_file(ROOT / "studies/coupler_accuracy/compare_inviscid_subcycling_3d.py"),
                hash_file(ROOT / "studies/coupler_accuracy/compare_interface_cadence_3d.py")]
    frozen = report(ROOT / "frozen-workspace.json")
    for row in frozen["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    unique = {row["path"]: row for row in sources}
    for row in unique.values():
        path = checked(row)
        if path.suffix == ".py":
            copy = args.output / "sources" / row["path"]
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(path.read_bytes())
    result = {"schema": "openonda-split-subcycling-comparison-3d/1", "status": "complete", "spatial_dimensions": 3,
              "candidate_directory": str(args.run.relative_to(ROOT)), "vpm_substeps": n, "timing_cases": timing_cases,
              "physical_time_bounds": [.5, 1.5], "frozen_original_files_verified": len(frozen["records"]),
              "independent_scalar_checks": len(differences), "maximum_scalar_check_difference": max(differences),
              "independently_recomputed_wall_forces": 2 * len(frame_results),
              "converged_intervals": iteration["converged_intervals"], "replayed_intervals": len(groups),
              "baseline": history_metrics(old_history), "candidate": history_metrics(history),
              "baseline_history": old_history, "candidate_history": history, "profile_frames": frame_results,
              "candidate_final_direct_check": direct_check, "figures": figures, "sources": list(unique.values()),
              "limitations": [
                  "The independent full reference and initial source fields agree bitwise. The one-substep wrapper is qualified by its separate three-interval control.",
                  "The candidate substeps native RK and GBD together. Physical exchange/renewal and outer stabilization remain at 0.05. Its contrast with finer RK alone changes GBD frequency and operator splitting together.",
                  "The selected vpm_boundary_condition panel scope excludes body velocity/gradient from particle stages while retaining it in target queries. This experiment preserves that existing behavior.",
                  "The four-schedule comparison retains both common-time and all-endpoint force statistics; the finer complete exchange also changes the outer stabilization frequency. The original finer-exchange experiment has no saved exterior profile frames.",
                  "Complete denotes this short diagnostic comparison, not developed-wake validation or achievement of the requested force/profile agreement.",
              ]}
    (args.output / "split-subcycling-comparison-3d.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "independent_scalar_checks", "maximum_scalar_check_difference", "converged_intervals", "baseline", "candidate")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-verification", type=Path, required=True)
    parser.add_argument("--control-schedule", type=Path, required=True)
    parser.add_argument("--schedule-verification", type=Path, required=True)
    parser.add_argument("--inviscid-comparison", type=Path, required=True)
    parser.add_argument("--cadence-comparison", type=Path, required=True)
    parser.add_argument("--baseline-prefix", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("wrapper_verification", "control_schedule", "schedule_verification", "inviscid_comparison", "cadence_comparison", "baseline_prefix", "run", "output"):
        setattr(args, key, getattr(args, key).resolve())
    run(args)
