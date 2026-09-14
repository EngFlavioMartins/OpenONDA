#!/usr/bin/env python3
"""Verify a fixed accepted prefix without modifying a running 3D wake trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from source.coupler.renewal_projection import gaussian_velocity_operator
from source.solvers.fvm.io.backup import decode_state, mesh_hash
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.vpm.boundary_elements.panels.kernels.induced_velocity import (
    compute_source_induced_velocity_kernel,
)
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.cube_snapshot_induction import volume_velocity
from studies.coupler_accuracy.native_volume_induction_3d import triangle_source_integrals
from studies.coupler_accuracy.verify_sampled_face_trials_3d import independent_cube_force


def rms(value, weight=None):
    value = np.asarray(value)
    squared = value**2 if value.ndim == 1 else np.sum(value**2, axis=1)
    return float(np.sqrt(np.average(squared, weights=weight)))


def norms(value):
    return {"vector_rms_over_Uinf": rms(value), "streamwise_rms_over_Uinf": rms(value[:, 0]),
            "vector_maximum_over_Uinf": float(np.linalg.norm(value, axis=1).max())}


def capture_json(path, destination):
    # Parent reports use write_text rather than atomic replacement. Only save
    # a parseable, unchanged read; a read race is not a stopped simulation.
    for _ in range(5):
        before = path.read_bytes()
        try:
            decoded = json.loads(before)
        except json.JSONDecodeError:
            time.sleep(.05)
            continue
        if path.read_bytes() == before:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(before)
            return decoded
    raise RuntimeError(f"Could not capture a stable report: {path}")


def direct_profile(fields, points):
    """Evaluate the same saved sources in f64, with independent kernel checks."""
    position = fields["particle_position"].astype(float)
    strength = fields["particle_vortex_strength"].astype(float)
    radius = fields["particle_core_radius"].astype(float)
    particles = volume_velocity(points, position, strength, radius)[0]
    selected = np.unique(np.linspace(0, len(points) - 1, 8, dtype=int))
    matrix = gaussian_velocity_operator(points[selected], position, radius)
    matrix_velocity = (matrix @ strength.ravel()).reshape(-1, 3)
    particle_check = float(np.max(np.abs(matrix_velocity - particles[selected])))
    assert particle_check < 2e-12
    vertices = fields["panel_vertices"].astype(float)
    panel_normals = fields["panel_normals"].astype(float)
    panel_strength = fields["panel_source_strength"].astype(float)
    vector_integral = triangle_source_integrals(points, vertices)[1]
    body = np.einsum("pti,t->pi", vector_integral, panel_strength) / (4 * np.pi)
    production_body = np.zeros_like(points)
    compute_source_induced_velocity_kernel(np.ascontiguousarray(vertices), np.ascontiguousarray(panel_normals),
                                           np.ascontiguousarray(panel_strength), np.ascontiguousarray(points), production_body)
    body_check = float(np.max(np.abs(body - production_body)))
    assert body_check < 2e-11
    result = particles + body + [1., 0., 0.]
    assert np.all(np.isfinite(result))
    return result, {"particle_matrix_check_maximum_difference": particle_check,
                    "panel_kernel_check_maximum_difference": body_check,
                    "direct_minus_saved_query": norms(result - fields["vpm_profile"])}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources, captures, parent = [], [], {}
    report_names = {"iteration": "interface-iteration-3d.json", "trial": "trial/cube-coupled-trial.json",
                    "history": "trial/comparison-history.json", "profiles": "profile-observation-3d.json",
                    "checkpoints": "accepted-fvm-checkpoints-3d.json"}
    for role, name in report_names.items():
        path = args.run / name
        copy = args.output / "captured-reports" / name
        parent[role] = capture_json(path, copy)
        sources.append(hash_file(copy))
        captures.append({"role": role, "original_path": str(path.relative_to(ROOT)), "captured": hash_file(copy)})
    for role in ("iteration", "trial", "profiles", "checkpoints"):
        assert parent[role]["status"] in ("running", "complete") and parent[role]["spatial_dimensions"] == 3
        for row in parent[role]["sources"]:
            assert hash_file(ROOT / row["path"]) == row
            sources.append(row)
    manifest_path = ROOT / "frozen-workspace.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    sources.append(hash_file(manifest_path))
    trial, profile, checkpoints, iteration = (parent[key] for key in ("trial", "profiles", "checkpoints", "iteration"))
    assert trial["fvm_dt"] == .01 and trial["vpm_dt"] == .05
    assert trial["particle_spacing"] == trial["requested_wall_spacing"] == .0625
    assert trial["fvm_numerics"]["kinematic_viscosity"] == .001 and trial["fvm_numerics"]["sgs"] == "none"
    assert trial["small_cells"] == 16936 and trial["full_cells"] == 53752
    assert trial["source_seed_time"] == .5 and trial["identical_native_shared_cells"]
    assert trial["requested_coupling_steps"] >= args.through_step
    history = [row for row in parent["history"] if row["coupling_step"] <= args.through_step]
    assert [row["coupling_step"] for row in history] == list(range(args.through_step + 1))
    assert profile["profile_every_fvm_steps"] == 25 and args.through_step % 5 == 0
    ticks = list(range(0, 5 * args.through_step + 1, 25))
    frames = {row["fvm_step"]: row for row in profile["frames"] if row["fvm_step"] <= ticks[-1]}
    backups = {row["fvm_step"]: row for row in checkpoints["frames"] if row["fvm_step"] <= ticks[-1]}
    assert sorted(frames) == sorted(backups) == ticks
    short_path = args.short_run / "interface-iteration-3d.json"
    short = json.loads(short_path.read_text())
    assert short["status"] == "complete" and short["spatial_dimensions"] == 3
    assert len(short["comparison"]) == 21 and history[:21] == short["comparison"]
    sources.append(hash_file(short_path))
    geometry_path = ROOT / profile["geometry"]["path"]
    assert hash_file(geometry_path) == profile["geometry"]
    geometry = read_arrays(geometry_path)
    sources.append(profile["geometry"])
    points, fluid, small = (geometry[key] for key in ("position", "fluid_mask", "small_mask"))
    assert points.shape == (418, 3) and not np.any(points[:, 2])
    np.testing.assert_array_equal(fluid, np.max(np.abs(points), axis=1) > .5)
    np.testing.assert_array_equal(small, fluid & (np.max(np.abs(points), axis=1) <= 1.5))
    shared = geometry["shared_cell_ids"]
    meshes, mesh_geometry = {}, {}
    for name, row in checkpoints["meshes"].items():
        assert hash_file(ROOT / row["path"]) == row
        meshes[name] = load_native_mesh(ROOT / row["path"])
        mesh_geometry[name] = compute_mesh_geometry(meshes[name], gradient_scheme="gauss", compute_lsq=False)
        sources.append(row)
    np.testing.assert_allclose(mesh_geometry["full"]["cell_centre"][shared], mesh_geometry["hybrid"]["cell_centre"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(mesh_geometry["full"]["cell_volume"][shared], mesh_geometry["hybrid"]["cell_volume"], rtol=0, atol=1e-14)
    cut = next(p for p in meshes["hybrid"]["boundary"] if p["name"] == "numericalBoundary")
    face_ids = np.arange(cut["start_face"], cut["start_face"] + cut["n_faces"])
    area = np.linalg.norm(mesh_geometry["hybrid"]["face_area_vector"][face_ids], axis=1)
    normal = mesh_geometry["hybrid"]["face_area_vector"][face_ids] / area[:, None]
    sweep_groups, numerical_differences = {}, []
    for row in iteration["sweeps"]:
        if row["coupling_step"] > args.through_step:
            continue
        sweep_groups.setdefault(row["coupling_step"], []).append(row)
        assert row["runtime_cpu_threads"] == 1
        assert row["accepted_fvm_step"] == 5 * row["coupling_step"]
        assert row["accepted_transfer_step"] == row["coupling_step"] + 1
        if row["sweep"] == 1:
            assert row["map_replay"]["bitwise_equal"] and row["map_replay"]["numerical_arrays_and_clocks"] == 24
            assert len(row["map_replay"]["sha256"]) == 24
        assert hash_file(ROOT / row["boundary_fields"]["path"]) == row["boundary_fields"]
        boundary = read_arrays(ROOT / row["boundary_fields"]["path"])
        for mode, key in (("normal_velocity", "normal_residual_rms"), ("tangential_gradient", "gradient_residual_rms")):
            delta = abs(rms(boundary["post_" + mode] - boundary["applied_" + mode], area) - row[key])
            assert delta < 2e-14
            numerical_differences.append(delta)
        expected = row["normal_residual_rms"] <= iteration["normal_tolerance"] and row["gradient_residual_rms"] <= iteration["gradient_tolerance"]
        assert row["converged"] == expected and row["map_evaluations"] == row["sweep"] + 1
        for phase in ("applied", "post"):
            np.testing.assert_allclose(np.sum(boundary[phase + "_velocity"] * normal, axis=1), boundary[phase + "_normal_velocity"], rtol=0, atol=2e-13)
        sources.append(row["boundary_fields"])
    assert sorted(sweep_groups) == list(range(1, args.through_step + 1))
    for step, rows in sweep_groups.items():
        assert [row["sweep"] for row in rows] == list(range(1, len(rows) + 1))
        assert len(rows) <= iteration["maximum_sweeps"] and not any(row["converged"] for row in rows[:-1])
        assert rows[-1]["converged"] or len(rows) == iteration["maximum_sweeps"]
        assert abs(rows[-1]["hybrid_drag_coefficient"] - history[step]["hybrid_drag_coefficient"]) < 2e-13
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    frame_records = []
    small_volume = mesh_geometry["hybrid"]["cell_volume"]
    near = np.max(np.abs(mesh_geometry["hybrid"]["cell_centre"]), axis=1) < 1
    query_points = points[fluid]
    direct_rows = sorted({0, *args.direct_steps, args.through_step})
    assert all(step * 5 in ticks for step in direct_rows)
    for tick in ticks:
        observation, backup = frames[tick], backups[tick]
        step = tick // 5
        assert observation["observer_state_bitwise_unchanged"] and backup["state_bitwise_unchanged"]
        assert observation["state_fingerprints"] == backup["state_fingerprints"]
        assert observation["coupling_step"] == backup["coupling_step"] == step
        assert observation["physical_time"] == backup["physical_time"] == history[step]["physical_time"]
        field_path = ROOT / observation["fields"]["path"]
        assert hash_file(field_path) == observation["fields"]
        fields = read_arrays(field_path)
        sources.append(observation["fields"])
        force = {}
        for name, checkpoint in backup["checkpoints"].items():
            assert hash_file(ROOT / checkpoint["path"]) == checkpoint
            raw = read_arrays(ROOT / checkpoint["path"])
            metadata = json.loads(raw["metadata"].item())
            assert metadata["mesh_hash"] == mesh_hash(meshes[name])
            state = decode_state(raw)
            assert int(state["step"]) == tick
            assert abs(float(state["time"]) - history[step]["elapsed_flow_time"]) < 1e-12
            field_key = "full_cell_velocity" if name == "full" else "small_cell_velocity"
            np.testing.assert_array_equal(state["velocity"][:meshes[name]["n_cells"]], fields[field_key])
            force[name] = independent_cube_force(state, meshes[name], mesh_geometry[name])
            force[name]["coefficient_vector"] = (2 * (np.asarray(force[name]["pressure_force"]) + force[name]["viscous_force"])).tolist()
            delta = abs(force[name]["drag_coefficient"] - history[step][name + "_drag_coefficient"])
            assert delta < 2e-13
            numerical_differences.append(delta)
            sources.append(checkpoint)
        error = fields["small_cell_velocity"] - fields["full_cell_velocity"][shared]
        for actual, key in ((rms(error, small_volume), "fvm_velocity_rms_over_Uinf"),
                            (rms(error[near], small_volume[near]), "fvm_near_body_velocity_rms_over_Uinf")):
            delta = abs(actual - history[step][key])
            assert delta < 2e-13
            numerical_differences.append(delta)
        for name, value, prefix in (("full_profile", fields["full_cell_velocity"], "full"),
                                    ("small_profile", fields["small_cell_velocity"], "small"),
                                    ("reference_on_small_stencil", fields["full_cell_velocity"][shared], "small")):
            weights, values = geometry[prefix + "_weights"], value[geometry[prefix + "_indices"]]
            expected = np.einsum("qk,qkj->qj", weights, values)
            bound = 2 * weights.shape[1] * np.finfo(float).eps * np.einsum("qk,qkj->qj", np.abs(weights), np.abs(values))
            assert np.all(np.abs(expected - fields[name]) <= np.maximum(bound, np.finfo(float).tiny))
        combined = fields["vpm_profile"].astype(np.float64, copy=True)
        # Preserve the f64 FVM samples when assembling the physical hybrid.
        # The saved particle query array itself is f32.
        combined[small[fluid]] = fields["small_profile"]
        np.testing.assert_array_equal(combined[small[fluid]], fields["small_profile"])
        matched_reference = fields["full_profile"].copy()
        matched_reference[small[fluid]] = fields["reference_on_small_stencil"]
        metrics = {}
        for line_y, line_name in ((0., "centreline"), (.75, "offaxis_y075")):
            line = query_points[:, 1] == line_y
            regions = {"whole": line, "upstream_exterior": line & (query_points[:, 0] < -1.5),
                       "downstream_exterior": line & (query_points[:, 0] > 1.5),
                       "near_wake_exterior": line & (query_points[:, 0] > 1.5) & (query_points[:, 0] <= 4),
                       "far_wake_exterior": line & (query_points[:, 0] > 4)}
            for region, mask in regions.items():
                metrics[line_name + "_" + region + "_vpm"] = norms((fields["vpm_profile"] - fields["full_profile"])[mask])
            metrics[line_name + "_composite_raw"] = norms((combined - fields["full_profile"])[line])
            metrics[line_name + "_composite_matched_sampling"] = norms((combined - matched_reference)[line])
            small_line = points[small, 1] == line_y
            metrics[line_name + "_small_fvm_matched_sampling"] = norms((fields["small_profile"] - fields["reference_on_small_stencil"])[small_line])
            metrics[line_name + "_reference_stencil_difference"] = norms(fields["full_profile"][line & small[fluid]] - fields["reference_on_small_stencil"][small_line])
        direct = None
        if step in direct_rows:
            exact, direct = direct_profile(fields, query_points)
            direct_path = args.output / f"direct-profile-step-{step:06d}.npz"
            np.savez_compressed(direct_path, position=query_points, direct_velocity=exact, saved_query_velocity=fields["vpm_profile"],
                                full_profile=fields["full_profile"])
            direct["fields"] = hash_file(direct_path)
            sources.append(direct["fields"])
        frame_records.append({"coupling_step": step, "fvm_step": tick, "physical_time": observation["physical_time"],
                              "forces": force, "metrics": metrics, "direct_profile_check": direct, "profile_fields": observation["fields"]})
        print(json.dumps({"verified_step": step, "time": observation["physical_time"],
                          "force_difference": (np.asarray(force["hybrid"]["coefficient_vector"]) - force["full"]["coefficient_vector"]).tolist(),
                          "centreline_downstream_rms": metrics["centreline_downstream_exterior_vpm"]["vector_rms_over_Uinf"],
                          "direct_check": direct["direct_minus_saved_query"] if direct else None}), flush=True)
    ti.reset()
    paths = [Path(__file__).resolve()] + [ROOT / name for name in (
        "source/coupler/renewal_projection.py", "source/solvers/fvm/io/backup.py", "source/solvers/fvm/io/mesh_storage.py",
        "source/solvers/fvm/mesh/geometry.py", "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py", "studies/coupler_accuracy/cube_snapshot_induction.py",
        "studies/coupler_accuracy/native_volume_induction_3d.py", "studies/coupler_accuracy/cube_boundary_oracle.py",
        "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py", "studies/coupler_accuracy/verify_sampled_face_trials_3d.py")]
    sources += [hash_file(path) for path in paths]
    # Archive every code dependency already named by the running trial too.
    unique = {row["path"]: row for row in sources}
    for row in unique.values():
        path = ROOT / row["path"]
        assert hash_file(path) == row
        if path.suffix == ".py":
            copy = args.output / "sources" / path.relative_to(ROOT)
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(path.read_bytes())
    for row in manifest["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    relative_drag = np.array([(r["hybrid_drag_coefficient"] - r["full_drag_coefficient"]) / r["full_drag_coefficient"] for r in history[1:]])
    vector_force_error = np.array([np.asarray(r["forces"]["hybrid"]["coefficient_vector"]) - r["forces"]["full"]["coefficient_vector"] for r in frame_records[1:]])
    result = {"schema": "openonda-long-wake-prefix-verification-3d/2", "status": "complete", "scope": "fixed accepted prefix",
              "spatial_dimensions": 3, "run_directory": str(args.run.relative_to(ROOT)), "captured_reports": captures,
              "parent_status_at_capture": {k: parent[k]["status"] for k in ("iteration", "trial", "profiles", "checkpoints")},
              "through_coupling_step": args.through_step, "physical_time_bounds": [history[0]["physical_time"], history[-1]["physical_time"]],
              "frozen_original_files_verified": len(manifest["records"]), "recorded_short_prefix_equal_intervals": 20,
              "converged_intervals": sum(r[-1]["converged"] for r in sweep_groups.values()), "replayed_intervals": len(sweep_groups),
              "independent_scalar_checks": len(numerical_differences), "maximum_scalar_check_difference": max(numerical_differences),
              "independently_recomputed_wall_forces": 2 * len(frame_records), "profile_frames": frame_records,
              "comparison_history": history, "profile_geometry": profile["geometry"], "sources": list(unique.values()),
              "drag_relative_rms_percent": float(100 * np.sqrt(np.mean(relative_drag**2))),
              "maximum_absolute_relative_drag_percent": float(100 * np.max(np.abs(relative_drag))),
              "force_component_difference_rms_at_profile_times": np.sqrt(np.mean(vector_force_error**2, axis=0)).tolist(),
              "elapsed_seconds": time.perf_counter() - started,
              "limitations": [
                  "Complete means this fixed prefix was verified, not that the parent long run completed, the wake is developed, or the requested agreement was achieved.",
                  "Drag is recorded every exchange; all three wall-force components are independently reconstructed only at profile/checkpoint times, every 0.25 flow-time units.",
                  "Short-run prefix agreement compares the recorded 20-interval history. No separate unobserved 70-interval control or bitwise comparison of every intermediate primitive state is claimed.",
                  "The profiles are one-dimensional observations of the actual 3D flow. Upstream, near-wake and far-wake regions are reported separately.",
                  "The composite uses hybrid FVM inside the small box and VPM outside. Raw and matched-sampling comparisons separately expose interpolation-stencil differences.",
                  "Direct profile checks use the same saved rounded particles and body strengths. Their difference from runtime queries includes tree approximation and evaluation arithmetic, not particle representation error.",
                  "Independent direct-kernel cross-checks use eight particle targets and all profile panel targets at each requested direct-check time; the body strengths are not re-solved.",
                  "History RMS statistics exclude the common initial observation; the complete unshifted history is retained.",
              ]}
    (args.output / "long-wake-prefix-verification-3d.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--short-run", type=Path, required=True)
    parser.add_argument("--through-step", type=int, required=True)
    parser.add_argument("--direct-steps", type=int, nargs="*", default=[20, 40])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.through_step < 20 or args.through_step % 5:
        parser.error("Require at least twenty exchanges and a saved profile endpoint")
    args.run, args.short_run, args.output = args.run.resolve(), args.short_run.resolve(), args.output.resolve()
    run(args)
