#!/usr/bin/env python3
"""Measure a stricter FMM stage separation on one fixed physical 3D source state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import openonda.vpm as vpm
from source.coupler.vorticity_transfer import _particle_state_snapshot
from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.induction.fmm import device as fmm_device
from studies.coupler_accuracy.cube_boundary_oracle import CASE, ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.particle_stage_induction_audit_3d import rms


def signature(solver):
    values = _particle_state_snapshot(solver)
    values["panel_strength"] = solver.panel_solver.lattice.source_strength.to_numpy()
    values["clocks"] = np.array([solver.time, solver.step, solver.time_step_size, solver.particles.step])
    return {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for key, value in values.items()}



def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen_path = ROOT / "frozen-workspace.json"
    frozen = json.loads(frozen_path.read_text())
    for row in frozen["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    reference_report = json.loads(args.reference.read_text())
    assert reference_report["status"] == "complete" and reference_report["schema"] == "openonda-particle-stage-induction-audit-3d/1"
    reference_path = args.reference.parent / "particle-stage-reference-fields.npz"
    assert hash_file(reference_path) in reference_report["sources"]
    reference = read_arrays(reference_path)
    prefix = json.loads(args.prefix.read_text())
    assert prefix["status"] == "complete" and prefix["schema"] == "openonda-long-wake-prefix-verification-3d/2"
    sources = [hash_file(frozen_path), hash_file(args.prefix), *prefix["sources"], hash_file(args.reference), *reference_report["sources"]]
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    frame = next(row for row in prefix["profile_frames"] if row["coupling_step"] == 20)
    fields = read_arrays(ROOT / frame["profile_fields"]["path"])
    manifest_path = args.backup_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    backup_path = args.backup_directory / manifest["artifacts"]["vpm"]
    assert hash_file(backup_path)["sha256"] == manifest["artifact_sha256"]["vpm"]
    assert manifest["vpm_step"] == 20 and manifest["time"] == 1.
    sources += [hash_file(manifest_path), hash_file(backup_path)]
    args.output.mkdir(parents=True)
    original_separation = fmm_device._GEOMETRIC_SEPARATION_FACTOR
    assert original_separation == 3.0
    # Set before any FMM kernel is compiled in this fresh process.
    fmm_device._GEOMETRIC_SEPARATION_FACTOR = args.separation
    panel = vpm.PanelSolver(max_n_panels=128, float_dtype="f32", linear_solver="SCIPY",
                            boundary_condition_type="NEUMANN", density=1,
                            freestream_velocity=np.array([1., 0., 0.]), coupling_scope="vpm_boundary_condition")
    domain = (-5.5, 10.5, -5.5, 5.5, -5.5, 5.5)
    case = vpm.VPMCase(
        numerics=vpm.Numerics(time_step_size=.05, freestream_velocity=(1, 0, 0),
                             viscous=vpm.ViscousConfig.gbd(particle_spacing=.0625, core_radius_ratio=1,
                                                         padding=5, kinematic_viscosity=.001,
                                                         threshold_mode="absolute", threshold=.02 * .0625**3, max_nodes=500000),
                             integrator=vpm.RK2(), induction=vpm.FMMInduction(), turbulence=vpm.TurbulenceConfig.dns(),
                             stabilization=vpm.StabilizationConfig.bounded_domain(domain), particle_kernel="GAUSSIAN",
                             precision="f32", compute_device="CPU", max_n_particles=500000,
                             max_evaluation_points=500000, domain_bounds=domain, write_precision="f32", panel_solver=panel,
                             bodies=(vpm.PanelBodySetup(stl=CASE / "assets/cube.stl", uid="body", reference_area=1),)),
        backup=vpm.Backup(interval_steps=0, directory="solution", log_directory="solution"),
        samplers=vpm.Samplers(samples=()), run=vpm.RunPlan(steps=400), directory=args.output / "vpm-probe")
    solver = vpm.VPMSolver(case)
    try:
        solver.load_backup(backup_path)
        assert solver.time == 1. and solver.step == 20 and solver.flow_model == "DNS"
        assert solver.axisymmetric_axis < 0 and solver.vlm_solver is None and solver.n_sources == 0
        for key, value in (("particle_position", solver.particle_position),
                           ("particle_vortex_strength", solver.particle_vortex_strength),
                           ("particle_core_radius", solver.particle_core_radius)):
            np.testing.assert_array_equal(value, fields[key])
        lattice, count = panel.lattice, panel.lattice.n_panels
        assert count == 108 and count < panel.far_field_min_panels
        np.testing.assert_array_equal(lattice.vertex_position.to_numpy()[:count], fields["panel_vertices"])
        np.testing.assert_array_equal(lattice.normal.to_numpy()[:count], fields["panel_normals"])
        strength = lattice.source_strength.to_numpy()
        strength[:count] = fields["panel_source_strength"]
        lattice.source_strength.from_numpy(strength)
        names = ("body_velocity", "body_velocity_field", "body_velocity_gradient", "body_velocity_gradient_field", "velocity_override", "velocity_override_gradient")
        assert all(getattr(solver.physics, key, None) is None for key in names)
        initial = signature(solver)
        positions = solver.particle_position.astype(float)
        cube_q = np.abs(positions) - .5
        clearance = np.linalg.norm(np.maximum(cube_q, 0), axis=1) + np.minimum(np.max(cube_q, axis=1), 0)
        assert clearance.min() > 1e-3
        regions = {"near_body": np.max(np.abs(positions), axis=1) < 1,
                   "near_wake": (positions[:, 0] > 1.5) & (positions[:, 0] <= 4) & (np.max(np.abs(positions[:, 1:]), axis=1) < 1.5)}
        rng = np.random.default_rng(20260914)
        selected = np.sort(np.concatenate([rng.choice(np.flatnonzero(mask), min(128, int(mask.sum())), replace=False) for mask in regions.values()]))
        assert len(np.unique(selected)) == len(selected)

        def evaluate(label):
            solver.stage_rhs.evaluate(
                StageState(position=solver.particles.position, vortex_strength=solver.particles.vortex_strength,
                           core_radius=solver.particles.core_radius, count=len(solver.particles), time=solver.time, stage_index=0),
                solver.time, StageRates(velocity=solver.integrator.stage_velocity[0],
                                        vortex_strength_rate=solver.integrator.stage_strength_rate[0],
                                        velocity_gradient=solver.integrator.stage_velocity_gradient))
            result = {"velocity": solver.physics._download_vector_field(solver.integrator.stage_velocity[0], len(solver.particles)).copy(),
                      "strength_rate": solver.physics._download_vector_field(solver.integrator.stage_strength_rate[0], len(solver.particles)).copy(),
                      "gradient": solver.physics._download_matrix_field(solver.integrator.stage_velocity_gradient, len(solver.particles)).copy()}
            assert all(np.all(np.isfinite(value)) for value in result.values())
            assert signature(solver) == initial
            path = args.output / (label + ".npz")
            np.savez_compressed(path, **result)
            sources.append(hash_file(path))
            return result

        prior_evaluations = int(solver.stage_rhs.induction.diagnostics.stage_evaluations)
        first = evaluate("first-stage")
        started = time.perf_counter()
        replay = evaluate("replayed-stage")
        replay_seconds = time.perf_counter() - started
        for key in first:
            np.testing.assert_array_equal(first[key], replay[key])
        np.testing.assert_array_equal(reference["indices"], selected)
        np.testing.assert_array_equal(reference["position"], positions[selected])
        if args.separation == 3.0:
            base_row = next(row for row in reference_report["sources"] if row["path"].endswith("body-stage-reference-at-one-point-five/native-stage-without-body.npz"))
            base = read_arrays(ROOT / base_row["path"])
            for key in first:
                np.testing.assert_array_equal(first[key], base[key])
        metrics = {}
        for name, mask in regions.items():
            sample = mask[selected]
            delta_u = (first["velocity"][selected] - reference["direct_velocity"])[sample]
            delta_j = (first["gradient"][selected] - reference["direct_gradient"])[sample]
            delta_rate = (first["strength_rate"][selected] - reference["direct_strength_rate"])[sample]
            metrics[name] = {"targets": int(sample.sum()), "velocity_error_rms_over_Uinf": rms(delta_u),
                             "gradient_error_frobenius_rms": rms(delta_j),
                             "gradient_relative_frobenius_error": rms(delta_j) / rms(reference["direct_gradient"][sample]),
                             "strength_rate_error_rms": rms(delta_rate),
                             "strength_rate_relative_error": rms(delta_rate) / rms(reference["direct_strength_rate"][sample])}
        induction = solver.stage_rhs.induction
        work = induction.workspace
        assert int(induction.diagnostics.stage_evaluations) - prior_evaluations == 2
        result = {"schema": "openonda-particle-stage-separation-probe-3d/1", "status": "complete", "spatial_dimensions": 3,
                  "physical_time": 1.5, "particles": len(positions), "geometric_separation_factor": args.separation,
                  "native_factor": original_separation, "native_control_bitwise_equal": args.separation == 3.0,
                  "replayed_stage_bitwise_equal": True, "primary_fields_and_clocks_unchanged": signature(solver) == initial,
                  "replayed_evaluation_and_archive_seconds": replay_seconds,
                  "m2l_pairs": int(work._m2l_count[None]), "near_pairs": int(work._near_count[None]),
                  "p2p_interactions_excluding_self": int(work._p2p_particle_count[None]),
                  "regions": metrics, "frozen_original_files_verified": len(frozen["records"]),
                  "limitations": [
                      "Only the internal geometric acceptance factor changes, before Taichi compilation. Fixed expansion order, kernel-tail tolerances, precision, stretching and sources remain the same.",
                      "This evaluates one physical stage twice; it advances neither particles nor FVM. Accuracy changes here do not establish improvement of a coupled trajectory.",
                      "The reported second-evaluation time includes field downloads, source-state hashing and compressed output. It is a diagnostic cost observation, not a clean FMM benchmark."]}
    finally:
        solver.physics.body_velocity_field = None
        solver.physics.body_velocity_gradient = None
        solver.close()
        fmm_device._GEOMETRIC_SEPARATION_FACTOR = original_separation
    paths = [Path(__file__).resolve(), ROOT / "source/solvers/vpm/core/solver.py",
             ROOT / "source/solvers/vpm/physics/stage_rhs.py", ROOT / "source/solvers/vpm/physics/induction/stretching.py",
             ROOT / "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
             ROOT / "source/solvers/vpm/physics/induction/fmm/device.py",
             ROOT / "studies/coupler_accuracy/particle_stage_induction_audit_3d.py"]
    sources += [hash_file(path) for path in paths]
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    for path in paths:
        target = args.output / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for row in frozen["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    result["sources"] = list({row["path"]: row for row in sources}.values())
    (args.output / "particle-stage-separation-probe-3d.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "limitations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--separation", type=float, choices=(3., 4.5, 6.), required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--backup-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.reference = args.reference.resolve()
    args.prefix, args.backup_directory, args.output = (path.resolve() for path in (args.prefix, args.backup_directory, args.output))
    run(args)
