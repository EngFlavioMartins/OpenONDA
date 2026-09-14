#!/usr/bin/env python3
"""Probe actual particle-stage body velocity/stretching on a saved 3D state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from source.coupler.vorticity_transfer import _particle_state_snapshot
from source.solvers.vpm.boundary_elements.panels.kernels.induced_velocity import (
    compute_source_induced_velocity_kernel,
)
from source.solvers.vpm.physics.induction.base import StageRates, StageState
from studies.coupler_accuracy.cube_boundary_oracle import CASE, ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.native_volume_induction_3d import triangle_source_integrals


def signature(solver):
    values = _particle_state_snapshot(solver)
    values["panel_strength"] = solver.panel_solver.lattice.source_strength.to_numpy()
    values["clocks"] = np.array([solver.time, solver.step, solver.time_step_size, solver.particles.step])
    return {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for key, value in values.items()}


def vector_rms(value):
    return float(np.sqrt(np.mean(np.sum(np.asarray(value, dtype=float)**2, axis=1))))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    prefix = json.loads(args.prefix.read_text())
    assert prefix["status"] == "complete" and prefix["schema"] == "openonda-long-wake-prefix-verification-3d/2"
    sources = [hash_file(args.prefix), *prefix["sources"]]
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
        circulation = solver.particle_vortex_strength.astype(float)
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

        base = evaluate("native-stage-without-body")
        query_without = solver.compute_velocity_at_points(positions[selected], include_freestream=True, include_body=False)
        query_with = solver.compute_velocity_at_points(positions[selected], include_freestream=True, include_body=True)
        native_body = panel.compute_induced_velocity(positions[selected])
        solver.physics.body_velocity_field = panel.accumulate_induced_velocity_on_field
        with_velocity = evaluate("stage-with-body-velocity")
        np.testing.assert_array_equal(with_velocity["strength_rate"], base["strength_rate"])
        np.testing.assert_array_equal(with_velocity["gradient"], base["gradient"])
        solver.physics.body_velocity_gradient = panel.compute_induced_velocity_gradient
        with_gradient = evaluate("stage-with-native-body-gradient")
        solver.physics.body_velocity_field = None
        solver.physics.body_velocity_gradient = None
        replay = evaluate("native-stage-replay")
        for key in base:
            np.testing.assert_array_equal(base[key], replay[key])
        delta_velocity = with_velocity["velocity"][selected].astype(float) - base["velocity"][selected]
        bound = 8 * np.finfo(np.float32).eps * (1 + np.abs(base["velocity"][selected]) + np.abs(native_body))
        assert np.all(np.abs(delta_velocity - native_body) <= bound)
        native_gradient = panel.compute_induced_velocity_gradient(positions[selected])
        expected_rate = np.einsum("nji,nj->ni", native_gradient, circulation[selected])
        observed_rate = with_gradient["strength_rate"][selected].astype(float) - base["strength_rate"][selected]
        rate_bound = 16 * np.finfo(np.float32).eps * (np.abs(base["strength_rate"][selected]) + np.abs(expected_rate) + 1e-12)
        assert np.all(np.abs(expected_rate - observed_rate) <= rate_bound)
        vertices, normals, sigma = fields["panel_vertices"].astype(float), fields["panel_normals"].astype(float), fields["panel_source_strength"].astype(float)

        def precise_velocity(points):
            output = np.zeros_like(points, dtype=float)
            compute_source_induced_velocity_kernel(np.ascontiguousarray(vertices), np.ascontiguousarray(normals),
                                                   np.ascontiguousarray(sigma), np.ascontiguousarray(points, dtype=float), output)
            return output

        points = positions[selected]
        analytical = np.einsum("pti,t->pi", triangle_source_integrals(points, vertices)[1], sigma) / (4 * np.pi)
        kernel_difference = float(np.max(np.abs(analytical - precise_velocity(points))))
        assert kernel_difference < 2e-11

        def gradient(probe):
            result = np.empty((len(points), 3, 3))
            for axis in range(3):
                offset = np.eye(3)[axis] * probe
                result[:, :, axis] = (precise_velocity(points - 2 * offset) - 8 * precise_velocity(points - offset)
                                      + 8 * precise_velocity(points + offset) - precise_velocity(points + 2 * offset)) / (12 * probe)
            return result

        coarse_gradient, fine_gradient = gradient(2e-4), gradient(1e-4)
        gradient_refinement_difference = float(np.max(np.abs(coarse_gradient - fine_gradient)))
        assert gradient_refinement_difference < 2e-7
        path = args.output / "selected-targets.npz"
        np.savez_compressed(path, particle_indices=selected, position=points, circulation=circulation[selected],
                            native_query_with_body=query_with, native_query_without_body=query_without,
                            native_body_velocity=native_body, precise_body_velocity=analytical,
                            native_body_gradient=native_gradient, coarse_body_gradient=coarse_gradient,
                            fine_body_gradient=fine_gradient)
        sources.append(hash_file(path))
        region_results = {}
        for name, mask in regions.items():
            sample = mask[selected]
            region_results[name] = {
                "particle_count": int(mask.sum()), "selected_points": int(sample.sum()),
                "body_velocity_rms_over_Uinf": vector_rms((with_velocity["velocity"] - base["velocity"])[mask]),
                "body_strength_rate_change_rms": vector_rms((with_gradient["strength_rate"] - base["strength_rate"])[mask]),
                "body_strength_rate_change_over_native_rate_rms": vector_rms((with_gradient["strength_rate"] - base["strength_rate"])[mask]) / vector_rms(base["strength_rate"][mask]),
                "stage_minus_query_without_body_rms_over_Uinf": vector_rms((base["velocity"][selected] - query_without)[sample]),
                "stage_minus_query_with_body_rms_over_Uinf": vector_rms((base["velocity"][selected] - query_with)[sample]),
                "completed_stage_minus_query_with_body_rms_over_Uinf": vector_rms((with_velocity["velocity"][selected] - query_with)[sample]),
                "native_body_gradient_error_frobenius_rms": float(np.sqrt(np.mean(np.sum((native_gradient[sample] - fine_gradient[sample])**2, axis=(1, 2))))),
                "precise_body_gradient_frobenius_rms": float(np.sqrt(np.mean(np.sum(fine_gradient[sample]**2, axis=(1, 2))))),
            }
        assert signature(solver) == initial
        result = {"schema": "openonda-body-stage-snapshot-probe-3d/1", "status": "complete", "spatial_dimensions": 3,
                  "physical_time": 1.5, "particles": len(positions), "minimum_cube_clearance": float(clearance.min()),
                  "primary_fields_and_clocks_unchanged": True, "native_stage_replay_bitwise_equal": True,
                  "independent_panel_kernel_difference": kernel_difference,
                  "precise_gradient_probe_refinement_maximum_difference": gradient_refinement_difference,
                  "body_velocity_increment_maximum_check_difference": float(np.max(np.abs(delta_velocity - native_body))),
                  "body_strength_rate_increment_maximum_check_difference": float(np.max(np.abs(expected_rate - observed_rate))),
                  "regions": region_results, "limitations": [
                      "This evaluates actual StageRHS rates on one restored physical 3D particle state. It advances neither particles nor FVM and does not establish improved coupled accuracy.",
                      "The original panel strengths are restored from the qualified accepted profile, without resolving the panel boundary problem.",
                      "Velocity-only and native-gradient probes activate the existing full-scope stage hooks temporarily, while keeping panel advancement and shedding disabled.",
                      "The f64 gradient comparison uses refined fourth-order differences at 256 selected fluid particle positions. It does not qualify every stage position or a new production gradient operator.",
                  ]}
    finally:
        solver.physics.body_velocity_field = None
        solver.physics.body_velocity_gradient = None
        solver.close()
    paths = [Path(__file__).resolve(), ROOT / "source/solvers/vpm/core/solver.py",
             ROOT / "source/solvers/vpm/physics/stage_rhs.py", ROOT / "source/solvers/vpm/physics/induction/stretching.py",
             ROOT / "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py"]
    sources += [hash_file(path) for path in paths]
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    for path in paths:
        target = args.output / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    result["sources"] = list({row["path"]: row for row in sources}.values())
    (args.output / "body-stage-snapshot-probe-3d.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "limitations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--backup-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.prefix, args.backup_directory, args.output = (path.resolve() for path in (args.prefix, args.backup_directory, args.output))
    run(args)
