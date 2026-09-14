#!/usr/bin/env python3
"""Verify matched advancing controls and the sampled boundary's actual inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.verify_sampled_native_face_3d import independent_gradient

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "studies/coupler_accuracy/results"


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(value, weight):
    return float(np.sqrt(np.average(np.sum(value**2, axis=1), weights=weight)))


def independent_cube_force(state, mesh, geo):
    """Rebuild the Gauss gradient and integrate wall pressure/deviatoric stress."""
    velocity, pressure = state["velocity"], state["kinematic_pressure"]
    n, ni = mesh["n_cells"], mesh["n_interior_faces"]
    owner, neighbour = mesh["owners"], mesh["neighbours"]
    w = geo["face_interpolation_weight"][:ni]
    face_velocity = velocity[owner[:ni]]*(1-w[:, None])+velocity[neighbour]*w[:, None]
    contribution = geo["face_area_vector"][:ni, :, None]*face_velocity[:, None, :]
    gradient = np.zeros((n, 3, 3))
    np.add.at(gradient, owner[:ni], contribution)
    np.add.at(gradient, neighbour, -contribution)
    np.add.at(gradient, owner[ni:], geo["face_area_vector"][ni:, :, None]*velocity[n:, None, :])
    gradient /= geo["cell_volume"][:, None, None]
    wall = next(patch for patch in mesh["boundary"] if patch["name"] == "cube")
    faces = np.arange(wall["start_face"], wall["start_face"]+wall["n_faces"])
    ghost = n+faces-ni
    np.testing.assert_array_equal(velocity[ghost], 0)
    sf = geo["face_area_vector"][faces]
    area = np.linalg.norm(sf, axis=1)
    normal = sf/area[:, None]
    wall_gradient = gradient[owner[faces]].copy()
    jump = (velocity[ghost]-velocity[owner[faces]])/geo["wall_distance"][faces, None]
    wall_gradient += normal[:, :, None]*(jump-np.einsum("nd,ndc->nc", normal, wall_gradient))[:, None, :]
    stress = .001*(wall_gradient+wall_gradient.swapaxes(1, 2)
                    -(2/3)*np.trace(wall_gradient, axis1=1, axis2=2)[:, None, None]*np.eye(3)[None])
    pressure_force = np.sum(pressure[ghost, None]*sf, axis=0)
    viscous_force = -np.einsum("ndc,nd->c", stress, sf)
    return {"pressure_force": pressure_force.tolist(), "viscous_force": viscous_force.tolist(),
            "pressure_drag_coefficient": float(2*pressure_force[0]), "viscous_drag_coefficient": float(2*viscous_force[0]),
            "drag_coefficient": float(2*(pressure_force[0]+viscous_force[0]))}


def run():
    kinds = ("control", "gradient", "both")
    directories = {kind: RESULTS / ("cube-3d-medium-laminar-sampled-face-"+kind) for kind in kinds}
    trial_dirs = {kind: directory if kind == "control" else directory / "trial" for kind, directory in directories.items()}
    reports = {kind: json.loads((directory / "cube-coupled-trial.json").read_text()) for kind, directory in trial_dirs.items()}
    source_count, metric_diff, replay_diff = 0, [], []

    def verify_sources(report, directory):
        nonlocal source_count
        for row in report["sources"]:
            path = ROOT / row["path"]
            archive = directory / "sources" / row["path"]
            assert digest(path) == row["sha256"]
            assert digest(archive if archive.exists() else path) == row["sha256"]
            source_count += 1

    def check(actual, claimed, tolerance=2e-13):
        difference = abs(actual-claimed)
        metric_diff.append(difference)
        assert difference <= tolerance, (actual, claimed, tolerance)

    oracle = RESULTS / "cube-3d-medium-laminar-oracle"
    mesh = load_native_mesh(oracle / "full-native-mesh.npz")
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    small = load_native_mesh(oracle / "small-native-mesh.npz")
    small_geo = compute_mesh_geometry(small, gradient_scheme="gauss", compute_lsq=False)
    mapping = arrays(oracle / "cell-and-face-map.npz")
    np.testing.assert_allclose(small_geo["cell_centre"], geo["cell_centre"][mapping["cell_ids"]], rtol=0, atol=1e-13)
    np.testing.assert_allclose(small_geo["cell_volume"], geo["cell_volume"][mapping["cell_ids"]], rtol=1e-13, atol=1e-14)
    near = np.max(np.abs(small_geo["cell_centre"]), axis=1) < 1
    control = reports["control"]
    fields = {kind: arrays(directory / "latest-comparison-fields.npz") for kind, directory in trial_dirs.items()}
    final, histories, forces, backup_records = {}, {}, {}, []
    for kind, report in reports.items():
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        assert report["identical_native_shared_cells"]
        assert report["full_cells"] == 53752 and report["small_cells"] == 16936
        assert report["requested_wall_spacing"] == report["particle_spacing"] == .0625
        assert report["fvm_dt"] == .01 and report["vpm_dt"] == .05
        assert report["fvm_numerics"]["sgs"] == "none" and report["requested_coupling_steps"] == 20
        assert report["transfer_method"] == "buffered_m4_renewal" and report["mixed_boundary_convection"] == "native"
        assert report["boundary_mode"] == "vorticity_mixed" and not report["experimental_residual_blend"]
        assert report["sources"] == control["sources"]
        verify_sources(report, trial_dirs[kind])
        history = report["comparison"]
        assert history == json.loads((trial_dirs[kind] / "comparison-history.json").read_text())
        assert len(history) == 21 and [row["coupling_step"] for row in history] == list(range(21))
        assert history[0] == control["comparison"][0]
        np.testing.assert_array_equal([row["physical_time"] for row in history], [row["physical_time"] for row in control["comparison"]])
        np.testing.assert_array_equal([row["full_drag_coefficient"] for row in history], [row["full_drag_coefficient"] for row in control["comparison"]])
        np.testing.assert_array_equal(fields[kind]["full_velocity"], fields["control"]["full_velocity"])
        np.testing.assert_array_equal(fields[kind]["position"], fields["control"]["position"])
        np.testing.assert_array_equal(fields[kind]["vpm_query_ids"], fields["control"]["vpm_query_ids"])
        for row in history:
            check(row["hybrid_drag_coefficient"]-row["full_drag_coefficient"], row["drag_coefficient_difference"])
        last, field = history[-1], fields[kind]
        assert last["physical_time"] == 1.5 and field["elapsed_time"] == 1.
        error = field["hybrid_velocity"]-field["full_velocity"]
        check(rms(error, small_geo["cell_volume"]), last["fvm_velocity_rms_over_Uinf"])
        check(rms(error[near], small_geo["cell_volume"][near]), last["fvm_near_body_velocity_rms_over_Uinf"])
        ids = field["vpm_query_ids"]
        check(rms(field["vpm_velocity"]-field["full_velocity"][ids], small_geo["cell_volume"][ids]), last["vpm_sampled_velocity_rms_over_Uinf"])
        backup_path = trial_dirs[kind] / "hybrid/solution/backups/fvm_000020.npz"
        backup = decode_state(arrays(backup_path))
        np.testing.assert_array_equal(backup["velocity"][:small["n_cells"]], field["hybrid_velocity"])
        assert int(backup["step"]) == 100 and len(backup["eddy_viscosity"]) == 0
        forces[kind] = independent_cube_force(backup, small, small_geo)
        check(forces[kind]["drag_coefficient"], last["hybrid_drag_coefficient"])
        backup_records.append({"path": str(backup_path.relative_to(ROOT)), "sha256": digest(backup_path)})
        final[kind] = {**last, "relative_drag_error_percent": 100*last["drag_coefficient_difference"]/last["full_drag_coefficient"],
                       "advancing_drag_history_absolute_error_rms": float(np.sqrt(np.mean([row["drag_coefficient_difference"]**2 for row in history[1:]]))),
                       "elapsed_seconds": report["wall_time_seconds"]}
        histories[kind] = history
        if kind == "control":
            continue
        directory = directories[kind]
        adapter = json.loads((directory / "sampled-boundary-trial.json").read_text())
        assert adapter["status"] == "complete" and adapter["mode"] == "native_"+kind
        verify_sources(adapter, directory)
        assert adapter["boundary_faces"] == 3456 and adapter["sample_cells"] == 13024 and adapter["gradient_cells"] == 6632
        calls = adapter["boundary_calls"]
        assert len(calls) > 20 and calls[0]["step"] == 0 and calls[-1]["step"] == 20
        assert calls[0]["elapsed_time"] == 0 and calls[-1]["elapsed_time"] == 1
        sample = arrays(directory / "boundary-sampling-geometry.npz")
        np.testing.assert_array_equal(sample["sample_position"], geo["cell_centre"][sample["sample_cells"]])
        np.testing.assert_array_equal(sample["face_position"], geo["face_centre"][sample["full_face_ids"]])
        selected = np.zeros(mesh["n_cells"], dtype=bool)
        selected[sample["gradient_cells"]] = True
        ni = mesh["n_interior_faces"]
        assert not np.any(selected[mesh["owners"][ni:]])
        sample["gradient_faces"] = np.flatnonzero(selected[mesh["owners"][:ni]] | selected[mesh["neighbours"][:ni]])
        for name, call in (("initial", calls[0]), ("latest", calls[-1])):
            saved = arrays(directory / (name+"-boundary-observation.npz"))
            gradient, derivative, face = independent_gradient(mesh, geo, sample, saved["sampled_velocity"])
            replay_diff += [float(np.max(np.abs(gradient-saved["cell_gradient"]))), float(np.max(np.abs(derivative-saved["normal_gradient"])))]
            np.testing.assert_allclose(gradient, saved["cell_gradient"], rtol=0, atol=2e-13)
            np.testing.assert_allclose(derivative, saved["normal_gradient"], rtol=0, atol=2e-13)
            np.testing.assert_allclose(face, saved["face_velocity"], rtol=0, atol=2e-15)
            if kind == "both":
                np.testing.assert_array_equal(saved["returned_velocity"], saved["face_velocity"])
            check(rms(saved["sampled_velocity"], None), call["sample_velocity_rms"])
            check(rms(saved["returned_velocity"], None), call["face_velocity_rms"])
            check(rms(saved["tangential_gradient"], sample["face_area"]), call["tangential_gradient_rms"])
            assert saved["step"] == call["step"] and saved["elapsed_time"] == call["elapsed_time"]
        final[kind]["boundary_queries"] = len(calls)
    old = json.loads((RESULTS / "cube-3d-medium-laminar-mixed-state-baseline/cube-coupled-trial.json").read_text())
    old_difference = {key: max(abs(a[key]-b[key]) for a, b in zip(control["comparison"], old["comparison"], strict=True))
                      for key in ("full_drag_coefficient", "hybrid_drag_coefficient", "fvm_velocity_rms_over_Uinf", "vpm_sampled_velocity_rms_over_Uinf")}
    test_files = [RESULTS / name for name in ("3d-face-velocity-sampling-regression.xml", "3d-sampled-face-adapter-regression.xml")]
    test_count = 0
    for path in test_files:
        tests = ET.parse(path).getroot().findall(".//testcase")
        assert not any(test.find(tag) is not None for test in tests for tag in ("failure", "error", "skipped"))
        test_count += len(tests)
    assert test_count == 7
    labels = {"control": "Current continuous trace", "gradient": "Native derivative, point velocity", "both": "Native derivative and face velocity"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    time = [row["physical_time"] for row in histories["control"]]
    axes[0, 0].plot(time, [row["full_drag_coefficient"] for row in histories["control"]], color="black", label="Full FVM reference", linewidth=2)
    metrics = ("hybrid_drag_coefficient", "fvm_near_body_velocity_rms_over_Uinf", "fvm_velocity_rms_over_Uinf", "vpm_sampled_velocity_rms_over_Uinf")
    ylabels = ("Drag coefficient", "Near-body FVM velocity RMS error / U∞", "Whole small-FVM velocity RMS error / U∞", "VPM velocity RMS error / U∞")
    for kind in kinds:
        for ax, metric in zip(axes.ravel(), metrics, strict=True):
            ax.plot(time, [row[metric] for row in histories[kind]], label=labels[kind])
    for ax, label in zip(axes.ravel(), ylabels, strict=True):
        ax.set_xlabel("Physical time [D/U∞]")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Advancing fully 3D medium cube: native boundary sampling\nSame small FVM mesh, initial state and particle transfer", fontsize=14)
    figure = RESULTS / "cube-3d-sampled-face-live.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    result = {"status": "complete", "source_records_checked": source_count, "metrics_recomputed": len(metric_diff),
              "maximum_metric_difference": max(metric_diff), "independent_boundary_stencil_replay_maximum_difference": max(replay_diff),
              "tests_passed": test_count, "reference_histories_and_final_velocities_bitwise_equal": True,
              "initial_comparisons_identical": True, "old_baseline_maximum_historical_difference": old_difference,
              "final": final, "independent_final_hybrid_wall_forces": forces, "force_checkpoint_sources": backup_records,
              "figure": {"path": str(figure.relative_to(ROOT)), "sha256": digest(figure)}}
    archive = RESULTS / "sampled-face-trials-verification-sources" / Path(__file__).relative_to(ROOT)
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, archive)
    result["verifier"] = {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__))}
    result["verifier_dependencies"] = []
    for name in ("studies/coupler_accuracy/verify_sampled_native_face_3d.py", "source/solvers/fvm/io/backup.py"):
        path = ROOT / name
        target = RESULTS / "sampled-face-trials-verification-sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        result["verifier_dependencies"].append({"path": name, "sha256": digest(path)})
    result["tests"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in test_files]
    (RESULTS / "sampled-face-trials-verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
