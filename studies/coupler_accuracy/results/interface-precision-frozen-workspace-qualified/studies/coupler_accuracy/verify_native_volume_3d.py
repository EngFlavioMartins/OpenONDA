#!/usr/bin/env python3
"""Recompute recorded volume-comparison metrics and verify source provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "studies/coupler_accuracy/results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def weighted_norm(error, weights):
    return np.sqrt(np.sum(weights[:, None]*np.asarray(error)**2)/np.sum(weights))


def run():
    full_dir = RESULTS / "cube-3d-native-volume-induction"
    part_dir = RESULTS / "cube-3d-partitioned-volume-induction"
    overlap_dir = RESULTS / "cube-3d-volume-overlap"
    full = json.loads((full_dir / "cube-native-volume-induction-3d.json").read_text())
    part = json.loads((part_dir / "cube-partitioned-volume-induction-3d.json").read_text())
    overlap = json.loads((overlap_dir / "cube-volume-overlap-3d.json").read_text())
    assert all(r["status"] == "complete" and r["spatial_dimensions"] == 3 for r in (full, part, overlap))
    sources, current_differences = [], []
    for directory, report in ((full_dir, full), (part_dir, part), (overlap_dir, overlap)):
        assert report["sources"] == json.loads((directory / "sources-at-start.json").read_text())
        for record in report["sources"]:
            path = ROOT / record["path"]
            archived = directory / "sources" / record["path"]
            checked = archived if archived.exists() else path
            assert digest(checked) == record["sha256"], checked
            sources.append({"result": directory.name, "path": record["path"], "verified_against": str(checked.relative_to(ROOT))})
            if digest(path) != record["sha256"]:
                current_differences.append({"result": directory.name, "path": record["path"],
                                            "recorded_sha256": record["sha256"], "current_sha256": digest(path)})
    a = read_arrays(full_dir / "volume-induction-comparison-fields.npz")
    b = read_arrays(part_dir / "partitioned-induction-fields.npz")
    native = read_arrays(full_dir / "native-induction-fields.npz")
    o = read_arrays(overlap_dir / "volume-overlap-fields.npz")
    initial = read_arrays(RESULTS / "cube-3d-oracle/initial-cell-fields.npz")
    boundary = read_arrays(RESULTS / "cube-3d-integrated-velocity-curl-reconstruction-boundary/boundary-fields.npz")
    old_panel = read_arrays(RESULTS / "cube-3d-panel-resolution-108/panel-resolution-fields.npz")
    np.testing.assert_array_equal(a["position"], b["position"])
    np.testing.assert_array_equal(a["position"], native["position"])
    np.testing.assert_array_equal(a["native_cell_centres"], initial["centres"])
    np.testing.assert_array_equal(a["native_cell_velocity"], initial["velocity"])
    assert full["target_slices"] == part["target_slices"]
    selections = {name: slice(*value) for name, value in full["target_slices"].items()}
    np.testing.assert_array_equal(a["position"][selections["wall"]], old_panel["wall_position"])
    np.testing.assert_array_equal(a["position"][selections["boundary"]], boundary["position"])
    metric_differences = []
    for report, fields in ((full, a), (part, b)):
        for row in report["records"]:
            prefix = row["representation"]+"__"+row["completion"]
            velocity = fields[prefix+"__velocity"]
            assert velocity.shape == a["position"].shape and np.all(np.isfinite(velocity))
            for group, recorded in row["cell_velocity_rms_over_Uinf"].items():
                ids = a[group+"__cell_ids"]
                np.testing.assert_array_equal(a["position"][selections[group]], a["native_cell_centres"][ids])
                computed = weighted_norm(velocity[selections[group]]-a["native_cell_velocity"][ids], a["native_cell_volume"][ids])
                metric_differences.append(abs(computed-recorded))
            un = fields[prefix+"__boundary_normal_velocity"]
            computed = weighted_norm((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"])
            metric_differences.append(abs(computed-row["boundary_normal_velocity_rms_error_over_Uinf"]))
            tangent = fields[prefix+"__boundary_native_tangential_gradient"]
            computed = weighted_norm(tangent-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"])
            metric_differences.append(abs(computed-row["boundary_native_tangential_gradient_rms_error"]))
    assert max(metric_differences) < 1e-14
    np.testing.assert_array_equal(a["position"], o["position"][:len(a["position"])])
    np.testing.assert_allclose(o["sharp_at_fvm_boundary__velocity"][:len(a["position"])],
                               b["near_volume_outer_gaussian_seed__panel108__velocity"], rtol=0, atol=1e-12)
    overlap_selections = {name: slice(*value) for name, value in overlap["target_slices"].items()}
    for row in overlap["records"]:
        name = row["name"]
        velocity = o[name+"__velocity"]
        for group, recorded in row["cell_velocity_rms_over_Uinf"].items():
            ids = a[group+"__cell_ids"]
            computed = weighted_norm(velocity[overlap_selections[group]]-a["native_cell_velocity"][ids], a["native_cell_volume"][ids])
            metric_differences.append(abs(computed-recorded))
        un = o[name+"__boundary_normal_velocity"]
        metric_differences.append(abs(weighted_norm((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"])
                                      - row["boundary_normal_velocity_rms_error_over_Uinf"]))
        for side, recorded in row["boundary_native_tangential_gradient_rms_error"].items():
            derivative = o[name+"__gradient_"+side]
            computed = weighted_norm(derivative-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"])
            metric_differences.append(abs(computed-recorded))
    assert max(metric_differences) < 1e-14
    seed = a["gaussian_seed_threshold_0.02__panel108__velocity"]
    old_wall_difference = np.max(np.abs(seed[selections["wall"]]-old_panel["donor_volume_vorticity__wall_velocity"]))
    old_tangent_difference = np.max(np.abs(a["gaussian_seed_threshold_0.02__panel108__boundary_native_tangential_gradient"]
                                          - boundary["donor_volume_vorticity__native_tangential_normal_gradient"]))
    assert old_wall_difference < 1e-12 and old_tangent_difference < 1e-8
    p0 = b["near_volume_outer_gaussian_all__freestream__velocity"]-[1, 0, 0]
    p1 = b["near_gaussian_outer_volume_control__freestream__velocity"]-[1, 0, 0]
    whole = native["velocity"][:, 1]+a["gaussian_sigma_0.125__freestream__velocity"]-[1, 0, 0]
    partition_difference = float(np.max(np.abs(p0+p1-whole)))
    assert partition_difference < 2e-15
    old = next(r for r in full["records"] if r["representation"] == "gaussian_seed_threshold_0.02" and r["completion"] == "panel108")
    candidate = next(r for r in part["records"] if r["representation"] == "near_volume_outer_gaussian_seed" and r["completion"] == "panel108")
    improvement = {key: 100*(1-candidate[key]/old[key]) for key in (
        "boundary_normal_velocity_rms_error_over_Uinf", "boundary_native_tangential_gradient_rms_error",
        "wall_normal_velocity_rms_over_Uinf", "wall_tangential_velocity_rms_over_Uinf")}
    improvement.update({group: 100*(1-candidate["cell_velocity_rms_over_Uinf"][group]/value)
                        for group, value in old["cell_velocity_rms_over_Uinf"].items()})
    regression = ET.parse(RESULTS / "3d-native-volume-regression.xml")
    tests = list(regression.iter("testcase"))
    assert len(tests) == 40 and not list(regression.iter("failure")) and not list(regression.iter("error"))
    before = ET.parse(RESULTS / "source-potential-before.xml")
    assert len(list(before.iter("failure"))) == 3
    artifacts = [Path(__file__), ROOT / "studies/coupler_accuracy/plot_cube_3d_study.py",
                 ROOT / "tests/coupler/test_native_volume_induction_3d.py",
                 ROOT / "tests/vpm/test_source_panel_potential.py",
                 ROOT / "source/solvers/vpm/boundary_elements/panels/kernels/source_potential.py",
                 RESULTS / "source-potential-before.py", RESULTS / "source-potential-before.xml",
                 RESULTS / "3d-native-volume-regression.xml", RESULTS / "cube-3d-native-volume-induction.png",
                 RESULTS / "cube-3d-partitioned-volume-induction.png", RESULTS / "cube-3d-volume-overlap.png"]
    for path in artifacts:
        if path.suffix == ".py":
            destination = RESULTS / "native-volume-verification-sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    result = {"status": "complete", "spatial_dimensions": 3, "source_records_verified": len(sources),
              "source_checks": sources, "current_files_differing_from_recorded_archive": current_differences,
              "post_run_changes": "Import formatting in the full study and potential test; volume/centroid and independent derivative-jump checks were added to the volume test. Numerical source kernels are unchanged after the completed runs began.",
              "metrics_recomputed": len(metric_differences), "maximum_recomputed_metric_difference": max(metric_differences),
              "old_seed_wall_velocity_maximum_difference": float(old_wall_difference),
              "old_seed_native_tangential_derivative_maximum_difference": float(old_tangent_difference),
              "partition_sum_maximum_difference": partition_difference,
              "sharp_small_volume_centred_measurement_relative_improvement_percent": improvement,
              "sharp_derivative_measurement_caveat": "The apparent centred derivative improvement is trace-dependent. The overlap qualification measures both normal-offset estimates and the actual native-boundary jump.",
              "overlap_native_boundary_jump_qualification": overlap["native_boundary_jump_qualification"],
              "focused_tests_passed": len(tests),
              "source_potential_tests_failed_before_fix": 3,
              "verification_artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in artifacts]}
    (RESULTS / "native-volume-verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key not in
                      ("source_checks", "verification_artifacts", "current_files_differing_from_recorded_archive")}, indent=2))


if __name__ == "__main__":
    run()
