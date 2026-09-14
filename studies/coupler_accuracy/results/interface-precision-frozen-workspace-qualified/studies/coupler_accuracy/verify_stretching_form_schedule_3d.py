#!/usr/bin/env python3
"""Verify the selected stretching contraction and unchanged physical cube schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [hash_file(Path(__file__).resolve())]

    def report(path):
        value = json.loads(path.read_text())
        assert value["status"] == "complete"
        sources.append(hash_file(path))
        for row in value.get("sources", []):
            assert hash_file(ROOT / row["path"]) == row
            sources.append(row)
        return value

    record = report(args.run / "stretching-form-trial-3d.json")
    assert record["schema"] == "openonda-stretching-form-trial-3d/1" and record["spatial_dimensions"] == 3
    assert record["outer_dt"] == .05 and record["fvm_dt"] == .01
    assert record["native_stretching_scheme"] == "TRANSPOSED" and record["geometric_separation_factor"] == 3.
    scheme, steps = record["stretching_scheme"], record["requested_exchanges"]
    assert scheme in ("TRANSPOSED", "DIRECT", "MIXED") and record["induction_factory_calls"] >= 1
    assert record["execution_environment"]["TI_CPU_MAX_NUM_THREADS"] == "1"
    for row in record["sources"]:
        archive = args.run / "stretching-form-sources" / row["path"]
        assert hash_file(archive)["sha256"] == row["sha256"]
        sources.append(hash_file(archive))
    for row in record["child_reports"]:
        assert hash_file(ROOT / row["path"]) == row
        report(ROOT / row["path"])
    row = record["consistency_qualification"]
    assert hash_file(ROOT / row["path"]) == row
    qualification = report(ROOT / row["path"])
    assert qualification["schema"] == "openonda-particle-stretching-consistency-3d/1"
    assert qualification["spatial_dimensions"] == 3 and qualification["physical_time"] == 1.5
    assert len(record["outer_advances"]) == steps
    stages = []
    for step, row in enumerate(record["outer_advances"], 1):
        assert row["start_step"] == step - 1 and row["accepted_step"] == step
        assert row["accepted_time"] == round(.05 * step, 12)
        assert row["stretching_scheme"] == scheme and row["schedule_checks_passed"]
        assert row["diffusion_calls"] == [.05]
        assert row["stabilization_phases"] == ["pre_evolution", "pre_strength", "post_evolution", "post_step"]
        np.testing.assert_array_equal([s["time"] for s in row["stages"]], [row["start_time"], row["start_time"] + .05])
        assert [s["index"] for s in row["stages"]] == [0, 1]
        assert row["stages"][0]["count"] == row["stages"][1]["count"] > 0
        for key in ("start_total_strength", "accepted_total_strength_before_fvm_renewal"):
            assert np.asarray(row[key]).shape == (3,) and np.all(np.isfinite(row[key]))
        for stage in row["stages"]:
            assert stage["stretching_scheme"] == scheme and stage["geometric_separation_factor"] == 3.
            assert stage["native_stage_evaluations"] == 1 and stage["outer_clocks_unchanged"]
            assert stage["m2l_pairs"] >= 0 and stage["near_pairs"] > 0
            assert 0 <= stage["p2p_interactions_excluding_self"] <= stage["count"] * (stage["count"] - 1)
            assert stage["contraction_check_passed"] and np.isfinite(stage["contraction_check_maximum_difference"])
            assert stage["contraction_check_maximum_difference"] >= 0
            total, norm_sum = np.asarray(stage["total_strength_rate"]), stage["sum_strength_rate_norms"]
            assert total.shape == (3,) and np.all(np.isfinite(total)) and np.isfinite(norm_sum) and norm_sum >= 0
            assert np.linalg.norm(total) <= norm_sum * (1 + 1e-14)
            relative = float(np.linalg.norm(total)) / max(norm_sum, np.finfo(float).tiny)
            assert stage["normalized_total_strength_rate"] == relative
            stages.append(stage)
    observer_hash = None
    if args.observer_qualification:
        assert scheme == "TRANSPOSED" and steps == 3
        observer = report(args.observer_qualification)
        assert observer["schema"] == "openonda-profile-observer-verification-3d/1"
        assert observer["advancing_comparison_intervals"] == 3 and observer["comparison_histories_bitwise_equal"]
        assert observer["checkpoint_bitwise_equal_counts"] == {"fvm": 17, "vpm_boundary_condition": 11, "vpm_numeric_datasets": 11}
        assert hash_file(args.run / "profile-observation-3d.json") in observer["sources"]
        observer_hash = hash_file(args.observer_qualification)
    frozen_path = ROOT / "frozen-workspace.json"
    frozen = json.loads(frozen_path.read_text())
    for row in frozen["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    sources.append(hash_file(frozen_path))
    result = {"schema": "openonda-stretching-form-schedule-verification-3d/1", "status": "complete", "spatial_dimensions": 3,
              "run_directory": str(args.run.relative_to(ROOT)), "accepted_intervals": steps, "stretching_scheme": scheme,
              "geometric_separation_factor": 3., "stage_evaluations": len(stages), "gbd_calls": steps,
              "outer_stabilization_phase_calls": 4 * steps, "observer_qualification": observer_hash,
              "maximum_stage_contraction_difference": max(row["contraction_check_maximum_difference"] for row in stages),
              "maximum_normalized_total_strength_rate": max(row["normalized_total_strength_rate"] for row in stages),
              "frozen_original_files_verified": len(frozen["records"]),
              "sources": list({row["path"]: row for row in sources}.values()), "limitations": [
                  "This verifies recorded native stage calls, contraction checks and schedules. It does not independently reconstruct every intermediate Jacobian or total-strength rate.",
                  "The transposed observer control establishes neutrality when provided. Improved coupled accuracy and the conservation tradeoff of a different formulation require their separate force/profile comparison."]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "limitations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--observer-qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if args.observer_qualification:
        args.observer_qualification = args.observer_qualification.resolve()
    run(args)
