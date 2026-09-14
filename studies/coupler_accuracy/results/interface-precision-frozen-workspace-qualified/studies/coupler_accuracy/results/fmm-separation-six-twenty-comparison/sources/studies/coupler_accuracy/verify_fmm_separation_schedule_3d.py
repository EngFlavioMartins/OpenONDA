#!/usr/bin/env python3
"""Verify the changed FMM factor and unchanged physical coupling schedule."""

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

    record = report(args.run / "fmm-separation-trial-3d.json")
    assert record["schema"] == "openonda-fmm-separation-trial-3d/1" and record["spatial_dimensions"] == 3
    assert record["outer_dt"] == .05 and record["fvm_dt"] == .01 and record["native_factor"] == 3.
    factor, steps = record["geometric_separation_factor"], record["requested_exchanges"]
    assert factor in (3., 4.5, 6.)
    assert record["execution_environment"]["TI_CPU_MAX_NUM_THREADS"] == "1"
    for row in record["sources"]:
        archive = args.run / "fmm-separation-sources" / row["path"]
        assert hash_file(archive)["sha256"] == row["sha256"]
        sources.append(hash_file(archive))
    for row in record["child_reports"]:
        assert hash_file(ROOT / row["path"]) == row
        report(ROOT / row["path"])
    assert len(record["stage_qualifications"]) == 2
    for row, expected in zip(record["stage_qualifications"], (factor, 3.), strict=True):
        assert hash_file(ROOT / row["path"]) == row
        qualification = report(ROOT / row["path"])
        assert qualification["schema"] == "openonda-particle-stage-separation-probe-3d/1"
        assert qualification["geometric_separation_factor"] == expected
        assert qualification["replayed_stage_bitwise_equal"] and qualification["primary_fields_and_clocks_unchanged"]
        if expected == 3.:
            assert qualification["native_control_bitwise_equal"]
        operator = next(row for row in qualification["sources"] if row["path"] == "source/solvers/vpm/physics/induction/fmm/device.py")
        assert operator in record["sources"]
    assert len(record["outer_advances"]) == steps
    stages = []
    for step, row in enumerate(record["outer_advances"], 1):
        assert row["start_step"] == step - 1 and row["accepted_step"] == step
        assert row["accepted_time"] == round(.05 * step, 12)
        assert row["geometric_separation_factor"] == factor and row["schedule_checks_passed"]
        assert row["diffusion_calls"] == [.05]
        assert row["stabilization_phases"] == ["pre_evolution", "pre_strength", "post_evolution", "post_step"]
        np.testing.assert_array_equal([s["time"] for s in row["stages"]], [row["start_time"], row["start_time"] + .05])
        assert [s["index"] for s in row["stages"]] == [0, 1]
        assert row["stages"][0]["count"] == row["stages"][1]["count"] > 0
        for stage in row["stages"]:
            assert stage["native_stage_evaluations"] == 1 and stage["outer_clocks_unchanged"]
            assert stage["m2l_pairs"] >= 0 and stage["near_pairs"] > 0
            assert 0 <= stage["p2p_interactions_excluding_self"] <= stage["count"] * (stage["count"] - 1)
            stages.append(stage)
    observer_hash = None
    if args.observer_qualification:
        assert factor == 3. and steps == 3
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
    result = {"schema": "openonda-fmm-separation-schedule-verification-3d/1", "status": "complete", "spatial_dimensions": 3,
              "run_directory": str(args.run.relative_to(ROOT)), "accepted_intervals": steps,
              "geometric_separation_factor": factor, "stage_evaluations": len(stages),
              "gbd_calls": steps, "outer_stabilization_phase_calls": 4 * steps,
              "recorded_p2p_interactions_excluding_self": sum(stage["p2p_interactions_excluding_self"] for stage in stages),
              "observer_qualification": observer_hash, "frozen_original_files_verified": len(frozen["records"]),
              "sources": list({row["path"]: row for row in sources}.values()),
              "limitations": [
                  "This verifies recorded native FMM calls and physical schedules. The separate saved-state probes quantify stage accuracy; this does not independently rebuild all intermediate interaction lists.",
                  "The default-factor observer control establishes neutrality when provided. Neither schedule checks nor that control establish improved coupled forces or profiles for a stricter factor."]}
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
