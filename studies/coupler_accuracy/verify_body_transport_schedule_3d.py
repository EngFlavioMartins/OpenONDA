#!/usr/bin/env python3
"""Check actual body-stage calls, held panel strengths and native time schedules."""

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

    record = report(args.run / "body-transport-3d.json")
    assert record["schema"] == "openonda-body-transport-3d/1" and record["spatial_dimensions"] == 3
    assert record["outer_dt"] == .05 and record["fvm_dt"] == .01
    enabled, steps = record["body_transport_enabled"], record["requested_exchanges"]
    for row in record["sources"]:
        archive = args.run / "body-transport-sources" / row["path"]
        assert hash_file(archive)["sha256"] == row["sha256"]
        sources.append(hash_file(archive))
    for row in record["child_reports"]:
        assert hash_file(ROOT / row["path"]) == row
        report(ROOT / row["path"])
    row = record["gradient_qualification"]
    assert hash_file(ROOT / row["path"]) == row
    qualification = report(ROOT / row["path"])
    assert qualification["schema"] == "openonda-analytical-panel-gradient-qualification-3d/1"
    operator_path = "studies/coupler_accuracy/analytical_panel_gradient_3d.py"
    assert next(row for row in qualification["sources"] if row["path"] == operator_path) in record["sources"]
    assert len(record["outer_advances"]) == steps
    stage_rows = []
    for step, row in enumerate(record["outer_advances"], 1):
        assert row["start_step"] == step - 1 and row["accepted_step"] == step
        assert row["accepted_time"] == round(.05 * step, 12)
        assert row["enabled"] == enabled and row["panels"] == 108
        assert len(row["panel_strength_sha256"]) == 64
        assert row["panel_strength_unchanged_during_vpm_advance"] and row["stage_hooks_restored"] and row["schedule_checks_passed"]
        assert row["diffusion_calls"] == [.05]
        assert row["stabilization_phases"] == ["pre_evolution", "pre_strength", "post_evolution", "post_step"]
        np.testing.assert_array_equal([s["time"] for s in row["stages"]], [row["start_time"], row["start_time"] + .05])
        assert [s["index"] for s in row["stages"]] == [0, 1]
        assert row["stages"][0]["count"] == row["stages"][1]["count"] > 0
        for stage in row["stages"]:
            assert stage["body_velocity_calls"] == stage["body_gradient_calls"] == int(enabled)
            assert stage["minimum_cube_clearance"] > 1e-5 and len(stage["position_sha256"]) == 64
            if enabled:
                values = [stage[key] for key in ("body_gradient_frobenius_rms", "body_gradient_maximum_trace", "body_gradient_maximum_antisymmetric_entry")]
                assert np.all(np.isfinite(values)) and np.all(np.asarray(values) >= 0)
            stage_rows.append(stage)
    observer_hash = None
    if args.observer_qualification:
        assert not enabled and steps == 3
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
    result = {"schema": "openonda-body-transport-schedule-verification-3d/1", "status": "complete", "spatial_dimensions": 3,
              "run_directory": str(args.run.relative_to(ROOT)), "accepted_intervals": steps,
              "body_transport_enabled": enabled, "stage_evaluations": 2 * steps,
              "body_velocity_calls": int(enabled) * 2 * steps, "body_gradient_calls": int(enabled) * 2 * steps,
              "gbd_calls": steps, "outer_stabilization_phase_calls": 4 * steps,
              "minimum_stage_cube_clearance": min(row["minimum_cube_clearance"] for row in stage_rows),
              "observer_qualification": observer_hash, "frozen_original_files_verified": len(frozen["records"]),
              "sources": list({row["path"]: row for row in sources}.values()),
              "limitations": [
                  "This checks recorded stage calls, panel-strength hold and original RK/GBD/outer schedules. The separate component qualification validates the gradient; this does not reconstruct every intermediate stage field.",
                  "Observer qualification establishes a disabled wrapper's neutrality when provided. Neither schedule checks nor that control demonstrate improved enabled coupled accuracy."]}
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
