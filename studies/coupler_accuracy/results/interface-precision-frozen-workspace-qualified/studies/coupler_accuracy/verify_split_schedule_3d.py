#!/usr/bin/env python3
"""Verify recorded RK/GBD substeps, clocks and every GBD recovery gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.run / "split-subcycling-3d.json"
    record = json.loads(path.read_text())
    assert record["schema"] == "openonda-split-subcycling-3d/1" and record["status"] == "complete"
    assert record["spatial_dimensions"] == 3 and record["outer_dt"] == .05 and record["fvm_dt"] == .01
    n, steps, limit = record["vpm_substeps"], record["requested_exchanges"], record["gbd_correction_limit"]
    assert n in (1, 2, 5) and limit == .08 and record["substep_dt"] == .05 / n
    sources = [hash_file(path), hash_file(Path(__file__).resolve())]
    for row in record["sources"] + record["child_reports"]:
        assert hash_file(ROOT / row["path"]) == row
        sources.append(row)
    for row in record["sources"]:
        archive = args.run / "split-sources" / row["path"]
        assert hash_file(archive)["sha256"] == row["sha256"]
        sources.append(hash_file(archive))
    assert len(record["outer_advances"]) == steps
    recovery_rows = []
    for step, row in enumerate(record["outer_advances"], 1):
        assert row["start_step"] == step - 1 and row["accepted_step"] == step
        assert row["accepted_time"] == round(.05 * step, 12)
        assert row["schedule_checks_passed"] and row["split_preserved_outer_clock"]
        assert row["outer_dt"] == .05 and row["vpm_substeps"] == n and row["substep_dt"] == .05 / n
        assert len(row["rk_calls"]) == len(row["diffusion_calls"]) == n
        assert len(row["stages"]) == 2 * n
        np.testing.assert_allclose([call["time"] for call in row["rk_calls"]], row["start_time"] + np.arange(n) * (.05 / n), rtol=0, atol=2e-16)
        expected_times = [call["time"] + c * call["dt"] for call in row["rk_calls"] for c in (0., 1.)]
        np.testing.assert_array_equal([stage["time"] for stage in row["stages"]], expected_times)
        assert [stage["index"] for stage in row["stages"]] == [0, 1] * n
        assert row["stabilization_phases"] == ["pre_evolution", "pre_strength", "post_evolution", "post_step"]
        assert len(row["substep_projection_corrections"]) == n
        assert row["interval_projection_maximum"] == max(row["substep_projection_corrections"])
        for index, (rk, diffusion) in enumerate(zip(row["rk_calls"], row["diffusion_calls"], strict=True)):
            assert rk["dt"] == diffusion["dt"] == .05 / n and diffusion["substep"] == index
            assert diffusion["accepted_step_before"] == step - 1 and diffusion["recovery_checked"]
            assert diffusion["particles_before"] == rk["count"]
            expected_count = row["particles_at_entry"] if index == 0 else row["diffusion_calls"][index - 1]["particles_after"]
            assert rk["count"] == expected_count
            assert all(stage["count"] == rk["count"] for stage in row["stages"][2 * index:2 * index + 2])
            assert diffusion["laplacian_substeps"] >= 1
            recovery = diffusion["moment_recovery"]
            values = [recovery[key] for key in ("correction_fraction", "normalized_vortex_strength_residual", "normalized_linear_impulse_residual", "normalized_angular_impulse_residual")]
            assert np.all(np.isfinite(values)) and np.all(np.asarray(values) >= 0)
            assert values[0] <= limit and max(values[1:]) <= 1e-5
            assert recovery["pruned_node_count"] == 0 or recovery["applied"]
            recovery_rows.append(values)
        assert row["particles_at_exit"] == row["diffusion_calls"][-1]["particles_after"]
    frozen_path = ROOT / "frozen-workspace.json"
    frozen = json.loads(frozen_path.read_text())
    for row in frozen["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    sources.append(hash_file(frozen_path))
    result = {"schema": "openonda-split-schedule-verification-3d/1", "status": "complete", "spatial_dimensions": 3,
              "run_directory": str(args.run.relative_to(ROOT)), "accepted_intervals": steps, "vpm_substeps": n,
              "rk_calls": steps * n, "rk_stages": steps * n * 2, "gbd_calls": len(recovery_rows),
              "outer_stabilization_phase_calls": steps * 4, "frozen_original_files_verified": len(frozen["records"]),
              "maximum_gbd_correction_fraction": float(np.max(recovery_rows, axis=0)[0]),
              "maximum_reported_gbd_residuals": np.max(recovery_rows, axis=0)[1:].tolist(),
              "sources": sources, "limitations": [
                  "This checks actual recorded calls and their reported GBD diagnostics against the unchanged production limits. It does not independently reconstruct every intermediate diffusion grid or its moments.",
                  "No accuracy or wrapper neutrality claim follows from schedule checks alone. Accepted fields and the unmodified control require separate comparisons."]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "limitations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    run(args)
