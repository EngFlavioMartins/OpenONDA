#!/usr/bin/env python3
"""Observe every unchanged correction trial and retain its rejected 3D fields."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.vpm.stabilization import divergence_relaxation as native
from studies.coupler_accuracy import particle_divergence_correction_probe_3d as probe
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.particle_stage_induction_audit_3d import direct_gaussian, rms
from studies.coupler_accuracy.particle_stretching_consistency_3d import curl, gaussian_vorticity


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    previous = json.loads(args.baseline_probe.read_text())
    assert previous["schema"] == "openonda-particle-divergence-correction-probe-3d/1"
    assert previous["status"] == "complete" and not previous["proposal_accepted"]
    assert previous["operator_settings"]["max_projection_sweeps"] == 3
    consistency = json.loads(args.consistency.read_text())
    assert consistency["status"] == "complete" and consistency["spatial_dimensions"] == 3
    sources = [hash_file(args.baseline_probe), hash_file(args.consistency),
               *previous["sources"], *consistency["sources"], hash_file(Path(__file__).resolve())]
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    args.output.mkdir(parents=True)
    report_path = args.output / "particle-divergence-gate-audit-3d.json"
    result = {"schema": "openonda-particle-divergence-gate-audit-3d/1", "status": "running",
              "spatial_dimensions": 3, "physical_time": 1.5, "attempts": [], "candidate_fields": [],
              "limitations": [
                  "This observes the original three-sweep correction without changing its arithmetic, arguments, gates, returns or exceptions. No coupled solver advances.",
                  "All recorded candidates remain rejected diagnostic fields. Their f64 field changes are not accepted particle uploads or improvements against full-FVM forces or profiles.",
                  "The physical-field audit uses the same 256 selected targets as the qualified consistency audit. It does not infer global physical accuracy from those samples.",
                  "The final rejection is compared with the prior uninstrumented probe. Internal candidate arrays were not retained by that earlier probe, so their cross-run bitwise equivalence is not claimed."]}
    original = native._constrained_divergence_relaxation_once
    signature = inspect.signature(original)
    candidate_records = {}

    def write_report():
        report_path.write_text(json.dumps(result, indent=2) + "\n")

    def observed(*positional, **keywords):
        arguments = signature.bind(*positional, **keywords)
        arguments.apply_defaults()
        inputs = [arguments.arguments[key] for key in ("position", "vortex_strength", "core_radius", "particle_volume")]
        before = probe.signature(inputs)
        attempt = {"index": len(result["attempts"]), "input_array_sha256": before,
                   "correction_scale": arguments.arguments["correction_scale"],
                   "restoration_scale": arguments.arguments["restoration_scale"],
                   "required_residual_ratio": arguments.arguments["max_residual_ratio"]}
        try:
            answer = original(*positional, **keywords)
            diagnostics = asdict(answer)
            del diagnostics["vortex_strength"], diagnostics["correction"]
            attempt.update(accepted=True, diagnostics=diagnostics)
            return answer
        except native.DivergenceRelaxationError as error:
            attempt.update(accepted=False, rejection={"gate": error.gate, "message": str(error)})
            trace = error.__traceback__
            while trace is not None and trace.tb_frame.f_code is not original.__code__:
                trace = trace.tb_next
            assert trace is not None
            values = dict(trace.tb_frame.f_locals)
            attempt["evaluated_gates"] = [
                {"name": name, "value": float(value), "limit": float(limit),
                 "passes": bool(np.isfinite(value) and value <= limit)}
                for name, value, limit in values.get("gates", ())]
            for key in ("iterations", "trust_region_scale", "initial_residual_norm", "final_residual_ratio",
                        "correction_norm_relative", "quadratic_restoration_fraction", "grid_divergence_before",
                        "relaxed_grid_divergence", "total_kinetic_energy_change_relative",
                        "total_enstrophy_change_relative", "total_helicity_change_relative", "variation_change_relative"):
                if key in values:
                    attempt[key] = float(values[key])
            for key in ("before_integrals", "candidate_integrals", "after_integrals"):
                if key in values:
                    attempt[key] = asdict(values[key])
            if "gates" in values:
                failed = [gate for gate in attempt["evaluated_gates"] if not gate["passes"]]
                assert failed and failed[0]["name"] == error.gate
                arrays = {key: values[key] for key in (
                    "position", "vortex_strength", "core_radius", "particle_volume", "relaxed",
                    "correction", "raw_correction", "invariant_correction", "residual", "relaxed_residual")}
                np.testing.assert_array_equal(arrays["raw_correction"] + arrays["invariant_correction"], arrays["correction"])
                np.testing.assert_array_equal(arrays["vortex_strength"] + arrays["correction"], arrays["relaxed"])
                assert np.linalg.norm(arrays["correction"]) / np.linalg.norm(arrays["vortex_strength"]) == values["correction_norm_relative"]
                assert np.linalg.norm(arrays["relaxed_residual"]) / np.linalg.norm(arrays["residual"]) == values["final_residual_ratio"]
                digest = hashlib.sha256(b"".join(value.tobytes() for value in arrays.values())).hexdigest()
                if digest not in candidate_records:
                    path = args.output / f"rejected-candidate-{len(candidate_records):02d}.npz"
                    np.savez_compressed(path, **arrays)
                    candidate_records[digest] = hash_file(path)
                    result["candidate_fields"].append(candidate_records[digest])
                attempt["fields"] = candidate_records[digest]
            del values, trace
            raise
        finally:
            assert probe.signature(inputs) == before
            attempt["input_arrays_unchanged"] = True
            result["attempts"].append(attempt)
            write_report()

    started = time.perf_counter()
    write_report()
    try:
        native._constrained_divergence_relaxation_once = observed
        try:
            probe.run(argparse.Namespace(consistency=args.consistency, output=args.output / "observed-probe"))
        finally:
            native._constrained_divergence_relaxation_once = original
        child_path = args.output / "observed-probe/particle-divergence-correction-probe-3d.json"
        child = json.loads(child_path.read_text())
        for key in ("schema", "status", "spatial_dimensions", "physical_time", "particle_count", "operator_settings",
                    "proposal_accepted", "rejection", "input_particle_arrays_unchanged", "frozen_original_files_verified"):
            assert child[key] == previous[key], key
        assert result["attempts"] and all(not row["accepted"] for row in result["attempts"])
        sources += [hash_file(child_path), *child["sources"], *result["candidate_fields"]]
        result.update(unchanged_final_rejection=child["rejection"], prior_probe_outcome_reproduced=True,
                      settings_unchanged=True, input_arrays_unchanged=True)
        saved_path = next(ROOT / row["path"] for row in consistency["sources"]
                          if row["path"].endswith("stretching-consistency-fields.npz"))
        selected = read_arrays(saved_path)
        indices, points = selected["indices"], selected["position"]
        first = read_arrays(ROOT / result["candidate_fields"][0]["path"])
        position, strength, radius = (first[key] for key in ("position", "vortex_strength", "core_radius"))
        np.testing.assert_array_equal(position[indices], points)
        np.testing.assert_array_equal(strength[indices], selected["target_strength"])
        assert np.all(radius == .0625)
        old_velocity, old_gradient = direct_gaussian(points, radius[indices], position, strength, radius)
        old_blob, old_blob_gradient = gaussian_vorticity(points, position, strength, radius)
        region_masks = {"near_body": np.max(np.abs(points), axis=1) < 1, "near_wake": points[:, 0] > 1.5}

        def metrics(velocity, gradient, blob, blob_gradient, local_strength):
            difference = blob - curl(gradient)
            direct_rate = np.einsum("nij,nj->ni", gradient, local_strength)
            transposed_rate = np.einsum("nji,nj->ni", gradient, local_strength)
            regions = {}
            for name, mask in region_masks.items():
                assert mask.sum() == 128
                regions[name] = {
                    "velocity_change_rms_over_Uinf": rms((velocity - old_velocity)[mask]),
                    "blob_minus_velocity_curl_relative_to_curl": rms(difference[mask]) / rms(curl(gradient)[mask]),
                    "blob_divergence_rms_over_gradient_rms": float(np.sqrt(np.mean(np.trace(blob_gradient[mask], axis1=1, axis2=2)**2))) / rms(blob_gradient[mask]),
                    "direct_minus_transposed_rate_relative_to_transposed": rms((direct_rate - transposed_rate)[mask]) / rms(transposed_rate[mask]),
                }
            return regions

        result["baseline_regions"] = metrics(old_velocity, old_gradient, old_blob, old_blob_gradient, strength[indices])
        repeated_differences = []
        for name, region in result["baseline_regions"].items():
            for key in ("blob_minus_velocity_curl_relative_to_curl", "blob_divergence_rms_over_gradient_rms",
                        "direct_minus_transposed_rate_relative_to_transposed"):
                difference = abs(region[key] - consistency["regions"][name][key])
                assert difference < 2e-13
                repeated_differences.append(difference)
        result["repeated_consistency_metric_maximum_difference"] = max(repeated_differences)
        result["rejected_candidate_regions"] = []
        for row in result["candidate_fields"]:
            arrays = read_arrays(ROOT / row["path"])
            for key, original_values in (("position", position), ("vortex_strength", strength), ("core_radius", radius)):
                np.testing.assert_array_equal(arrays[key], original_values)
            value = arrays["relaxed"]
            velocity, gradient = direct_gaussian(points, radius[indices], position, value, radius)
            blob, blob_gradient = gaussian_vorticity(points, position, value, radius)
            field_path = args.output / (Path(row["path"]).stem + "-selected-fields.npz")
            np.savez_compressed(field_path, position=points, velocity=velocity, gradient=gradient,
                                blob=blob, blob_gradient=blob_gradient, target_strength=value[indices])
            sources.append(hash_file(field_path))
            result["rejected_candidate_regions"].append({"fields": row, "selected_fields": hash_file(field_path),
                                                        "regions": metrics(velocity, gradient, blob, blob_gradient, value[indices])})
        frozen_path = ROOT / "frozen-workspace.json"
        frozen = json.loads(frozen_path.read_text())
        for row in frozen["records"]:
            assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
        result["frozen_original_files_verified"] = len(frozen["records"])
        unique = {row["path"]: row for row in sources}
        for row in unique.values():
            path = ROOT / row["path"]
            assert hash_file(path) == row
            if path.suffix == ".py":
                archive = args.output / "sources" / row["path"]
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(path.read_bytes())
        result.update(status="complete", sources=list(unique.values()))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_report()
    print(json.dumps({"status": result["status"], "attempts": len(result["attempts"]),
                      "unique_rejected_fields": len(result["candidate_fields"]),
                      "unchanged_final_rejection": result["unchanged_final_rejection"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consistency", type=Path, required=True)
    parser.add_argument("--baseline-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("consistency", "baseline_probe", "output"):
        setattr(args, key, getattr(args, key).resolve())
    run(args)
