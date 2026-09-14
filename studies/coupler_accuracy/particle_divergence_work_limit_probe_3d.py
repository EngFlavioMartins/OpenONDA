#!/usr/bin/env python3
"""Increase only the sweep budget of the guarded correction on the physical 3D cube."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time

import h5py
import numpy as np

from source.solvers.vpm.stabilization.divergence_relaxation import (
    DivergenceRelaxationError,
    constrained_divergence_relaxation,
)
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.particle_stage_induction_audit_3d import direct_gaussian, rms
from studies.coupler_accuracy.particle_stretching_consistency_3d import curl, gaussian_vorticity


def signature(arrays):
    return [hashlib.sha256(value.tobytes()).hexdigest() for value in arrays]


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    consistency = json.loads(args.consistency.read_text())
    assert consistency["schema"] == "openonda-particle-stretching-consistency-3d/1" and consistency["status"] == "complete"
    assert consistency["physical_time"] == 1.5 and consistency["spatial_dimensions"] == 3
    sources = [hash_file(args.consistency), *consistency["sources"]]
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    backup = next(row for row in sources if row["path"].endswith("vpm_000020.h5"))
    saved_row = next(row for row in sources if row["path"].endswith("stretching-consistency-fields.npz"))
    prefix_row = next(row for row in sources if row["path"].endswith("long-wake-prefix-verification-3d.json"))
    prefix = json.loads((ROOT / prefix_row["path"]).read_text())
    frame = next(row for row in prefix["profile_frames"] if row["coupling_step"] == 20)
    assert hash_file(ROOT / frame["profile_fields"]["path"]) == frame["profile_fields"]
    sources.append(frame["profile_fields"])
    profile = read_arrays(ROOT / frame["profile_fields"]["path"])
    saved = read_arrays(ROOT / saved_row["path"])
    with h5py.File(ROOT / backup["path"], "r") as handle:
        arrays = [handle["particles/" + key][:].astype(float) for key in ("position", "vortex_strength", "core_radius", "particle_volume")]
    position, strength, radius, volume = arrays
    assert len(position) == 28441 and np.all(radius == .0625) and np.all(volume > 0)
    for array, key in zip(arrays[:3], ("particle_position", "particle_vortex_strength", "particle_core_radius"), strict=True):
        np.testing.assert_array_equal(array, profile[key])
    np.testing.assert_array_equal(position[saved["indices"]], saved["position"])
    np.testing.assert_array_equal(strength[saved["indices"]], saved["target_strength"])
    before = signature(arrays)
    settings = {name: value.default for name, value in inspect.signature(constrained_divergence_relaxation).parameters.items()
                if value.default is not inspect.Parameter.empty}
    settings["grid_spacing"] = .0625
    settings["max_projection_sweeps"] = args.max_projection_sweeps
    own_paths = [Path(__file__).resolve(), ROOT / "studies/coupler_accuracy/particle_stretching_consistency_3d.py",
                 ROOT / "studies/coupler_accuracy/particle_stage_induction_audit_3d.py",
                 ROOT / "source/solvers/vpm/stabilization/divergence_relaxation.py",
                 ROOT / "source/solvers/vpm/stabilization/filament_refinement.py",
                 ROOT / "source/solvers/vpm/numerics/fourier_integrals.py"]
    sources += [hash_file(path) for path in own_paths]
    args.output.mkdir(parents=True)
    for path in own_paths:
        archive = args.output / "sources" / path.relative_to(ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
    report_path = args.output / "particle-divergence-correction-probe-3d.json"
    result = {"schema": "openonda-particle-divergence-correction-probe-3d/2", "status": "running", "spatial_dimensions": 3,
              "physical_time": 1.5, "particle_count": len(position), "operator_settings": settings,
              "stage": "production correction proposal", "sources": list({row["path"]: row for row in sources}.values()),
              "limitations": [
                  "The existing host correction is tested with its unchanged final acceptance gates and grid spacing 0.0625; only the maximum projection-sweep budget increases. It preserves this snapshot's moments; this does not reproduce the stabilization manager's separate initial-reference history.",
                  "No solver state advances. A rejected proposal is a result of this component experiment, not a failed or restarted coupled simulation.",
                  "If accepted, selected direct fields are reevaluated after the production f32 upload conversion. These field changes alone do not establish improved FVM forces or profiles."]}
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    started = time.perf_counter()
    try:
        try:
            proposal = constrained_divergence_relaxation(position, strength, radius, volume, grid_spacing=.0625,
                                                          max_projection_sweeps=args.max_projection_sweeps)
        except DivergenceRelaxationError as error:
            result.update(proposal_accepted=False, rejection={"gate": error.gate, "message": str(error)})
        else:
            result["proposal_accepted"] = True
            proposal_data = asdict(proposal)
            del proposal_data["vortex_strength"], proposal_data["correction"]
            result["operator_diagnostics"] = proposal_data
            result["stage"] = "independent fields after f32 upload"
            report_path.write_text(json.dumps(result, indent=2) + "\n")
            uploaded = proposal.vortex_strength.astype(np.float32).astype(float)
            points, indices = saved["position"], saved["indices"]
            velocity, gradient = direct_gaussian(points, radius[indices], position, uploaded, radius)
            old_velocity = direct_gaussian(points, radius[indices], position, strength, radius)[0]
            blob, blob_gradient = gaussian_vorticity(points, position, uploaded, radius)
            velocity_curl = curl(gradient)
            direct = np.einsum("nij,nj->ni", gradient, uploaded[indices])
            transposed = np.einsum("nji,nj->ni", gradient, uploaded[indices])
            regions = {"near_body": np.max(np.abs(points), axis=1) < 1, "near_wake": points[:, 0] > 1.5}
            result["uploaded_regions"] = {}
            for name, mask in regions.items():
                assert mask.sum() == 128
                divergence = np.trace(blob_gradient[mask], axis1=1, axis2=2)
                result["uploaded_regions"][name] = {
                    "particle_velocity_change_rms_over_Uinf": rms((velocity - old_velocity)[mask]),
                    "blob_minus_velocity_curl_relative_to_curl": rms((blob - velocity_curl)[mask]) / rms(velocity_curl[mask]),
                    "blob_divergence_rms_over_gradient_rms": float(np.sqrt(np.mean(divergence**2))) / rms(blob_gradient[mask]),
                    "direct_minus_transposed_rate_relative_to_transposed": rms((direct - transposed)[mask]) / rms(transposed[mask]),
                }
            output_fields = args.output / "accepted-correction-fields.npz"
            np.savez_compressed(output_fields, position=position, vortex_strength_f64=proposal.vortex_strength,
                                vortex_strength_uploaded=uploaded, core_radius=radius, particle_volume=volume,
                                selected_position=points, selected_velocity=velocity, selected_gradient=gradient,
                                selected_blob=blob, selected_blob_gradient=blob_gradient)
            result["sources"].append(hash_file(output_fields))
        assert signature(arrays) == before
        result["input_particle_arrays_unchanged"] = True
        frozen_path = ROOT / "frozen-workspace.json"
        frozen = json.loads(frozen_path.read_text())
        for row in frozen["records"]:
            assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
        result["frozen_original_files_verified"] = len(frozen["records"])
        result["sources"].append(hash_file(frozen_path))
        for row in result["sources"]:
            assert hash_file(ROOT / row["path"]) == row
        result.update(status="complete", stage="complete")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "limitations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consistency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-projection-sweeps", type=int, choices=(4, 6), required=True)
    args = parser.parse_args()
    args.consistency, args.output = args.consistency.resolve(), args.output.resolve()
    run(args)
