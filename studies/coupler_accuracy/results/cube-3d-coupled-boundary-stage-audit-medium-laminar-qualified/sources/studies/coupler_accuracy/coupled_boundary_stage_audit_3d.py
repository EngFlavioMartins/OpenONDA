#!/usr/bin/env python3
"""Compare accepted FVM and post-replacement VPM boundary traces at one 3D endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.fvm.fields.mixed_velocity_boundary import (
    reconstruct_normal_velocity_tangential_gradient,
)
from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.native_face_velocity_sampling_3d import InteriorFaceVelocitySampler


def stage_errors(applied, replaced, reference, area):
    if applied.ndim == 1:
        applied, replaced, reference = (v[:, None] for v in (applied, replaced, reference))
    error, change = applied-reference, replaced-applied
    before, after, jump = (field_rms(value, area) for value in (error, replaced-reference, change))
    product = float(np.average(np.sum(error*change, axis=1), weights=area))
    np.testing.assert_allclose(after**2, before**2+2*product+jump**2, rtol=0, atol=1e-14)
    return {"fvm_trace_error_rms": before, "post_replacement_trace_error_rms": after,
            "replacement_change_rms": jump, "fvm_error_dot_replacement_change": product}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve()]
    small_path, full_path, map_path = (args.oracle / name for name in ("small-native-mesh.npz", "full-native-mesh.npz", "cell-and-face-map.npz"))
    small, full, mapping = load_native_mesh(small_path), load_native_mesh(full_path), read_arrays(map_path)
    for patch in full["boundary"]:
        patch["velocity_type"] = "fixedValue"
    sg = compute_mesh_geometry(small, gradient_scheme="gauss", compute_lsq=False)
    fg = compute_mesh_geometry(full, gradient_scheme="gauss", compute_lsq=False)
    cut = next(patch for patch in small["boundary"] if patch["name"] == "numericalBoundary")
    faces = np.arange(cut["start_face"], cut["start_face"]+cut["n_faces"])
    full_faces, signs = mapping["face_ids"][faces], mapping["signs"][faces]
    sf = sg["face_area_vector"][faces]
    area = np.linalg.norm(sf, axis=1)
    normal = sf/area[:, None]
    distance = np.sum(sg["cell_connection_vector"][faces]*normal, axis=1)
    owners, ghosts = small["owners"][faces], small["n_cells"]+faces-small["n_interior_faces"]
    np.testing.assert_allclose(sf, fg["face_area_vector"][full_faces]*signs[:, None], rtol=0, atol=1e-13)
    np.testing.assert_allclose(sg["cell_centre"], fg["cell_centre"][mapping["cell_ids"]], rtol=0, atol=1e-13)
    np.testing.assert_allclose(sg["cell_volume"], fg["cell_volume"][mapping["cell_ids"]], rtol=0, atol=1e-14)
    reference_path = args.reference / "reference-flux-replay-3d.json"
    reference = json.loads(reference_path.read_text())
    assert reference["status"] == "complete" and reference["spatial_dimensions"] == 3
    for row in reference["sources"]:
        assert hash_file(ROOT / row["path"]) == row
    endpoint = next(row for row in reference["records"] if row["name"] == "final")
    backup_path = ROOT / endpoint["backup"]["path"]
    assert hash_file(backup_path) == endpoint["backup"]
    full_state = decode_state(read_arrays(backup_path))
    sampler = InteriorFaceVelocitySampler(full, fg, full_faces, signs)
    reference_gt = sampler.evaluate(full_state["velocity"][sampler.sample_cells])["tangential_gradient"]
    reference_un = full_state["volumetric_face_flux"][full_faces]*signs/area
    paths += [small_path, full_path, map_path, reference_path, backup_path]
    records, states = [], []
    for trial in args.trial:
        name = trial.parent.name if trial.name == "trial" else trial.name
        report_path = trial / "cube-coupled-trial.json"
        manifest_path = trial / "hybrid/solution/backups/manifest.json"
        comparison_path = trial / "latest-comparison-fields.npz"
        report, manifest = (json.loads(path.read_text()) for path in (report_path, manifest_path))
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        assert report["boundary_mode"] == "vorticity_mixed" and manifest["coupling_step"] == 20
        for original in report["sources"]:
            if original["path"] in ("source/coupler/boundary.py", "source/coupler/solver.py", "source/solvers/fvm/fields/mixed_velocity_boundary.py"):
                assert hash_file(ROOT / original["path"]) == original
        artifacts = {}
        for key in ("fvm", "vpm_boundary_condition"):
            path = manifest_path.parent / manifest["artifacts"][key]
            assert hash_file(path)["sha256"] == manifest["artifact_sha256"][key]
            artifacts[key] = decode_state(read_arrays(path))
            paths.append(path)
        fvm, post = artifacts["fvm"], artifacts["vpm_boundary_condition"]
        comparison = read_arrays(comparison_path)
        np.testing.assert_array_equal(fvm["velocity"][:small["n_cells"]], comparison["hybrid_velocity"])
        np.testing.assert_array_equal(full_state["velocity"][mapping["cell_ids"]], comparison["full_velocity"])
        np.testing.assert_allclose(report["source_seed_time"]+float(fvm["time"]), endpoint["physical_time"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(float(fvm["time"]), manifest["time"], rtol=0, atol=1e-12)
        assert int(fvm["step"]) == manifest["fvm_step"] == 100
        assert int(post["boundary_schema_version"]) == 3 and bool(post["has_normal_velocity"]) and bool(post["has_tangential_gradient"])
        un = fvm["volumetric_face_flux"][faces]/area
        face_u, owner_u = fvm["velocity"][ghosts], fvm["velocity"][owners]
        np.testing.assert_allclose(np.sum(face_u*normal, axis=1), un, rtol=0, atol=2e-13)
        gt = (face_u-owner_u)/distance[:, None]
        gt -= np.sum(gt*normal, axis=1)[:, None]*normal
        recovered = reconstruct_normal_velocity_tangential_gradient(owner_u, normal, distance, un, gt)
        np.testing.assert_allclose(recovered, face_u, rtol=0, atol=2e-13)
        np.testing.assert_allclose(np.sum(post["velocity"]*normal, axis=1), post["normal_velocity"], rtol=0, atol=2e-13)
        np.testing.assert_allclose(np.sum(post["tangential_gradient"]*normal, axis=1), 0, rtol=0, atol=1e-12)
        row = {"name": name, "physical_time": endpoint["physical_time"], "coupling_step": manifest["coupling_step"],
               "normal_velocity": stage_errors(un, post["normal_velocity"], reference_un, area),
               "tangential_gradient": stage_errors(gt, post["tangential_gradient"], reference_gt, area),
               "fvm_cut_flux": float(area @ un), "post_replacement_cut_flux": float(area @ post["normal_velocity"]),
               "maximum_ghost_normal_flux_difference": float(np.max(np.abs(np.sum(face_u*normal, axis=1)-un)))}
        states.append((name, un, gt, post["normal_velocity"], post["tangential_gradient"]))
        records.append(row)
        paths += [report_path, manifest_path, comparison_path]
    paths += [ROOT / name for name in (
        "source/coupler/boundary.py", "source/coupler/solver.py", "source/coupler/backup.py",
        "source/solvers/fvm/fields/mixed_velocity_boundary.py", "source/solvers/fvm/fields/gradients.py",
        "source/solvers/fvm/io/backup.py", "source/solvers/fvm/io/mesh_storage.py", "source/solvers/fvm/mesh/geometry.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py", "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
        "studies/coupler_accuracy/native_face_velocity_sampling_3d.py", "studies/coupler_accuracy/native_face_trace_3d.py")]
    paths = list(dict.fromkeys(paths))
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            archive = args.output / "sources" / path.relative_to(ROOT)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(path.read_bytes())
    saved = {"face_position": sg["face_centre"][faces], "area": area, "normal": normal, "small_face_ids": faces,
             "full_face_ids": full_faces, "signs": signs, "reference_normal_velocity": reference_un, "reference_tangential_gradient": reference_gt}
    for name, un, gt, post_un, post_gt in states:
        saved.update({name+"__fvm_normal_velocity": un, name+"__fvm_face_increment_gradient": gt,
                      name+"__post_normal_velocity": post_un, name+"__post_tangential_gradient": post_gt})
    fields_path = args.output / "coupled-boundary-stage-fields.npz"
    np.savez_compressed(fields_path, **saved)
    result = {"schema": "openonda-coupled-boundary-stage-audit-3d/1", "status": "complete", "spatial_dimensions": 3,
              "small_fvm_cells": small["n_cells"], "boundary_faces": len(faces), "records": records,
              "sources": sources, "fields": hash_file(fields_path), "limitations": [
                  "The FVM trace is recovered from its saved conservative flux and face-valued ghost velocities after the last FVM substep.",
                  "The boundary-history backup contains VPM data recomputed after particle replacement at that same time. The FVM is not re-solved with that endpoint.",
                  "This fixed-time difference follows the documented partitioned algorithm; it is not a restart corruption or evidence of physical time advance during replacement.",
                  "The tangential-gradient comparison uses the recovered mixed face increment and the full reference's native face derivative; these are discrete boundary observations.",
                  "The endpoint jump and reference errors do not, by themselves, predict whether interface iteration improves an advancing trajectory."]}
    (args.output / "coupled-boundary-stage-audit-3d.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trial", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.oracle, args.reference, args.output = (path.resolve() for path in (args.oracle, args.reference, args.output))
    args.trial = [path.resolve() for path in args.trial]
    run(args)
