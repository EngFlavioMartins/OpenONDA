#!/usr/bin/env python3
"""Measure stretching-form and vorticity consistency on the qualified 3D cube state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.particle_stage_induction_audit_3d import direct_gaussian, rms


def gaussian_vorticity(points, position, strength, radius):
    """Full source sum, with J[i,j] = derivative of vorticity i in direction j."""
    value = np.zeros((len(points), 3))
    gradient = np.zeros((len(points), 3, 3))
    for start in range(0, len(position), 512):
        cut = slice(start, start + 512)
        offset = points[:, None] - position[None, cut]
        sigma = radius[None, cut]
        zeta = np.exp(-np.sum(offset * offset, axis=2) / sigma**2) / (np.pi**1.5 * sigma**3)
        value += zeta @ strength[cut]
        gradient += np.einsum("qs,si,qsj->qij", -2 * zeta / sigma**2, strength[cut], offset)
    return value, gradient


def curl(gradient):
    return np.column_stack((gradient[:, 2, 1] - gradient[:, 1, 2],
                            gradient[:, 0, 2] - gradient[:, 2, 0],
                            gradient[:, 1, 0] - gradient[:, 0, 1]))


def weighted_angle(strength, vorticity):
    weight = np.linalg.norm(strength, axis=1)
    denominator = weight * np.linalg.norm(vorticity, axis=1)
    usable = denominator > np.finfo(float).tiny
    assert usable.any() and weight[usable].sum() > 0
    cosine = np.einsum("ni,ni->n", strength[usable], vorticity[usable]) / denominator[usable]
    return float(np.average(np.degrees(np.arccos(np.clip(cosine, -1, 1))), weights=weight[usable]))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = []

    def checked(row):
        assert hash_file(ROOT / row["path"]) == row
        sources.append(row)
        return ROOT / row["path"]

    audit = json.loads(args.audit.read_text())
    assert audit["schema"] == "openonda-particle-stage-induction-audit-3d/1" and audit["status"] == "complete"
    assert audit["spatial_dimensions"] == 3 and audit["physical_time"] == 1.5 and audit["checked_targets"] == 256
    sources.append(hash_file(args.audit))
    for row in audit["sources"]:
        checked(row)
    reference_row = next(row for row in sources if row["path"].endswith("particle-stage-reference-fields.npz"))
    prefix_row = next(row for row in sources if row["path"].endswith("long-wake-prefix-verification-3d.json"))
    reference = read_arrays(checked(reference_row))
    prefix = json.loads(checked(prefix_row).read_text())
    assert prefix["status"] == "complete" and prefix["schema"] == "openonda-long-wake-prefix-verification-3d/2"
    frame = next(row for row in prefix["profile_frames"] if row["coupling_step"] == 20)
    fields = read_arrays(checked(frame["profile_fields"]))
    position, strength, radius = (fields[key].astype(float) for key in
                                  ("particle_position", "particle_vortex_strength", "particle_core_radius"))
    indices, points = reference["indices"], reference["position"]
    np.testing.assert_array_equal(points, position[indices])
    assert len(position) == 28441 and np.all(radius == .0625)
    target_strength = strength[indices]
    gradient = reference["direct_gradient"]
    blob, blob_gradient = gaussian_vorticity(points, position, strength, radius)
    velocity_curl = curl(gradient)
    direct = np.einsum("nij,nj->ni", gradient, target_strength)
    transposed = np.einsum("nji,nj->ni", gradient, target_strength)
    np.testing.assert_array_equal(transposed, reference["direct_strength_rate"])
    difference = direct - transposed
    identity = np.cross(velocity_curl, target_strength)
    identity_error = float(np.max(np.abs(difference - identity)))
    identity_bound = 32 * np.finfo(float).eps * np.max(np.sum(np.abs(gradient), axis=(1, 2)) * np.linalg.norm(target_strength, axis=1))
    assert identity_error <= identity_bound

    # Differentiate the reconstructed field independently of its analytic
    # Gaussian derivative. Use selected targets from both physical regions.
    subset = np.unique(np.linspace(0, len(points) - 1, 8, dtype=int))
    fd_checks = []
    for h in (2e-5, 1e-5):
        fd = np.empty((len(subset), 3, 3))
        for axis in range(3):
            offset = h * np.eye(3)[axis]
            shifted = [gaussian_vorticity(points[subset] + multiple * offset, position, strength, radius)[0]
                       for multiple in (-2, -1, 1, 2)]
            fd[:, :, axis] = (shifted[0] - 8 * shifted[1] + 8 * shifted[2] - shifted[3]) / (12 * h)
        error = float(np.max(np.abs(fd - blob_gradient[subset])))
        relative = rms(fd - blob_gradient[subset]) / rms(blob_gradient[subset])
        assert error < 1e-6 and relative < 1e-8
        fd_checks.append({"probe": h, "maximum_difference": error, "relative_frobenius_error": relative})

    # A fully 3D finite particle set illustrates the conservation tradeoff.
    # It is a component example, not a substitute for the physical cube flow.
    rng = np.random.default_rng(519)
    toy_position, toy_strength = rng.normal(size=(9, 3)), rng.normal(size=(9, 3))
    toy_radius = np.full(9, .3)
    _, toy_gradient = direct_gaussian(toy_position, toy_radius, toy_position, toy_strength, toy_radius)
    toy_direct = np.einsum("nij,nj->ni", toy_gradient, toy_strength)
    toy_transposed = np.einsum("nji,nj->ni", toy_gradient, toy_strength)
    conservation_bound = 64 * np.finfo(float).eps * np.sum(np.linalg.norm(toy_transposed, axis=1))
    assert np.linalg.norm(toy_transposed.sum(axis=0)) < conservation_bound
    assert np.linalg.norm(toy_direct.sum(axis=0)) > .01
    np.testing.assert_allclose(toy_direct - toy_transposed, np.cross(curl(toy_gradient), toy_strength), rtol=0, atol=2e-14)
    # Rotation covariance exercises all vector and derivative components.
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    toy_blob, toy_blob_gradient = gaussian_vorticity(toy_position, toy_position, toy_strength, toy_radius)
    rotated_blob, rotated_gradient = gaussian_vorticity(toy_position @ rotation.T, toy_position @ rotation.T,
                                                       toy_strength @ rotation.T, toy_radius)
    rotation_value_error = float(np.max(np.abs(rotated_blob - toy_blob @ rotation.T)))
    rotation_gradient_error = float(np.max(np.abs(rotated_gradient - np.einsum("ia,nab,jb->nij", rotation, toy_blob_gradient, rotation))))
    assert rotation_value_error < 2e-12 and rotation_gradient_error < 2e-12

    divergence = np.trace(blob_gradient, axis1=1, axis2=2)
    regions = {"near_body": np.max(np.abs(points), axis=1) < 1,
               "near_wake": (points[:, 0] > 1.5) & (points[:, 0] <= 4) & (np.max(np.abs(points[:, 1:]), axis=1) < 1.5)}
    metrics = {}
    for name, mask in regions.items():
        assert mask.sum() == 128
        reference_rate = rms(transposed[mask])
        metrics[name] = {
            "targets": int(mask.sum()),
            "direct_minus_transposed_rate_rms": rms(difference[mask]),
            "direct_minus_transposed_rate_relative_to_transposed": rms(difference[mask]) / reference_rate,
            "native_fmm_rate_error_relative_to_transposed": audit["regions"][name]["stage_strength_rate_relative_error"],
            "blob_minus_velocity_curl_rms": rms(blob[mask] - velocity_curl[mask]),
            "blob_minus_velocity_curl_relative_to_curl": rms(blob[mask] - velocity_curl[mask]) / rms(velocity_curl[mask]),
            "blob_vorticity_rms": rms(blob[mask]), "velocity_curl_rms": rms(velocity_curl[mask]),
            "blob_divergence_rms_over_gradient_rms": float(np.sqrt(np.mean(divergence[mask]**2))) / rms(blob_gradient[mask]),
            "strength_weighted_angle_to_blob_degrees": weighted_angle(target_strength[mask], blob[mask]),
            "strength_weighted_angle_to_velocity_curl_degrees": weighted_angle(target_strength[mask], velocity_curl[mask]),
            "using_blob_in_cross_product_relative_rate_discrepancy": rms((difference - np.cross(blob, target_strength))[mask]) / reference_rate,
        }
    args.output.mkdir(parents=True)
    output_fields = args.output / "stretching-consistency-fields.npz"
    np.savez_compressed(output_fields, position=points, indices=indices, target_strength=target_strength,
                        velocity_gradient=gradient, velocity_curl=velocity_curl,
                        blob_vorticity=blob, blob_gradient=blob_gradient,
                        direct_strength_rate=direct, transposed_strength_rate=transposed)
    sources.append(hash_file(output_fields))
    for path in (Path(__file__).resolve(), ROOT / "studies/coupler_accuracy/particle_stage_induction_audit_3d.py",
                 ROOT / "source/solvers/vpm/diagnostics/resolution.py"):
        sources.append(hash_file(path))
        archive = args.output / "sources" / path.relative_to(ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
    frozen_path = ROOT / "frozen-workspace.json"
    frozen = json.loads(frozen_path.read_text())
    assert frozen["status"] == "complete"
    for row in frozen["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    sources.append(hash_file(frozen_path))
    unique = {row["path"]: row for row in sources}
    for row in unique.values():
        assert hash_file(ROOT / row["path"]) == row
    result = {"schema": "openonda-particle-stretching-consistency-3d/1", "status": "complete",
              "spatial_dimensions": 3, "physical_time": 1.5, "source_particles": len(position), "selected_targets": len(points),
              "regions": metrics, "cross_product_identity_maximum_difference": identity_error,
              "independent_blob_gradient_checks": fd_checks,
              "rotation_value_maximum_difference": rotation_value_error, "rotation_gradient_maximum_difference": rotation_gradient_error,
              "synthetic_direct_total_rate": toy_direct.sum(axis=0).tolist(),
              "synthetic_transposed_total_rate": toy_transposed.sum(axis=0).tolist(),
              "synthetic_transposed_conservation_bound": conservation_bound,
              "frozen_original_files_verified": len(frozen["records"]), "sources": list(unique.values()),
              "literature": [{"title": "Winckelmans, Topics in vortex methods (1989), sections 3.2 and 3.4",
                               "url": "https://thesis.caltech.edu/697/5/winckelmans-gs_1989.pdf"}],
              "limitations": [
                  "This compares two discrete stretching rates on an identical accepted state, using the qualified direct Gaussian Jacobian. Their difference is not a measured error against the full FVM trajectory.",
                  "The curl of induced velocity is distinct from the raw Gaussian vorticity sum. Only the former gives the exact algebraic direct-minus-transposed identity with J[i,j] = du_i/dx_j.",
                  "The finite-set example verifies the transposed form's conservation rationale; no global conservation error of the 28,441-particle physical state is inferred from a target subset.",
                  "The body potential field is excluded, as in the particle-stage audit. Its exact Jacobian is symmetric, so it contributes equally to both contractions at an identical state.",
                  "No solver state advances and no production default changes. The sampled blob projection difference does not alone identify an optimal replacement or establish developed-wake accuracy.",
              ]}
    (args.output / "particle-stretching-consistency-3d.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "literature", "limitations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.audit, args.output = args.audit.resolve(), args.output.resolve()
    run(args)
