#!/usr/bin/env python3
"""Compare saved physical FMM stage rates with independent 3D Gaussian sums."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import erf

from source.solvers.vpm.kernels.base import make_vortex_kernel
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.cube_snapshot_induction import volume_velocity


def direct_gaussian(points, target_radius, position, strength, source_radius):
    """Sum Gaussian velocity and its analytic Jacobian, including the self limit."""
    velocity, jacobian = np.zeros_like(points), np.zeros((len(points), 3, 3))
    for start in range(0, len(position), 512):
        cut = slice(start, start + 512)
        r = points[:, None] - position[None, cut]
        r2 = np.sum(r * r, axis=2)
        sigma = .5 * (target_radius[:, None] + source_radius[None, cut])
        q = np.sqrt(r2) / sigma
        safe2 = np.where(r2 > 0, r2, 1.)
        f = (erf(q) - 2 / np.sqrt(np.pi) * q * np.exp(-q * q)) / (4 * np.pi * safe2**1.5)
        g = (np.exp(-q * q) / (np.pi**1.5 * sigma**3) - 3 * f) / safe2
        small = q < .01
        q2 = q * q
        f[small] = ((1 - .6 * q2 + 3 / 14 * q2**2 - q2**3 / 18 + q2**4 / 88)
                    / (3 * np.pi**1.5 * sigma**3))[small]
        g[small] = ((-6 / 5 + 6 / 7 * q2 - q2**2 / 3 + q2**3 / 11)
                    / (3 * np.pi**1.5 * sigma**5))[small]
        cross = np.cross(strength[None, cut], r)
        velocity += np.einsum("qs,qsi->qi", f, cross)
        jacobian += np.einsum("qs,qsi,qsj->qij", g, cross, r)
        weighted = np.einsum("qs,si->qi", f, strength[cut])
        jacobian[:, 0, 1] -= weighted[:, 2]
        jacobian[:, 0, 2] += weighted[:, 1]
        jacobian[:, 1, 0] += weighted[:, 2]
        jacobian[:, 1, 2] -= weighted[:, 0]
        jacobian[:, 2, 0] -= weighted[:, 1]
        jacobian[:, 2, 1] += weighted[:, 0]
    return velocity, jacobian


def rms(value):
    return float(np.sqrt(np.mean(np.sum(np.asarray(value, dtype=float)**2, axis=tuple(range(1, value.ndim))))))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    probe = json.loads(args.probe.read_text())
    assert probe["schema"] == "openonda-body-stage-snapshot-reference-3d/1" and probe["status"] == "complete"
    assert probe["spatial_dimensions"] == 3 and probe["native_stage_replay_bitwise_equal"]
    sources = [hash_file(args.probe), *probe["sources"]]
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    targets_path, stage_path = (args.probe.parent / name for name in ("selected-targets.npz", "native-stage-without-body.npz"))
    assert hash_file(targets_path) in sources and hash_file(stage_path) in sources
    selected, stage = read_arrays(targets_path), read_arrays(stage_path)
    prefix_row = next(row for row in sources if row["path"].endswith("long-wake-prefix-verification-3d.json"))
    prefix = json.loads((ROOT / prefix_row["path"]).read_text())
    frame = next(row for row in prefix["profile_frames"] if row["coupling_step"] == 20)
    fields = read_arrays(ROOT / frame["profile_fields"]["path"])
    position, strength, radius = (fields[key].astype(float) for key in
                                  ("particle_position", "particle_vortex_strength", "particle_core_radius"))
    indices, points = selected["particle_indices"], selected["position"]
    np.testing.assert_array_equal(points, position[indices])
    # This snapshot has one GBD core radius. Particle-pair and source-only
    # point-query smoothing then agree, so their contrast isolates evaluation.
    assert np.all(radius == .0625)
    velocity, jacobian = direct_gaussian(points, radius[indices], position, strength, radius)
    legacy_velocity = volume_velocity(points, position, strength, radius)[0]
    velocity_check = float(np.max(np.abs(velocity - legacy_velocity)))
    assert velocity_check < 2e-12
    kernel = make_vortex_kernel("GAUSSIAN")
    host_jacobian = np.zeros_like(jacobian)
    for start in range(0, len(position), 512):
        cut = slice(start, start + 512)
        host_jacobian += kernel.gradient_pair(points[:, None] - position[None, cut], strength[None, cut],
                                              radius[indices, None], radius[None, cut]).sum(axis=1)
    gradient_check = float(np.max(np.abs(host_jacobian - jacobian)))
    assert gradient_check < 2e-11
    sample = np.unique(np.linspace(0, len(points) - 1, 8, dtype=int))
    fd_checks = []
    for h in (2e-5, 1e-5):
        fd = np.empty((len(sample), 3, 3))
        for axis in range(3):
            offset = np.eye(3)[axis] * h
            fd[:, :, axis] = (volume_velocity(points[sample] - 2 * offset, position, strength, radius)[0]
                               - 8 * volume_velocity(points[sample] - offset, position, strength, radius)[0]
                               + 8 * volume_velocity(points[sample] + offset, position, strength, radius)[0]
                               - volume_velocity(points[sample] + 2 * offset, position, strength, radius)[0]) / (12 * h)
        difference = float(np.max(np.abs(fd - jacobian[sample])))
        assert difference < 2e-8
        fd_checks.append({"probe": h, "maximum_gradient_difference": difference})
    direct_rate = np.einsum("nji,nj->ni", jacobian, strength[indices])
    stage_rate_from_gradient = np.einsum("nji,nj->ni", stage["gradient"][indices].astype(float), strength[indices])
    rate_check = float(np.max(np.abs(stage_rate_from_gradient - stage["strength_rate"][indices])))
    rate_bound = 16 * np.finfo(np.float32).eps * np.einsum("nji,nj->ni", np.abs(stage["gradient"][indices]), np.abs(strength[indices]))
    assert np.all(np.abs(stage_rate_from_gradient - stage["strength_rate"][indices]) <= rate_bound + 1e-20)
    full_velocity = velocity + [1., 0., 0.]
    regions = {"near_body": np.max(np.abs(points), axis=1) < 1,
               "near_wake": (points[:, 0] > 1.5) & (points[:, 0] <= 4) & (np.max(np.abs(points[:, 1:]), axis=1) < 1.5)}
    metrics = {}
    for name, mask in regions.items():
        metrics[name] = {
            "targets": int(mask.sum()),
            "stage_velocity_error_rms_over_Uinf": rms((stage["velocity"][indices] - full_velocity)[mask]),
            "query_velocity_error_rms_over_Uinf": rms((selected["native_query_without_body"] - full_velocity)[mask]),
            "stage_gradient_error_frobenius_rms": rms((stage["gradient"][indices] - jacobian)[mask]),
            "stage_gradient_relative_frobenius_error": rms((stage["gradient"][indices] - jacobian)[mask]) / rms(jacobian[mask]),
            "stage_strength_rate_error_rms": rms((stage["strength_rate"][indices] - direct_rate)[mask]),
            "stage_strength_rate_relative_error": rms((stage["strength_rate"][indices] - direct_rate)[mask]) / rms(direct_rate[mask]),
            "direct_gradient_frobenius_rms": rms(jacobian[mask]),
            "direct_strength_rate_rms": rms(direct_rate[mask]),
        }
    args.output.mkdir(parents=True)
    path = args.output / "particle-stage-reference-fields.npz"
    np.savez_compressed(path, position=points, indices=indices, direct_velocity=full_velocity,
                        direct_gradient=jacobian, direct_strength_rate=direct_rate,
                        stage_velocity=stage["velocity"][indices], stage_gradient=stage["gradient"][indices],
                        stage_strength_rate=stage["strength_rate"][indices], query_velocity=selected["native_query_without_body"])
    paths = [Path(__file__).resolve(), ROOT / "studies/coupler_accuracy/cube_snapshot_induction.py",
             ROOT / "source/solvers/vpm/kernels/base.py", ROOT / "source/solvers/vpm/physics/induction/fmm/device.py"]
    sources += [hash_file(path), *[hash_file(path) for path in paths]]
    for path in paths:
        archive = args.output / "sources" / path.relative_to(ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    result = {"schema": "openonda-particle-stage-induction-audit-3d/1", "status": "complete",
              "spatial_dimensions": 3, "physical_time": 1.5, "sources_count": len(position),
              "checked_targets": len(points), "uniform_core_radius": .0625,
              "independent_velocity_check_maximum_difference": velocity_check,
              "host_pair_gradient_check_maximum_difference": gradient_check,
              "independent_velocity_difference_gradient_checks": fd_checks,
              "native_stage_stretching_contraction_check_maximum_difference": rate_check,
              "regions": metrics, "sources": list({row["path"]: row for row in sources}.values()),
              "limitations": [
                  "This audits the saved actual FMM stage at one physical particle state. It does not advance either solver or attribute the coupled trajectory error to FMM alone.",
                  "The arbitrary-target FMM boundary currently uses direct regularized kernels. Its accuracy does not establish the accuracy of the separate hierarchical particle-stage path.",
                  "All selected targets use the same 28,441 sources, including the finite self-gradient limit. Uniform core radii remove the separate pair-versus-source smoothing distinction.",
                  "The full direct stage-rate comparison uses the native transposed stretching convention. It does not establish that this discrete formulation matches the FVM vorticity equation exactly."]}
    (args.output / "particle-stage-induction-audit-3d.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "limitations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.probe, args.output = args.probe.resolve(), args.output.resolve()
    run(args)
