#!/usr/bin/env python3
"""Qualify a 3D source Jacobian against independently integrated stage targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from studies.coupler_accuracy.analytical_panel_gradient_3d import source_panel_gradient
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    probe = json.loads(args.probe.read_text())
    assert probe["status"] == "complete" and probe["schema"] == "openonda-body-stage-snapshot-reference-3d/1"
    assert probe["spatial_dimensions"] == 3 and probe["native_stage_replay_bitwise_equal"]
    sources = [hash_file(args.probe), *probe["sources"]]
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    selected_path = args.probe.parent / "selected-targets.npz"
    assert hash_file(selected_path) in sources
    selected = read_arrays(selected_path)
    prefix_row = next(row for row in sources if row["path"].endswith("long-wake-prefix-verification-3d.json"))
    prefix = json.loads((ROOT / prefix_row["path"]).read_text())
    frame = next(row for row in prefix["profile_frames"] if row["coupling_step"] == 20)
    fields = read_arrays(ROOT / frame["profile_fields"]["path"])
    vertices, strengths = fields["panel_vertices"].astype(float), fields["panel_source_strength"].astype(float)
    points = selected["position"]
    jacobian = source_panel_gradient(points, vertices, strengths)
    difference = float(np.max(np.abs(jacobian - selected["quadrature_body_gradient"])))
    assert difference < 2e-10
    symmetry = float(np.max(np.abs(jacobian - jacobian.transpose(0, 2, 1))))
    divergence = float(np.max(np.abs(np.trace(jacobian, axis1=1, axis2=2))))
    assert symmetry < 2e-11 and divergence < 2e-11
    rotation, _ = np.linalg.qr(np.array([[1., 2., 3.], [-3., 1., 2.], [2., -3., 1.]]))
    assert np.linalg.det(rotation) > 0
    covariance = []
    for scale in (.07, 3.2):
        shift = np.array([1.7, -2.3, .4])
        rotated = source_panel_gradient(scale * (points @ rotation.T) + shift,
                                        scale * (vertices @ rotation.T) + shift, strengths)
        expected = np.einsum("ij,njk,lk->nil", rotation, jacobian, rotation) / scale
        relative = float(np.max(np.abs(rotated - expected)) / np.max(np.abs(expected)))
        assert relative < 2e-10
        covariance.append({"length_scale": scale, "relative_maximum_difference": relative})
    linearity = []
    for scale in (1e-18, -2.3, 1e18):
        scaled = source_panel_gradient(points, vertices, scale * strengths)
        relative = float(np.max(np.abs(scaled - scale * jacobian)) / np.max(np.abs(scale * jacobian)))
        assert relative < 2e-13
        linearity.append({"strength_scale": scale, "relative_maximum_difference": relative})
    np.testing.assert_array_equal(source_panel_gradient(points, vertices, np.zeros_like(strengths)), np.zeros_like(jacobian))
    started = time.perf_counter()
    all_gradient = source_panel_gradient(fields["particle_position"], vertices, strengths)
    elapsed = time.perf_counter() - started
    np.testing.assert_array_equal(all_gradient[selected["particle_indices"]], jacobian)
    args.output.mkdir(parents=True)
    output_fields = args.output / "analytical-panel-gradient-fields.npz"
    np.savez_compressed(output_fields, position=points, gradient=jacobian, all_particle_gradient=all_gradient)
    paths = [Path(__file__).resolve(), ROOT / "studies/coupler_accuracy/analytical_panel_gradient_3d.py",
             ROOT / "studies/coupler_accuracy/native_volume_induction_3d.py"]
    sources += [hash_file(path) for path in paths] + [hash_file(output_fields)]
    for path in paths:
        target = args.output / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    result = {"schema": "openonda-analytical-panel-gradient-qualification-3d/1", "status": "complete",
              "spatial_dimensions": 3, "independent_targets": len(points), "all_particles": len(all_gradient),
              "all_particle_evaluation_seconds_after_compilation": elapsed,
              "maximum_quadrature_difference": difference, "maximum_antisymmetric_entry": symmetry,
              "maximum_divergence": divergence, "covariance_checks": covariance, "linearity_checks": linearity,
              "sources": list({row["path"]: row for row in sources}.values()),
              "limitations": [
                  "Independent area quadrature covers 256 physical off-surface targets, including particles 0.03125 from the unit cube. All 28,441 saved particle gradients are finite; only the selected subset has an independent integral reference.",
                  "Covariance checks rotate, translate and scale those 3D targets and panels. Strength checks include very small, negative and large amplitudes.",
                  "The study operator differentiates source velocity; it does not alter particle stretching conventions or qualify a coupled trajectory. Surface traces and singular source-edge targets are outside scope."]}
    (args.output / "analytical-panel-gradient-qualification-3d.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "limitations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.probe, args.output = args.probe.resolve(), args.output.resolve()
    run(args)
