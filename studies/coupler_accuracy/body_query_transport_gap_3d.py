#!/usr/bin/env python3
"""Measure the body field included in queries but excluded by the selected RK scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import taichi as ti

from source.solvers.vpm.boundary_elements.panels.kernels.induced_velocity import (
    compute_source_induced_velocity_kernel,
)
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.native_volume_induction_3d import triangle_source_integrals
from studies.coupler_accuracy.verify_accepted_wake_prefix_3d import norms


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    prefix = json.loads(args.prefix.read_text())
    assert prefix["status"] == "complete" and prefix["schema"] == "openonda-long-wake-prefix-verification-3d/2"
    sources = [hash_file(args.prefix), *prefix["sources"]]
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    geometry = read_arrays(ROOT / prefix["profile_geometry"]["path"])
    points = geometry["position"][geometry["fluid_mask"]]
    args.output.mkdir(parents=True)
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    results, kernel_errors = [], []
    for frame in prefix["profile_frames"]:
        fields = read_arrays(ROOT / frame["profile_fields"]["path"])
        vertices, normal, strength = (fields[key].astype(float) for key in ("panel_vertices", "panel_normals", "panel_source_strength"))
        body = np.einsum("pti,t->pi", triangle_source_integrals(points, vertices)[1], strength) / (4 * np.pi)
        native = np.zeros_like(points)
        compute_source_induced_velocity_kernel(np.ascontiguousarray(vertices), np.ascontiguousarray(normal), np.ascontiguousarray(strength), np.ascontiguousarray(points), native)
        difference = float(np.max(np.abs(body - native)))
        assert difference < 2e-11
        kernel_errors.append(difference)
        direct = frame.get("direct_profile_check")
        completed = None
        if direct:
            path = ROOT / direct["fields"]["path"]
            assert hash_file(path) == direct["fields"]
            values = read_arrays(path)
            np.testing.assert_array_equal(values["position"], points)
            completed = values["direct_velocity"]
        regions = {}
        for y, name in ((0., "centreline"), (.75, "offaxis_y075")):
            line = points[:, 1] == y
            masks = {"small_fvm": line & (np.max(np.abs(points), axis=1) <= 1.5),
                     "upstream": line & (points[:, 0] < -1.5),
                     "near_wake": line & (points[:, 0] > 1.5) & (points[:, 0] <= 4),
                     "far_wake": line & (points[:, 0] > 4)}
            for region, mask in masks.items():
                value = {"body_velocity": norms(body[mask]), "sample_points": int(mask.sum())}
                if completed is not None:
                    value["body_completed_reference_error"] = norms((completed - fields["full_profile"])[mask])
                    value["particle_plus_freestream_reference_error"] = norms((completed - body - fields["full_profile"])[mask])
                regions[name + "_" + region] = value
        path = args.output / f"body-query-step-{frame['coupling_step']:06d}.npz"
        np.savez_compressed(path, position=points, body_velocity=body)
        sources.append(hash_file(path))
        results.append({"coupling_step": frame["coupling_step"], "physical_time": frame["physical_time"], "regions": regions})
    paths = [Path(__file__).resolve(), ROOT / "source/solvers/vpm/core/solver.py",
             ROOT / "source/solvers/vpm/physics/stage_rhs.py", ROOT / "source/solvers/vpm/coupling/stepper.py",
             ROOT / "studies/coupler_accuracy/cube_coupled_trial.py"]
    sources += [hash_file(path) for path in paths]
    for path in paths:
        copy = args.output / "sources" / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    result = {
        "schema": "openonda-body-query-transport-gap-3d/1", "status": "complete", "spatial_dimensions": 3,
        "scope": "saved-source evaluation and frozen-code inspection",
        "selected_panel_scope": "vpm_boundary_condition",
        "configuration_finding": "The selected scope installs a body target-query callback, but sets physics.body_velocity, body_velocity_field and body_velocity_gradient to None. CouplingStepper.advance_panel also returns without advancing panels in this scope. The study adds no body-stage override.",
        "profile_frames": results, "maximum_independent_panel_kernel_difference": max(kernel_errors),
        "sources": list({row["path"]: row for row in sources}.values()),
        "limitations": [
            "The callback finding comes from the archived selected configuration and implementation. This script does not execute a particle RK stage or a body-transport counterfactual.",
            "Regional magnitudes are line norms in the 3D solution. They are not volume norms or bounds on accumulated trajectory, stretching, force or wake errors.",
            "Removing the body term from a saved query compares two instantaneous fields on the same sources. It does not predict the trajectory obtained by changing particle transport.",
            "Enabling the full panel coupling scope would additionally alter panel advancement and other behavior; it would not isolate the body transport contribution.",
        ],
    }
    (args.output / "body-query-transport-gap-3d.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "frames": len(results), "maximum_kernel_difference": max(kernel_errors), "final": results[-1]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.prefix, args.output = args.prefix.resolve(), args.output.resolve()
    run(args)
