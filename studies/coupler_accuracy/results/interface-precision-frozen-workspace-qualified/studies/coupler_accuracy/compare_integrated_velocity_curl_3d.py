#!/usr/bin/env python3
"""Compare three frozen observations using the same unused cells and faces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    families = (("point_raw", "cube-3d-cell-integral-point-control"),
                ("integrated_raw", "cube-3d-cell-integral-reconstruction"),
                ("integrated_velocity_curl", "cube-3d-integrated-velocity-curl-reconstruction"))
    matrix_dir = args.results / "cube-3d-native-velocity-curl-integrals"
    paths = [Path(__file__), matrix_dir / "velocity-curl-integrals.npy",
             matrix_dir / "velocity-curl-integral-inputs.npz", matrix_dir / "velocity-curl-integral-audit.json"]
    for _, name in families:
        paths += [args.results / name / f for f in ("native-reconstruction-3d.json", "held-fields.npz")]
        paths.append(args.results / (name + "-boundary") / "native-curl-3d.json")
    sources = [hash_file(p) for p in paths]
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2) + "\n")
    new_dir = args.results / families[-1][1]
    with np.load(new_dir / "held-fields.npz", allow_pickle=False) as data:
        anchor = {k: data[k].copy() for k in data.files}
    metadata = json.loads((new_dir / "native-reconstruction-3d.json").read_text())
    assert metadata["fit_data"] == "all_donor" and metadata["held_cells"] == 328
    matrix = np.load(matrix_dir / "velocity-curl-integrals.npy", mmap_mode="r", allow_pickle=False)
    n = len(anchor["renewable_position"])
    held_matrix = np.array(matrix[3*metadata["fit_cells"]:, :3*n], order="C", copy=True)
    held_matrix /= np.repeat(anchor["volume"], 3)[:, None]
    del matrix
    prior = anchor["donor_volume_vorticity__strength"]
    baseline_curl = anchor["donor_volume_vorticity__cell_integrated_velocity_curl"]
    omega_rms = field_rms(anchor["fvm_vorticity"], anchor["volume"])
    baseline_boundary = None
    records = []
    for label, name in families:
        directory = args.results / name
        report = json.loads((directory / "native-reconstruction-3d.json").read_text())
        audit = json.loads((args.results / (name+"-boundary") / "native-curl-3d.json").read_text())
        assert report["fit_data"] == "all_donor" and report["fit_cells"] == 2512
        assert report["held_cells"] == 328 and report["spatial_dimensions"] == 3
        assert audit["boundary_faces"] == 864
        boundary = {r["name"]: r for r in audit["results"]}
        if baseline_boundary is None:
            baseline_boundary = boundary["donor_volume_vorticity"]
        else:
            for key in ("boundary_normal_velocity_rms_error_over_Uinf", "boundary_tangential_normal_gradient_rms_error",
                        "boundary_native_flux_tangential_normal_gradient_rms_error"):
                assert boundary["donor_volume_vorticity"][key] == baseline_boundary[key]
        with np.load(directory / "held-fields.npz", allow_pickle=False) as fields:
            for key in ("position", "volume", "fvm_velocity", "fvm_vorticity", "renewable_position", "radius"):
                np.testing.assert_array_equal(fields[key], anchor[key])
            np.testing.assert_array_equal(fields["donor_volume_vorticity__strength"], prior)
            for index, row in enumerate(report["results"]):
                if index == 1:
                    continue  # The rejected unregularized historical fit is not a candidate.
                strength = fields[row["name"] + "__strength"]
                curl = baseline_curl + (held_matrix @ (strength-prior).ravel()).reshape(-1, 3)
                velocity = fields[row["name"] + "__velocity"]
                point_curl = fields[row["name"] + "__velocity_curl"]
                if label == "integrated_velocity_curl":
                    np.testing.assert_allclose(curl, fields[row["name"] + "__cell_integrated_velocity_curl"],
                                               rtol=0, atol=3e-14)
                values = {"family": label, "name": row["name"], "objective_index": index-2,
                          "held_velocity_rms_over_Uinf": field_rms(velocity-anchor["fvm_velocity"], anchor["volume"]),
                          "held_cell_integrated_velocity_curl_relative_error": field_rms(curl-anchor["fvm_vorticity"], anchor["volume"])/omega_rms,
                          "held_point_velocity_curl_relative_error": field_rms(point_curl-anchor["fvm_vorticity"], anchor["volume"])/omega_rms,
                          "strength_l1_over_prior": row["strength_l1_over_prior"],
                          **{key: boundary[row["name"]][key] for key in (
                              "boundary_normal_velocity_rms_error_over_Uinf",
                              "boundary_tangential_normal_gradient_rms_error",
                              "boundary_native_flux_tangential_normal_gradient_rms_error")}}
                records.append(values)
    output = {"schema": "openonda-integrated-velocity-curl-comparison-3d/1", "spatial_dimensions": 3,
              "held_cells": 328, "fit_cells": 2512, "boundary_faces": 864, "held_vorticity_rms": omega_rms,
              "records": records, "sources": sources,
              "verification": "Identical geometry, volumes, positions, radii, donor baseline and boundary target convention; old strengths are measured without refitting.",
              "limitations": ["Unused cells lie in a weak-vorticity outer layer; their relative errors do not characterize near-body vorticity.",
                              "Frozen component comparison; no new live trajectory or acceptance threshold is supplied."]}
    (args.output / "integrated-velocity-curl-comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "studies/coupler_accuracy/results")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
