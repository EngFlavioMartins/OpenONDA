#!/usr/bin/env python3
"""Independently integrate saved normal traces and recompute their cell moments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def integrate(native, fields, affine):
    """Symmetric three-point triangle quadrature; no analytic moment helper."""
    nf = len(fields["face_origin"])
    flux, moment, area = np.zeros(nf), np.zeros((nf, 3)), np.zeros(nf)
    barycentric = (np.ones((3, 3)) + 3 * np.eye(3)) / 6
    value = fields["affine_face_velocity"] if affine else fields["constant_face_velocity"]
    gradient = fields["affine_face_gradient"] if affine else None
    shift = fields["normal_shift"][2 if affine else 1]
    for start in range(0, len(native.triangles), 32768):
        rows = slice(start, start + 32768)
        triangles, face = native.triangles[rows], native.face_ids[rows]
        sf = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
        ta = np.linalg.norm(sf, axis=1)
        relative = np.einsum("qv,tvi->tqi", barycentric, triangles - fields["face_origin"][face, None])
        u = np.broadcast_to(value[face, None], relative.shape).copy()
        if gradient is not None:
            u += np.einsum("tqi,tij->tqj", relative, gradient[face])
        u += shift[face, None, None] * (sf / ta[:, None])[:, None]
        flux_density = np.einsum("tqi,ti->tq", u, sf) / 3
        np.add.at(flux, face, flux_density.sum(axis=1))
        np.add.at(moment, face, (relative * flux_density[:, :, None]).sum(axis=1))
        np.add.at(area, face, ta)
    return flux, moment, area


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    parent_path = args.run / "flux-moment-compatibility-3d.json"
    parent = json.loads(parent_path.read_text())
    assert parent["status"] == "complete" and parent["spatial_dimensions"] == 3
    sources = []
    for row in parent["sources"]:
        assert hash_file(ROOT / row["path"]) == row
        if row["path"].endswith(".py"):
            assert hash_file(args.run / "sources" / row["path"])["sha256"] == row["sha256"]
        sources.append(row)
    records, metric_differences = [], []
    for name in ("full", "hybrid"):
        mesh_path = ROOT / next(row["path"] for row in sources if row["path"].endswith(f"trial/{name}/solution/mesh.npz"))
        mesh = load_native_mesh(mesh_path)
        native = NativeVolumeSources.from_mesh(mesh)
        n, ni = mesh["n_cells"], mesh["n_interior_faces"]
        for row in (r for r in parent["records"] if r["name"] == name):
            field_path = ROOT / row["fields"]["path"]
            assert hash_file(field_path) == row["fields"]
            f = read_arrays(field_path)
            sources.append(row["fields"])
            ids, volume = f["shared_cell_ids"], f["polyhedron_volume"]
            radius = np.max(np.abs(f["fvm_cell_centre"][ids]), axis=1)
            masks = {"all_shared": np.ones(len(ids), dtype=bool), "near_body_0.8": radius < .8,
                     "near_body_1.0": radius < 1, "overlap": (radius >= .75) & (radius < 1.25), "outer_shared": radius >= 1.25}
            for variant in (1, 2):
                flux, moment, area = integrate(native, f, variant == 2)
                flux_error = float(np.max(np.abs(flux - f["stored_flux"]) / area))
                moment_error = float(np.max(np.linalg.norm(moment - f["face_first_moment"][variant], axis=1) / area**1.5))
                assert flux_error < 2e-12 and moment_error < 2e-12
                net = np.bincount(mesh["owners"], weights=flux, minlength=n) - np.bincount(mesh["neighbours"], weights=flux[:ni], minlength=n)
                first = np.zeros((n, 3))
                for axis in range(3):
                    own = (f["face_origin"][:, axis] - f["cell_centroid"][mesh["owners"], axis]) * flux + moment[:, axis]
                    nei = (f["face_origin"][:ni, axis] - f["cell_centroid"][mesh["neighbours"], axis]) * flux[:ni] + moment[:ni, axis]
                    first[:, axis] = np.bincount(mesh["owners"], weights=own, minlength=n) - np.bincount(mesh["neighbours"], weights=nei, minlength=n)
                implied = first / volume[:, None]
                velocity_difference = float(np.max(np.abs(implied - f["implied_velocity"][variant])))
                assert velocity_difference < 2e-12
                assert float(np.max(np.abs(net - f["net_flux"]) / volume)) < 2e-11
                for region, mask in masks.items():
                    selected = ids[mask]
                    expected = row["variants"][variant]["regions"][region]
                    for key, target in (("mean_velocity_difference_rms_over_Uinf", f["stored_velocity"]),
                                        ("transported_momentum_difference_rms_as_velocity", f["stored_velocity"] * (f["fvm_volume"] / volume)[:, None])):
                        error = implied[selected] - target[selected]
                        actual = float(np.sqrt(np.sum(volume[selected] * np.sum(error**2, axis=1)) / volume[selected].sum()))
                        difference = abs(actual - expected[key])
                        assert difference < 2e-13
                        metric_differences.append(difference)
                records.append({"name": name, "fvm_step": row["fvm_step"], "variant": row["variants"][variant]["name"],
                                "maximum_face_flux_difference_as_normal_velocity": flux_error,
                                "maximum_face_moment_difference_over_area_to_three_halves": moment_error,
                                "maximum_implied_velocity_difference": velocity_difference})
    sources.append(hash_file(parent_path))
    path = Path(__file__).resolve()
    target = args.output.parent / (args.output.stem + "-sources") / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    sources.append(hash_file(path))
    result = {"schema": "openonda-flux-moment-verification-3d/1", "status": "complete", "spatial_dimensions": 3,
              "sources": sources, "records": records, "independently_recomputed_metrics": len(metric_differences),
              "maximum_metric_difference": max(metric_differences),
              "limitations": ["Surface quadrature verifies the stated normal-trace integrals and cell observations. It does not construct a divergence-free volume field or validate advancing flow agreement."]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"observations": len(records), "metrics": len(metric_differences), "maximum_metric_difference": max(metric_differences)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    run(args)
