#!/usr/bin/env python3
"""Measure exterior curl of the frozen shared-trace source correction in 3D."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.native_linear_volume_induction_3d import (
    LinearNativeVolumeSources,
    curl_affine,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(path):
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def rms(value):
    return float(np.sqrt(np.mean(np.sum(value*value, axis=-1))))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    experiments, paths = [], [Path(__file__).resolve()]
    paths += [ROOT / path for path in (
        "studies/coupler_accuracy/native_linear_volume_induction_3d.py",
        "studies/coupler_accuracy/native_volume_induction_3d.py",
        "source/solvers/fvm/io/mesh_storage.py", "source/solvers/fvm/mesh/geometry.py")]
    for directory in args.source:
        report_path = directory / "cube-shared-trace-sources-3d.json"
        source_path = directory / "shared-trace-source-fields.npz"
        report = json.loads(report_path.read_text())
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        for row in report["sources"]:
            assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
        mesh_record = next(row for row in report["sources"] if row["path"].endswith("/small-native-mesh.npz"))
        mesh_path = ROOT / mesh_record["path"]
        paths += [report_path, source_path, mesh_path]
        source = arrays(source_path)
        mesh = load_native_mesh(mesh_path)
        geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
        native = NativeVolumeSources.from_mesh(mesh)
        linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
        for key in ("volume", "centroid", "covariance"):
            np.testing.assert_array_equal(source[key], getattr(linear, key))
        base, updated = np.array([2, 5]), np.array([4, 7])
        gamma = source["delta_circulation"][updated]-source["delta_circulation"][base]
        moment = source["first_moment"][updated]-source["first_moment"][base]
        coefficient, gradient = linear.coefficients(gamma, moment)
        active = np.any(gamma != 0, axis=(0, 2)) | np.any(moment != 0, axis=(0, 2, 3))
        assert active.any()
        interior = native.neighbours >= 0
        triangle_active = active[native.owners].copy()
        triangle_active[interior] |= active[native.neighbours[interior]]
        support = native.triangles[triangle_active]
        bounds = np.stack((support.min(axis=(0, 1)), support.max(axis=(0, 1))))
        cut = next(patch for patch in mesh["boundary"] if patch["name"] == "numericalBoundary")
        faces = np.arange(cut["start_face"], cut["start_face"]+cut["n_faces"])
        face_normal = geometry["face_area_vector"][faces]
        side = 2*np.argmax(np.abs(face_normal), axis=1)+(face_normal.sum(axis=1) > 0).astype(int)
        rng = np.random.default_rng(20260913)
        selected = np.concatenate([np.sort(rng.choice(np.flatnonzero(side == k), 4, replace=False)) for k in range(6)])
        faces, side = faces[selected], side[selected]
        position = geometry["face_centre"][faces]
        step = report["particle_spacing"]*1e-4
        offsets = np.eye(3)[:, None, :]*np.array([1., -1., .5, -.5])[None, :, None]*step
        targets = position[:, None, None]+offsets[None]
        distance = np.linalg.norm(np.maximum(bounds[0]-targets, 0)+np.maximum(targets-bounds[1], 0), axis=-1)
        assert distance.min() > 10*step, "Every derivative target must be outside the active-source bounding box"
        values = linear.evaluate(targets.reshape(-1, 3), coefficient).reshape(len(position), 3, 4, 2, 3)
        jacobian = ((values[:, :, 0]-values[:, :, 1])/(2*step)).transpose(0, 2, 1, 3)
        jacobian_half = ((values[:, :, 2]-values[:, :, 3])/step).transpose(0, 2, 1, 3)
        curl, curl_half = curl_affine(jacobian), curl_affine(jacobian_half)
        divergence, divergence_half = np.trace(jacobian, axis1=-2, axis2=-1), np.trace(jacobian_half, axis1=-2, axis2=-1)
        np.testing.assert_allclose(jacobian, jacobian_half, rtol=0, atol=1e-6)

        # Independent divergence-theorem check on the raw piecewise affine field.
        density = gamma/linear.volume[None, :, None]
        centre = native.triangles.mean(axis=1)
        jump = density[:, native.owners]+np.einsum("ti,stij->stj", centre-linear.centroid[native.owners], gradient[:, native.owners])
        jump[:, interior] -= density[:, native.neighbours[interior]]+np.einsum(
            "ti,stij->stj", centre[interior]-linear.centroid[native.neighbours[interior]], gradient[:, native.neighbours[interior]])
        jump_normal = np.einsum("sti,ti->st", jump, native.normals)
        triangle_area = np.linalg.norm(np.cross(native.triangles[:, 1]-native.triangles[:, 0],
                                                 native.triangles[:, 2]-native.triangles[:, 0]), axis=1)/2
        raw_divergence = np.trace(gradient, axis1=-2, axis2=-1)
        volume_integral = raw_divergence @ linear.volume
        face_integral = jump_normal @ triangle_area
        np.testing.assert_allclose(volume_integral, face_integral, rtol=0, atol=1e-11)
        rows = []
        for k, kind in enumerate(("point", "cell_average")):
            difference = curl_half[:, k]-curl[:, k]
            row = {"input": kind, "baseline": str(source["source_names"][base[k]]),
                   "updated": str(source["source_names"][updated[k]]),
                   "exterior_curl_rms": rms(curl_half[:, k]),
                   "exterior_curl_maximum_norm": float(np.max(np.linalg.norm(curl_half[:, k], axis=1))),
                   "curl_step_halving_difference_rms": rms(difference),
                   "curl_step_halving_maximum_difference": float(np.max(np.abs(difference))),
                   "velocity_divergence_rms": float(np.sqrt(np.mean(divergence_half[:, k]**2))),
                   "jacobian_step_halving_maximum_difference": float(np.max(np.abs(jacobian[:, k]-jacobian_half[:, k]))),
                   "raw_volume_divergence_L1": float(np.abs(raw_divergence[k]) @ linear.volume),
                   "raw_normal_jump_L1": float(np.abs(jump_normal[k]) @ triangle_area),
                   "raw_divergence_theorem_difference": float(abs(volume_integral[k]-face_integral[k])),
                   "sides": {name: {"curl_rms": rms(curl_half[side == i, k]),
                                    "curl_step_halving_difference_rms": rms(difference[side == i])}
                             for i, name in enumerate(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"))}}
            rows.append(row)
        name = directory.name.removeprefix("cube-3d-shared-trace-sources-").removesuffix("-qualified")
        field_path = args.output / (name+"-curl-fields.npz")
        np.savez_compressed(field_path, position=position, faces=faces, side=side, support_bounds=bounds,
                            step=step, targets=targets, velocity=values, jacobian=jacobian, jacobian_half=jacobian_half,
                            curl=curl, curl_half=curl_half, divergence=divergence, divergence_half=divergence_half,
                            gamma_change=gamma, moment_change=moment, raw_divergence=raw_divergence,
                            raw_normal_jump=jump_normal, active_cells=np.flatnonzero(active))
        experiment = {"source": label(directory), "small_fvm_cells": mesh["n_cells"],
                      "particle_spacing": report["particle_spacing"], "physical_time": report["physical_time"],
                      "targets": len(position), "velocity_evaluations": int(np.prod(targets.shape[:-1])),
                      "derivative_step": step, "minimum_distance_from_source_bounding_box": float(distance.min()),
                      "support_bounds": bounds.tolist(), "records": rows,
                      "fields": {"path": label(field_path), "sha256": digest(field_path)}}
        experiments.append(experiment)
        print(json.dumps(experiment, indent=2), flush=True)
    sources = [{"path": label(path), "sha256": digest(path)} for path in dict.fromkeys(paths)]
    for path in dict.fromkeys(paths):
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    result = {"schema": "openonda-shared-trace-curl-3d/1", "status": "complete", "spatial_dimensions": 3,
              "experiments": experiments, "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": [
                  "This measures only the shared-update minus its matching native-face-moment baseline, using the same frozen physical sources.",
                  "Gaussian complements are identical and cancel; the body source-potential response has zero continuous curl outside the body.",
                  "The raw correction is exactly zero at all derivative targets, which lie outside its active-cell bounding box.",
                  "A nonzero exterior curl demonstrates a nonlocal solenoidal projection of this affine source correction. It does not prove this is the dominant coupling error.",
                  "Central differences at two step sizes check numerical stability; 24 deterministic face centres are a diagnostic sample, not a whole-boundary norm.",
                  "The coarse and medium physical seeds have different viscosity closures and are not a mesh-convergence study."]}
    (args.output / "shared-trace-curl-3d.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source = [path.resolve() for path in args.source]
    args.output = args.output.resolve()
    run(args)
