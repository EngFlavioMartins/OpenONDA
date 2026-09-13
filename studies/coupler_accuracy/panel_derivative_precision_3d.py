#!/usr/bin/env python3
"""Isolate target-derivative precision on a saved fully 3D cube panel field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import load_stl
from source.solvers.vpm.boundary_elements.panels.kernels.induced_velocity import (
    compute_source_induced_velocity_kernel,
)
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays


def rms(value, area):
    return float(np.sqrt(np.average(np.sum(value*value, axis=-1), weights=area)))


def tangent(value, normal):
    return value-np.sum(value*normal, axis=-1)[..., None]*normal


def quadrature(vertices, strengths, points, normals, order):
    """Integrate R/r^3 and its analytic target directional derivative over area."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    a, b = np.meshgrid((nodes+1)/2, (nodes+1)/2, indexing="ij")
    weight = np.outer(weights/2, weights/2)*(1-a)
    edge1, edge2 = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    source = (vertices[:, None, 0]+a.ravel()[None, :, None]*edge1[:, None]
              +(1-a).ravel()[None, :, None]*b.ravel()[None, :, None]*edge2[:, None]).reshape(-1, 3)
    jacobian = np.linalg.norm(np.cross(edge1, edge2), axis=1)
    amplitude = (strengths[:, :, None]*jacobian[None, :, None]*weight.ravel()[None, None]).reshape(len(strengths), -1)/(4*np.pi)
    velocity, derivative = (np.empty((len(strengths), len(points), 3)) for _ in range(2))
    for start in range(0, len(points), 32):
        stop = min(start+32, len(points))
        r = points[start:stop, None]-source[None]
        radius2 = np.sum(r*r, axis=-1)
        assert radius2.min() > .5**2
        inverse3 = radius2**-1.5
        kernel = r*inverse3[..., None]
        velocity[:, start:stop] = np.einsum("qsc,ks->kqc", kernel, amplitude)
        direction = normals[start:stop, None]
        derivative_kernel = direction*inverse3[..., None]-3*r*(np.sum(r*direction, axis=-1)*inverse3/radius2)[..., None]
        derivative[:, start:stop] = np.einsum("qsc,ks->kqc", derivative_kernel, amplitude)
    return velocity, tangent(derivative, normals)


def evaluate(vertices, normal, strengths, points, dtype):
    vertices, normal, strengths, points = (np.ascontiguousarray(v, dtype=dtype) for v in (vertices, normal, strengths, points))
    result = np.empty((len(strengths), len(points), 3), dtype=dtype)
    for i, strength in enumerate(strengths):
        compute_source_induced_velocity_kernel(vertices, normal, strength, points, result[i])
    return result.astype(np.float64)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parent_path = args.source / "cube-moment-panel-resolution-3d.json"
    field_path = args.source / "moment-panel-resolution-fields.npz"
    parent, fields = json.loads(parent_path.read_text()), read_arrays(field_path)
    assert parent["status"] == "complete" and parent["spatial_dimensions"] == 3 and parent["panels"] == 108
    boundary_record = next(row for row in parent["sources"] if row["path"].endswith("/boundary-fields.npz"))
    boundary_path, surface_path = ROOT / boundary_record["path"], ROOT / parent["surface"]["path"]
    assert hash_file(boundary_path) == boundary_record and hash_file(surface_path) == parent["surface"]
    boundary = read_arrays(boundary_path)
    points, normal, area = boundary["position"], boundary["native_unit_normal"], boundary["native_vector_area"]
    cut = slice(*parent["target_slices"]["boundary"])
    np.testing.assert_array_equal(points, fields["position"][cut])
    vertices, _ = load_stl(str(surface_path))
    panel_normal = fields["panel_normal"]
    cross = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])
    np.testing.assert_allclose(cross/np.linalg.norm(cross, axis=1)[:, None], panel_normal, rtol=0, atol=1e-15)
    paths = [Path(__file__).resolve(), parent_path, field_path, boundary_path, surface_path]
    paths += [ROOT / name for name in (
        "source/solvers/vpm/boundary_elements/panels/geometry/stl_io.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
        "source/solvers/vpm/core/solver.py", "studies/coupler_accuracy/cube_boundary_oracle.py",
        "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            archive = args.output / "sources" / path.relative_to(ROOT)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(path.read_bytes())
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    replay = evaluate(vertices, panel_normal, fields["panel_strength"], points, np.float64)
    original = np.stack([fields[name+"__body_velocity"][cut] for name in parent["source_names"]])
    np.testing.assert_allclose(replay, original, rtol=0, atol=2e-12)
    replay_error = float(np.max(np.abs(replay-original)))
    # Both evaluation precisions now use exactly the same rounded panel field.
    vertices, panel_normal, strength = (v.astype(np.float32).astype(np.float64) for v in (vertices, panel_normal, fields["panel_strength"]))
    perturbed = np.nextafter(strength.astype(np.float32), np.float32(np.inf)).astype(np.float64)
    all_strength = np.vstack((strength, perturbed))
    q8, d8 = quadrature(vertices, all_strength, points, normal, 8)
    q12, d12 = quadrature(vertices, all_strength, points, normal, 12)
    quadrature_error = max(float(np.max(np.abs(q12-q8))), float(np.max(np.abs(d12-d8))))
    np.testing.assert_allclose(q8, q12, rtol=0, atol=2e-11)
    np.testing.assert_allclose(d8, d12, rtol=0, atol=2e-11)
    step = max(1e-6, 1e-3*args.particle_spacing)
    records, saved = [], {"position": points, "normal": normal, "area": area, "vertices": vertices,
                           "panel_normal": panel_normal, "strength": strength, "perturbed_strength": perturbed,
                           "quadrature_velocity": q12, "quadrature_tangential_gradient": d12}
    for runtime, modes in (("f64", (np.float64,)), ("f32", (np.float32, np.float64))):
        if runtime == "f32":
            ti.reset()
            ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1)
        for dtype in modes:
            name = f"runtime_{runtime}_evaluation_{np.dtype(dtype).name}"
            center = evaluate(vertices, panel_normal, all_strength, points, dtype)
            derivatives = []
            for scale in (1., .5):
                offset = step*scale*normal
                plus = evaluate(vertices, panel_normal, all_strength, points+offset, dtype)
                minus = evaluate(vertices, panel_normal, all_strength, points-offset, dtype)
                derivatives.append(tangent((plus-minus)/(2*step*scale), normal))
            saved[name+"__velocity"] = center
            saved[name+"__gradient"] = derivatives[0]
            saved[name+"__gradient_half"] = derivatives[1]
            for state, source_name in enumerate(parent["source_names"]):
                perturbed_state = state+len(strength)
                actual_response = derivatives[0][perturbed_state]-derivatives[0][state]
                expected_response = d12[perturbed_state]-d12[state]
                row = {"mode": name, "source_name": source_name,
                       "velocity_error_rms": rms(center[state]-q12[state], area),
                       "gradient_error_rms": rms(derivatives[0][state]-d12[state], area),
                       "gradient_half_step_error_rms": rms(derivatives[1][state]-d12[state], area),
                       "gradient_step_halving_change_rms": rms(derivatives[1][state]-derivatives[0][state], area),
                       "one_ulp_strength_response_rms": rms(actual_response, area),
                       "quadrature_one_ulp_strength_response_rms": rms(expected_response, area),
                       "one_ulp_strength_response_error_rms": rms(actual_response-expected_response, area)}
                records.append(row)
                print(json.dumps(row), flush=True)
    ti.reset()
    result_fields = args.output / "panel-derivative-precision-fields.npz"
    np.savez_compressed(result_fields, **saved)
    result = {"schema": "openonda-panel-derivative-precision-3d/1", "status": "complete", "spatial_dimensions": 3,
              "panels": len(vertices), "boundary_points": len(points), "particle_spacing": args.particle_spacing, "difference_step": step,
              "physical_seed_time": parent["physical_time"], "source_small_cells": parent["small_fvm_cells"],
              "original_body_replay_maximum_difference": replay_error, "quadrature_order_check_maximum_difference": quadrature_error,
              "records": records, "fields": hash_file(result_fields), "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Frozen coarse physical body strengths, with the same rounded geometry and strengths for both evaluation precisions; no body re-solve or evolving flow.",
                              "The independent reference differentiates the surface-integral kernel and compares tensor-product Duffy quadrature orders 8 and 12.",
                              "One-ULP strength perturbations diagnose operator sensitivity. They are not measured strength changes from an interface sweep.",
                              "This test does not by itself identify the source of the live iteration floor or demonstrate improved coupled forces or velocity profiles."]}
    (args.output / "panel-derivative-precision-3d.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particle-spacing", type=float, default=.125)
    args = parser.parse_args()
    if not np.isfinite(args.particle_spacing) or args.particle_spacing <= 0:
        parser.error("Positive finite particle spacing is required")
    args.source, args.output = args.source.resolve(), args.output.resolve()
    run(args)
