#!/usr/bin/env python3
"""Independently check volume Gaussian integrals using a closed surface flux.

The actual cropped boundary is warped. Integrate its native fan triangles;
only the solid cube has an exact rectangular-box Gaussian integral.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import load_npz
from scipy.special import erf, roots_jacobi

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file


def gaussian_surface_flux(triangles, position, radius, *, order, axis=0):
    """Integrate a Gaussian antiderivative through oriented closed triangles.

    H_axis = erf((x_axis-xp_axis)/sigma)/2 times the two transverse 1D
    Gaussians, with other H components zero. Then div(H) = zeta in 3D.
    This surface rule is independent of the volume tetrahedron quadrature.
    No radial cutoff is valid for H, whose erf factor has a nonzero far limit.
    """
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    position = np.asarray(position, dtype=float).reshape(-1, 3)
    radius = np.broadcast_to(np.asarray(radius, dtype=float), (len(position),))
    if order < 1 or axis not in (0, 1, 2) or np.any(radius <= 0):
        raise ValueError("Positive quadrature order/radii and a coordinate axis are required")
    rules = [roots_jacobi(order, alpha, 0) for alpha in (1, 0)]
    nodes = [(x + 1) / 2 for x, _ in rules]
    weights = [w / 2**(alpha + 1) for (_, w), alpha in zip(rules, (1, 0), strict=True)]
    r, s = np.meshgrid(*nodes, indexing="ij")
    a, b = np.meshgrid(*weights, indexing="ij")
    barycentric = np.column_stack((r.ravel(), ((1-r)*s).ravel(), ((1-r)*(1-s)).ravel()))
    reference = triangles.mean(axis=(0, 1))
    triangles = triangles - reference
    position = position - reference
    vector_area = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
    nonzero = vector_area[:, axis] != 0
    points = np.einsum("qv,tvd->tqd", barycentric, triangles[nonzero]).reshape(-1, 3)
    weight = (vector_area[nonzero, axis, None] * (2*a*b).ravel()).ravel()
    transverse = [i for i in range(3) if i != axis]
    result = np.zeros(len(position))
    for start in range(0, len(points), 128):
        delta = (points[start:start+128, None, :] - position) / radius[None, :, None]
        field = (0.5 * erf(delta[:, :, axis])
                 * np.exp(-delta[:, :, transverse[0]]**2 - delta[:, :, transverse[1]]**2)
                 / (np.pi * radius**2))
        result += weight[start:start+128] @ field
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), args.integrals / "cell-integral-inputs.npz",
             args.integrals / "cell-integrals.npz", args.mesh, args.failure]
    sources = [hash_file(p) for p in paths]
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2) + "\n")
    started = time.perf_counter()
    inputs = np.load(args.integrals / "cell-integral-inputs.npz", allow_pickle=False)
    matrix = load_npz(args.integrals / "cell-integrals.npz")
    mesh = load_native_mesh(args.mesh)
    outer = next(p for p in mesh["boundary"] if p["name"] == "numericalBoundary")
    wall = next(p for p in mesh["boundary"] if p["name"] == "cube")
    triangles = []
    for face in range(outer["start_face"], outer["start_face"] + outer["n_faces"]):
        vertices = mesh["vertex_position"][mesh["faces"][face]]
        triangles.append(np.stack((np.broadcast_to(vertices.mean(axis=0), vertices.shape),
                                   vertices, np.roll(vertices, -1, axis=0)), axis=1))
    triangles = np.concatenate(triangles)
    for face in range(wall["start_face"], wall["start_face"] + wall["n_faces"]):
        vertices = mesh["vertex_position"][mesh["faces"][face]]
        assert np.all(np.abs(vertices) <= 0.5 + 1e-13)
        assert np.any(np.all(np.abs(vertices-0.5) < 1e-13, axis=0)
                      | np.all(np.abs(vertices+0.5) < 1e-13, axis=0))
    position, radius = inputs["position"], inputs["radius"]
    body = np.prod((erf((0.5-position)/radius[:, None])
                    - erf((-0.5-position)/radius[:, None])) / 2, axis=1)
    surface8 = gaussian_surface_flux(triangles, position, radius, order=8)
    print(json.dumps({"event": "surface_order_8_complete", "elapsed_seconds": time.perf_counter()-started}), flush=True)
    surface10 = gaussian_surface_flux(triangles, position, radius, order=10)
    selected = np.linspace(0, len(position)-1, 32, dtype=int)
    other_axis = gaussian_surface_flux(triangles, position[selected], radius[selected], order=10, axis=1)
    volume = np.asarray(matrix.sum(axis=0)).ravel()
    refined_difference = float(np.max(np.abs(surface10-surface8)))
    axis_difference = float(np.max(np.abs(surface10[selected]-other_axis)))
    volume_difference = float(np.max(np.abs(volume-(surface10-body))))
    assert max(refined_difference, axis_difference, volume_difference) < 2e-9
    with np.load(args.failure, allow_pickle=False) as data:
        np.testing.assert_array_equal(position[:len(data["solve_prior"])], data["solve_position"])
        magnitude = np.linalg.norm(data["solve_prior"], axis=1)
    n = len(magnitude)
    record = {"schema": "openonda-cell-integral-region-audit-3d/1", "spatial_dimensions": 3,
              "cells": matrix.shape[0], "sources_count": matrix.shape[1], "outer_faces": outer["n_faces"],
              "outer_triangles": len(triangles), "solid_cube_half_width": 0.5,
              "maximum_surface_quadrature_8_vs_10_difference": refined_difference,
              "maximum_surface_coordinate_axis_difference": axis_difference,
              "maximum_volume_vs_surface_integral_difference": volume_difference,
              "fan_volume_sum": float(inputs["polyhedron_volume"].sum()),
              "fvm_volume_sum": float(inputs["fvm_volume"].sum()),
              "prior_strength_magnitude_sum": float(magnitude.sum()),
              "prior_individual_gaussian_envelope_fraction_in_body": float(magnitude @ body[:n] / magnitude.sum()),
              "prior_individual_gaussian_envelope_fraction_in_fluid_region": float(magnitude @ volume[:n] / magnitude.sum()),
              "prior_individual_gaussian_envelope_fraction_outside_cut": float(magnitude @ (1-surface10[:n]) / magnitude.sum()),
              "limitations": ["The surface flux integrates an antiderivative of the raw Gaussian, not the particle velocity.",
                              "Envelope fractions sum separately weighted particle envelopes; they do not measure the norm of summed vector vorticity or its induced velocity curl."],
              "elapsed_seconds": time.perf_counter()-started, "sources": sources}
    np.savez_compressed(args.output / "region-integrals.npz", volume=volume, surface=surface10,
                        body=body, position=position, radius=radius)
    (args.output / "cell-integral-region-audit-3d.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    results = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--integrals", type=Path, default=results / "cube-3d-cell-integral-reconstruction")
    parser.add_argument("--mesh", type=Path, default=results / "cube-3d-oracle/small-native-mesh.npz")
    parser.add_argument("--failure", type=Path, default=results / "cube-3d-frozen-projected-guard/hybrid/renewal_projection_failure_oracle.npz")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
