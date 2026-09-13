#!/usr/bin/env python3
"""Build and independently qualify the frozen cube's integrated velocity curl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import load_npz

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.native_cell_integrals_3d import NativeCellIntegration
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import (
    gaussian_velocity_curl_volume_integral,
    native_velocity_curl_cell_integrals,
)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = [Path(__file__), args.mesh, args.raw_integrals / "cell-integral-inputs.npz",
             args.raw_integrals / "cell-integrals.npz"]
    paths += [ROOT / path for path in (
        "studies/coupler_accuracy/native_velocity_curl_integrals_3d.py",
        "studies/coupler_accuracy/native_cell_integrals_3d.py",
        "studies/coupler_accuracy/joint_reconstruction_3d.py",
        "source/solvers/fvm/io/mesh_storage.py", "source/solvers/fvm/mesh/geometry.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.parent == Path(__file__).parent:
            (args.output / path.name).write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2) + "\n")
    mesh = load_native_mesh(args.mesh)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    with np.load(args.raw_integrals / "cell-integral-inputs.npz", allow_pickle=False) as data:
        inputs = {key: data[key].copy() for key in data.files}
    ids, position, radius = (inputs[key] for key in ("cell_ids", "position", "radius"))
    volume = geometry["cell_volume"][ids]
    np.testing.assert_array_equal(volume, inputs["fvm_volume"])
    matrix_path = args.output / "velocity-curl-integrals.npy"
    matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=float,
                                        shape=(3*len(ids), 3*len(position)))

    def progress(done, total, order, change):
        print(json.dumps({"event": "stokes_face_integration", "done": done, "total": total,
                          "order": order, "remainder_estimate": float(change),
                          "elapsed_seconds": time.perf_counter()-started}), flush=True)

    matrix, diagnostic = native_velocity_curl_cell_integrals(
        mesh, geometry, ids, position, radius, matrix=matrix, progress=progress)
    matrix.flush()
    raw = load_npz(args.raw_integrals / "cell-integrals.npz")
    blocks = matrix.reshape(len(ids), 3, len(position), 3)
    trace_error, symmetry_error = 0., 0.
    for start in range(0, len(ids), 32):
        local = slice(start, start+32)
        trace = np.einsum("cipi->cp", blocks[local])
        trace_error = max(trace_error, float(np.max(np.abs(trace-2*raw[local].toarray()) / volume[local, None])))
        difference = blocks[local] - blocks[local].transpose(0, 3, 2, 1)
        symmetry_error = max(symmetry_error, float(np.max(np.abs(difference) / volume[local, None, None, None])))
    assert max(trace_error, symmetry_error) < 2e-8
    selected = np.unique(np.r_[np.linspace(0, len(ids)-1, 16, dtype=int), np.argsort(volume)[-4:],
                               np.argsort(diagnostic["cell_summed_remainder_over_fvm_volume"])[-4:],
                               np.argsort(np.abs(inputs["polyhedron_volume"]/volume-1))[-4:]])
    integration = NativeCellIntegration.from_mesh(mesh, geometry, ids[selected])
    verification = []
    for local, cell in enumerate(selected):
        distance = np.linalg.norm(position-geometry["cell_centre"][ids[cell]], axis=1) / radius
        columns = np.unique(np.r_[np.argsort(distance)[:8], np.linspace(0, len(position)-1, 8, dtype=int)])
        args_volume = (integration, local, position[columns], radius[columns])
        lower = gaussian_velocity_curl_volume_integral(*args_volume, order=10)
        higher = gaussian_velocity_curl_volume_integral(*args_volume, order=12)
        block_columns = (3*columns[:, None]+np.arange(3)).ravel()
        stokes = matrix[3*cell:3*cell+3, block_columns]
        refinement = float(np.max(np.abs(higher-lower)) / volume[cell])
        difference = float(np.max(np.abs(stokes-higher)) / volume[cell])
        assert max(refinement, difference) < 2e-8
        verification.append({"local_cell": int(cell), "source_columns": columns.tolist(),
                             "volume_order_10_vs_12_difference_over_fvm_volume": refinement,
                             "stokes_vs_volume_difference_over_fvm_volume": difference})
        print(json.dumps({"event": "independent_volume_check", **verification[-1]}), flush=True)
    np.savez_compressed(args.output / "velocity-curl-integral-inputs.npz", **inputs)
    record = {"schema": "openonda-native-velocity-curl-integrals-3d/1", "status": "complete",
              "spatial_dimensions": 3, "cells": len(ids), "sources_count": len(position),
              "quadrature": diagnostic, "maximum_trace_minus_twice_raw_integral_over_fvm_volume": trace_error,
              "maximum_tensor_symmetry_defect_over_fvm_volume": symmetry_error,
              "independent_volume_checks": verification,
              "matrix": hash_file(matrix_path), "sources": sources,
              "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["This is the continuous Gaussian velocity curl integrated over native fluid cells, not the FVM's discrete curl applied to sampled particle velocity.",
                              "Source panels and uniform freestream have zero continuous curl in the fluid interior; their velocity contributions remain part of separate velocity and boundary checks.",
                              "All source tails are retained. Quadrature checks qualify this frozen observation, not evolving transfer accuracy."]}
    (args.output / "velocity-curl-integral-audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"event": "integrated_curl_complete", "elapsed_seconds": record["elapsed_seconds"],
                      "trace_error": trace_error, "symmetry_error": symmetry_error}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    results = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--raw-integrals", type=Path, default=results / "cube-3d-cell-integral-reconstruction")
    parser.add_argument("--mesh", type=Path, default=results / "cube-3d-oracle/full-native-mesh.npz")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
