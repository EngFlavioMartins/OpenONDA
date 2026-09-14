#!/usr/bin/env python3
"""Test f64 auxiliary body queries during the scoped 3D interface iteration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.vpm.boundary_elements.panels.kernels.induced_velocity import (
    compute_source_induced_velocity_kernel,
)
from source.solvers.vpm.core.solver import VPMSolver
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_interface_iteration_3d import run as run_iteration


@contextmanager
def precise_auxiliary_body_queries(directory, record):
    """Promote only the auxiliary source-panel target evaluation used by differences."""
    original = VPMSolver._nonparticle_target_velocity

    def evaluate(solver, evaluation_position):
        if solver._body_induced_fn is None:
            return original(solver, evaluation_position)
        panel = solver.panel_solver
        if (solver.n_sources or solver.vlm_solver is not None or panel is None
                or panel.boundary_condition_type != "NEUMANN" or panel.float_dtype != "f32"
                or solver._pressure_body_induced_fn != panel.compute_induced_velocity):
            raise ValueError("This study requires the unmodified f32 Neumann panel body callback, without auxiliary sources or VLM")
        panel._ensure_initialized()
        count = panel.lattice.n_panels
        if count != 108 or count >= panel.far_field_min_panels:
            raise ValueError("The qualified study uses the direct 108-panel cube")
        points = np.ascontiguousarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
        vertices, normals, strength = (np.ascontiguousarray(field.to_numpy()[:count], dtype=np.float64) for field in (
            panel.lattice.vertex_position, panel.lattice.normal, panel.lattice.source_strength))
        value = np.zeros_like(points)
        compute_source_induced_velocity_kernel(vertices, normals, strength, points, value)
        assert np.all(np.isfinite(value)) and value.dtype == np.float64
        record["auxiliary_calls"] += 1
        record["auxiliary_points"] += len(points)
        if "first_query" not in record:
            old = original(solver, evaluation_position)
            path = directory / "first-auxiliary-query.npz"
            np.savez_compressed(path, position=points, vertices=vertices, normal=normals, strength=strength,
                                original_velocity=old, promoted_velocity=value)
            record["first_query"] = hash_file(path)
            record["first_query_velocity_difference_rms"] = float(np.sqrt(np.mean(np.sum((old-value)**2, axis=1))))
        return value

    VPMSolver._nonparticle_target_velocity = evaluate
    try:
        yield
    finally:
        VPMSolver._nonparticle_target_velocity = original


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    qualification_path = ROOT / "studies/coupler_accuracy/results/cube-3d-panel-derivative-precision-coarse-qualified/panel-derivative-precision-3d.json"
    qualification = json.loads(qualification_path.read_text())
    assert qualification["status"] == "complete" and qualification["spatial_dimensions"] == 3
    assert qualification["quadrature_order_check_maximum_difference"] < 2e-11
    for row in qualification["sources"]:
        assert hash_file(ROOT / row["path"]) == row
    paths = [Path(__file__).resolve(), ROOT / "source/solvers/vpm/core/solver.py",
             ROOT / "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
             ROOT / "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py", qualification_path]
    sources = [hash_file(path) for path in paths]
    archives = {path: path.read_bytes() for path in paths if path.suffix == ".py"}
    record = {"schema": "openonda-panel-target-precision-3d/1", "status": "running", "spatial_dimensions": 3,
              "mode": "f64_auxiliary_panel_queries", "stored_panel_precision": "f32", "particle_precision": "f32",
              "auxiliary_calls": 0, "auxiliary_points": 0, "sources": sources,
              "limitations": ["Only the auxiliary panel query used by target finite differences is promoted. Stored geometry/strengths, panel solve, complete point velocity and particle stage body operations retain their existing precision.",
                              "The same fixed-predictor interval and first-map replay checks apply. Reference fields never enter the hybrid boundary or renewal.",
                              "This is a scoped 108-panel serial cube experiment, not a production precision option or a developed-wake accuracy claim."]}
    started = time.perf_counter()
    try:
        with precise_auxiliary_body_queries(args.output, record):
            run_iteration(args)
        assert record["auxiliary_calls"] > 0
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter()-started
        if args.output.exists():
            for path, content in archives.items():
                target = args.output / "precision-sources" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            child = args.output / "interface-iteration-3d.json"
            if child.exists():
                record["interface_report"] = hash_file(child)
            (args.output / "panel-target-precision-3d.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({key: record[key] for key in ("status", "auxiliary_calls", "first_query_velocity_difference_rms", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--particle-spacing", type=float, default=.0625)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--relaxation", type=float, default=1.)
    parser.add_argument("--normal-tolerance", type=float, default=1e-6)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.steps < 1 or args.iterations < 0 or args.particle_spacing <= 0:
        parser.error("Positive steps and spacing, and a nonnegative sweep count, are required")
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    run(args)
