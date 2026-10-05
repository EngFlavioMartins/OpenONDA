"""Measure a bounded cylinder run in an isolated case, using its actual mesh.

Run under an external memory-limited cgroup when reproducing an OOM. This
verification entry point never writes to the tutorial's output directories.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import resource
import time

import numpy as np

from openonda import coupler
from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def run(options):
    module = load_case_module(CASE)
    overrides = {}
    for name, value in (
        ("cores", options.cores),
        ("particle_limit", options.capacity),
        ("compute_device", options.device),
    ):
        if value is not None:
            overrides[name] = value
    flow, particles, exchange, _mesh = module.build_case(overrides=overrides)
    seed = partial(
        module.cylinder_initial_velocity,
        freestream_velocity=module.STARTUP_FREESTREAM_VELOCITY,
        **module.INITIAL_PERTURBATION,
    )
    started = time.perf_counter()
    with coupler.create_coupler(
        flow, particles, exchange, mesh=options.mesh, case_dir=options.directory
    ) as driver:
        accepted = driver.run(
            start_from=options.start_from,
            max_coupling_steps=options.exchanges,
            backup_at_stop=True,
            initial_velocity=seed,
        )
        native = driver.fvm_solver
        rank = native.parallel.rank
        assert np.isfinite(native.velocity).all()
        assert np.isfinite(native.kinematic_pressure).all()
        report = {
            "rank": rank,
            "cores": flow.cores,
            "capacity": particles.numerics.max_n_particles,
            "accepted_step": accepted,
            "accepted_time": native.time,
            "cells": native.mesh_data["n_cells"],
            "peak_rank_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "wall_seconds": time.perf_counter() - started,
        }
        report["vpm"] = driver.apply_vpm(
            lambda solver: {
                "backend": solver.compute_device,
                "particles": solver.particles.n_particles_total,
                "time": solver.time,
                "step": solver.step,
                "position_finite": bool(np.isfinite(solver.particle_position).all()),
                "strength_finite": bool(np.isfinite(solver.particle_vortex_strength).all()),
                "one_plane": bool(np.all(solver.particle_position[:, 2] == 0)),
            }
        )
        (options.directory / f"memory-rank-{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    if rank == 0:
        records = [
            json.loads(line)
            for line in (options.directory / "solution/coupler_diagnostics.jsonl")
            .read_text()
            .splitlines()
        ]
        assert all(record["interface_iteration"]["converged"] for record in records)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--exchanges", type=int, default=25)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--device")
    parser.add_argument("--start-from", default="initial")
    run(parser.parse_args())
