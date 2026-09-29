"""Check repeated CUDA induction after restoring a copied Lamb–Oseen DVH case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import taichi as ti

from openonda.tutorial_runner import load_case_module


def process_vram_mib() -> float | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    for row in result.stdout.splitlines():
        pid, memory = (value.strip() for value in row.split(","))
        if int(pid) == os.getpid():
            return float(memory)
    return None


def check(directory: Path) -> dict:
    case = load_case_module(directory)
    constructor = case.vpm.VPMSolver
    solvers = []

    def capture(*args, **kwargs):
        solver = constructor(*args, **kwargs)
        # Force state restoration even when the tutorial's target is complete.
        solver.run = lambda *, start_from: solver.start_from(start_from)
        solvers.append(solver)
        return solver

    case.vpm.VPMSolver = capture
    try:
        case.run_case("vortex", "DVH", compute_device="CUDA")
    finally:
        case.vpm.VPMSolver = constructor
    solver = solvers[0]
    try:
        from source.solvers.vpm.config import constants

        assert constants.TAICHI_BACKEND == "CUDA", constants.TAICHI_BACKEND
        samples = []
        baseline = None
        tree_ids = set()
        for _ in range(8):
            solver.stepper._update_velocity_and_gradients(announce=False)
            ti.sync()
            velocities = solver.particles.velocity_cpu(use_cache=False).copy()
            assert np.isfinite(velocities).all()
            if baseline is None:
                baseline = velocities
            else:
                np.testing.assert_allclose(velocities, baseline, rtol=1e-6, atol=1e-7)
            tree_ids.add(id(solver.physics._treecode))
            samples.append(process_vram_mib())
        assert len(tree_ids) == 1, "Repeated evaluation replaced the tree workspace"
        measured = [value for value in samples if value is not None]
        assert measured, "NVIDIA process memory unavailable"
        assert max(measured[1:]) - min(measured[1:]) <= 8, samples
        checkpoint = max(
            (directory / "solution/vortex_dvh/vpm").glob("vpm_*.h5"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        with checkpoint.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "case": "vortex_dvh",
            "backend": "CUDA",
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
            "restored_step": solver.step,
            "restored_time": solver.time,
            "particles": solver.particles.n_particles_total,
            "tree_source_capacity": solver.physics._treecode.max_n_particles,
            "evaluations": len(samples),
            "process_vram_mib": samples,
            "tree_workspaces": len(tree_ids),
            "repeated_velocity_agreement": True,
        }
    finally:
        solver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(args.case_directory)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
