#!/usr/bin/env python3
"""Replay a cube checkpoint with an explicit, audited vorticity-relaxation trial.

The checkpoint and tutorial outputs are read-only inputs. Each trial uses a new
output directory, the saved mesh, all four FVM ranks, and the original health
limits. This is a continuation experiment, not a replacement for a fresh run
or a spatial/time convergence study.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

import openonda.coupler as coupling
import openonda.vpm as vpm

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/02_cube_flow"


def particle_health(solver) -> dict:
    """Observe already-refreshed fields without another induction evaluation."""
    gradient = np.asarray(solver.particle_velocity_gradient, dtype=np.float64)
    strain = 0.5 * (gradient + gradient.transpose(0, 2, 1))
    increments = solver.time_step_size * np.abs(strain).sum(axis=2).max(axis=1)
    index = int(np.argmax(increments))
    strength = np.linalg.norm(solver.particle_vortex_strength, axis=1)
    return {
        "step": int(solver.step),
        "time": float(solver.time),
        "time_step_size": float(solver.time_step_size),
        "particles": int(len(strength)),
        "strain_increment_infinity": float(increments[index]),
        "limiting_particle": index,
        "limiting_position": solver.particle_position[index].tolist(),
        "maximum_speed": float(np.linalg.norm(solver.particle_velocity, axis=1).max()),
        "maximum_vorticity": float((strength / solver.particle_volume).max()),
        "particle_enstrophy": float(np.sum(strength.astype(float) ** 2 / solver.particle_volume)),
        "relaxation_moment_transfer": solver.stabilization.pedrizzetti_moment_transfer.tolist(),
    }


def run(args) -> None:
    source_paths = [
        Path(__file__).resolve(),
        CASE / "setup.py",
        ROOT / "source/coupler/boundary.py",
        ROOT / "source/solvers/vpm/config/health.py",
        ROOT / "source/solvers/vpm/core/evolution.py",
        ROOT / "source/solvers/vpm/physics/diffusion/grid.py",
        ROOT / "source/solvers/vpm/stabilization/manager.py",
        ROOT / "source/solvers/vpm/stabilization/operators.py",
    ]
    sources = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    setup = runpy.run_path(str(CASE / "setup.py"))
    manifest_path = args.checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Reconstruct the saved stabilization policy before strict restart checks.
    # The subsequent trial changes only the explicitly requested relaxation.
    saved_stabilization = manifest["config"]["vpm"]["stabilization"]
    baseline = setup["VPM_CASE"]
    scalar_policy = {
        key: value
        for key, value in saved_stabilization.items()
        if key not in {"filament_refinement", "divergence_relaxation"}
    }
    numerics = replace(
        baseline.numerics,
        stabilization=replace(baseline.numerics.stabilization, **scalar_policy),
    )
    baseline = replace(baseline, numerics=numerics)
    with coupling.create_coupler(
        setup["FVM_SETUP"],
        baseline,
        setup["COUPLER_SETUP"],
        mesh=CASE / "constant/mesh.npz",
        case_dir=args.output,
        require_empty_output=True,
    ) as driver:
        driver.initialize()
        start = driver.load_backup(args.checkpoint)

        def instrument(solver):
            factor = args.relaxation_rate * solver.time_step_size
            if not 0.0 <= factor <= 1.0:
                raise ValueError("relaxation_rate * time_step_size must lie in [0, 1]")
            if factor:
                policy = replace(
                    solver.stabilization_config,
                    pedrizzetti_relaxation_factor=factor,
                    pedrizzetti_relaxation_preserve_moments=True,
                )
                # All storage exists with relaxation disabled. Publish the
                # experimental policy to each owner, including backup identity.
                solver.stabilization_config = policy
                solver.stabilization.config = policy
                solver.stabilization.ctx = replace(solver.stabilization.ctx, config=policy)
                solver.setup = replace(solver.setup, stabilization=policy)
                solver.numerics = solver.setup
                solver.case = replace(solver.case, numerics=solver.setup)
            record = {
                "sources_sha256": sources,
                "checkpoint": str(args.checkpoint),
                "checkpoint_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "relaxation_rate_per_second": args.relaxation_rate,
                "relaxation_factor_per_step": factor,
                "preserve_moments": bool(factor),
                "maximum_steps": args.steps,
                "status": "running",
            }
            (args.output / "trial.json").write_text(json.dumps(record, indent=2) + "\n")
            original = solver.execute_scheduled_samplers

            def observe():
                failure = None
                try:
                    original()
                except vpm.HealthError as exc:
                    if not exc.restartable:
                        raise
                    failure = exc
                row = particle_health(solver)
                row["error"] = str(failure) if failure else None
                with (args.output / "health.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print("HEALTH " + json.dumps(row), flush=True)
                if failure is not None:
                    np.savez_compressed(
                        args.output / "failed-particle-state.npz",
                        position=solver.particle_position,
                        vortex_strength=solver.particle_vortex_strength,
                        velocity_gradient=solver.particle_velocity_gradient,
                        particle_volume=solver.particle_volume,
                        core_radius=solver.particle_core_radius,
                    )
                    raise failure

            solver.execute_scheduled_samplers = observe

        driver.apply_vpm(instrument)
        status = "failed"
        try:
            driver.solve(start_step=start, max_coupling_steps=args.steps, backup_at_stop=True)
            status = "completed"
        finally:
            if driver._is_master:
                path = args.output / "trial.json"
                record = json.loads(path.read_text())
                record["status"] = status
                path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CASE / "solution/backups")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--relaxation-rate", type=float, default=0.0)
    arguments = parser.parse_args()
    if (
        arguments.steps < 1
        or not np.isfinite(arguments.relaxation_rate)
        or arguments.relaxation_rate < 0
    ):
        parser.error("steps must be positive and relaxation-rate finite and nonnegative")
    arguments.checkpoint = arguments.checkpoint.resolve()
    arguments.output = arguments.output.resolve()
    run(arguments)
