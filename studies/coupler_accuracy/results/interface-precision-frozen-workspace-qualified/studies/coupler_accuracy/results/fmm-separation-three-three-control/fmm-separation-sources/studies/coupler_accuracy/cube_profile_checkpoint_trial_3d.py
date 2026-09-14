#!/usr/bin/env python3
"""Save both accepted FVM states alongside the matched 3D profile observations."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from source.solvers.fvm.io.backup import save_backup
from studies.coupler_accuracy import cube_interface_profile_observer_3d as profiles
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_coupled_trial import MatchedComparison


@contextmanager
def accepted_checkpoints(args, record):
    original = MatchedComparison.measure

    def measure(observer, particle_solver, **kwargs):
        original(observer, particle_solver, **kwargs)
        tick = int(observer.small.step)
        if tick % args.profile_every_fvm_steps and tick != args.steps * args.substeps:
            return
        before = profiles.state_signature(observer, particle_solver)
        files = {}
        for name, solver in (("full", observer.full), ("hybrid", observer.small)):
            path = args.output / "accepted-fvm-checkpoints" / f"{name}-{tick:06d}.npz"
            save_backup(solver, path)
            files[name] = hash_file(path)
        assert profiles.state_signature(observer, particle_solver) == before
        if "meshes" not in record:
            record["meshes"] = {name: hash_file(args.output / "trial" / name / "solution/mesh.npz") for name in ("full", "hybrid")}
        comparison = observer.records[-1]
        record["frames"].append({
            "fvm_step": tick, "coupling_step": int(particle_solver.step),
            "physical_time": float(observer.seed_time + particle_solver.time),
            "checkpoints": files, "state_bitwise_unchanged": True,
            "state_fingerprints": before, "full_drag_coefficient": comparison["full_drag_coefficient"],
            "hybrid_drag_coefficient": comparison["hybrid_drag_coefficient"],
        })
        (args.output / "accepted-fvm-checkpoints-3d.json").write_text(json.dumps(record, indent=2) + "\n")

    MatchedComparison.measure = measure
    try:
        yield
    finally:
        MatchedComparison.measure = original


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [Path(__file__).resolve(), Path(profiles.__file__).resolve(), ROOT / "source/solvers/fvm/io/backup.py"]
    sources = [hash_file(path) for path in paths]
    record = {
        "schema": "openonda-accepted-fvm-checkpoints-3d/1", "status": "running", "spatial_dimensions": 3,
        "sources": sources, "frames": [], "limitations": [
            "Checkpoints observe the independent full FVM and hybrid FVM after accepted coupling intervals. Neither checkpoint feeds the hybrid.",
            "Both canonical FVM states are saved at every profile time, allowing independent force and field reconstruction.",
            "This does not add a coupled restart schedule or claim that the VPM state is fully restartable at every profile time.",
            "The underlying fixed-predictor and precision study remains serial, laminar, fully 3D and scoped to the 108-panel cube.",
        ],
    }
    try:
        with accepted_checkpoints(args, record):
            profiles.run(args)
        assert record["frames"][-1]["fvm_step"] == args.steps * args.substeps
        assert [hash_file(path) for path in paths] == sources
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if args.output.exists():
            for path in paths:
                target = args.output / "checkpoint-sources" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            report = args.output / "profile-observation-3d.json"
            if report.exists():
                record["profile_report"] = hash_file(report)
            (args.output / "accepted-fvm-checkpoints-3d.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--substeps", type=int, choices=(1, 5), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--profile-every-fvm-steps", type=int, default=25)
    parser.add_argument("--particle-spacing", type=float, default=.0625)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--relaxation", type=float, default=1.)
    parser.add_argument("--normal-tolerance", type=float, default=1e-6)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if (args.steps < 1 or args.iterations < 0 or args.particle_spacing != .0625
            or args.profile_every_fvm_steps < 1 or args.profile_every_fvm_steps % args.substeps):
        parser.error("Require positive steps, matched medium spacing and a positive profile interval divisible by substeps")
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    run(args)
