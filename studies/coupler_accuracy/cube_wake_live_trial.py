"""Bounded fresh cube trial using a frozen source tree and the unchanged mesh.

The solver owns MPI launch and output. This harness changes the output
directory and requested stopping point, with an optional storage capacity
limit that leaves physical operators unchanged. Algorithm variants are
explicit source-tree snapshots, recorded beside the results.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from cube_wake_particle_probe import load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--particle-capacity", type=int)
    args = parser.parse_args()
    source = args.source_tree.resolve()
    output = args.output.resolve()
    baseline = args.baseline.resolve()
    case = load_case(source)
    vpm = replace(case.VPM_CASE, directory=output)
    if args.particle_capacity is not None:
        vpm = replace(
            vpm,
            numerics=replace(
                vpm.numerics,
                max_n_particles=args.particle_capacity,
                max_evaluation_points=args.particle_capacity,
            ),
        )
    with case.coupling.create_coupler(
        case.FVM_SETUP,
        vpm,
        case.COUPLER_SETUP,
        mesh=baseline / "solution/coupled/mesh.npz",
        require_empty_output=True,
    ) as solver:
        solver.initialize()
        if solver.fvm_solver.parallel.is_root:
            files = {
                str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for base in ("source", "openonda")
                for p in (source / base).rglob("*.py")
            }
            (output / "trial.json").write_text(
                json.dumps(
                    {
                        "source_tree": str(source),
                        "base_commit": (source / "SOURCE_REVISION").read_text().strip(),
                        "source_changes": (source / "STUDY_CHANGE.txt").read_text()
                        if (source / "STUDY_CHANGE.txt").exists()
                        else "none",
                        "source_sha256": files,
                        "mesh_sha256": hashlib.sha256(
                            (baseline / "solution/coupled/mesh.npz").read_bytes()
                        ).hexdigest(),
                        "requested_steps": args.steps,
                        "fvm_cores": case.FVM_SETUP.cores,
                        "particle_storage_capacity": vpm.numerics.max_n_particles,
                        "capacity_policy": "Allocation ceiling changed; particle spacing, GBD grid limit, physical operators and thresholds unchanged. Verify that population caps were never activated before accepting the comparison.",
                    },
                    indent=2,
                )
                + "\n"
            )
        solver.run(max_coupling_steps=args.steps, backup_at_stop=True)


if __name__ == "__main__":
    main()
