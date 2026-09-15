"""Run an isolated ring control without deleting the tutorial's saved results."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from openonda.tutorial_runner import load_case_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "control",
        choices=(
            "baseline",
            "particle_splitting",
            "half_dt",
            "finer_spacing",
            "tighter_tree",
            "no_remesh",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-time", type=float, default=2.25)
    parser.add_argument("--device", choices=("CPU", "AUTO"), default="CPU")
    args = parser.parse_args()
    if args.end_time <= 0:
        parser.error("end-time must be positive")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "solution").exists():
        parser.error("output already contains a solution; choose a new directory")
    tutorial = Path(__file__).resolve().parents[1]
    setup = load_case_module(tutorial)
    if args.control == "finer_spacing":
        setup.PARTICLE_SPACING = 0.04
        # Refine quadrature while retaining the same Gaussian representation.
        # The toroidal constructor's core ratio must reflect the fixed width.
    case = setup.build_case(
        "particle_splitting" if args.control == "particle_splitting" else "baseline",
        compute_device=args.device,
    )
    if args.control == "finer_spacing":
        rings = tuple(
            replace(
                ring,
                distribution=replace(
                    ring.distribution,
                    core_radius_ratio=setup.PARTICLE_CORE_RADIUS / setup.PARTICLE_SPACING,
                ),
            )
            for ring in case.initial_conditions
        )
        case = replace(case, initial_conditions=rings)
    numerics = case.numerics
    if args.control == "finer_spacing":
        numerics = replace(
            numerics,
            stabilization=replace(
                numerics.stabilization, regularization_grid_spacing=setup.PARTICLE_SPACING
            ),
        )
    if args.control == "half_dt":
        numerics = replace(
            numerics,
            time_step_size=numerics.time_step_size / 2,
            stabilization=replace(
                numerics.stabilization,
                regularization_interval_steps=2
                * numerics.stabilization.regularization_interval_steps,
            ),
        )
    elif args.control == "tighter_tree":
        numerics = replace(numerics, induction=replace(numerics.induction, theta=0.3))
    elif args.control == "no_remesh":
        numerics = replace(
            numerics, stabilization=replace(numerics.stabilization, regularization_interval_steps=0)
        )
    steps = round(args.end_time / numerics.time_step_size)
    case = replace(
        case,
        name=args.control,
        directory=directory,
        numerics=numerics,
        backup=replace(case.backup, directory="solution", log_directory="solution"),
        samplers=replace(case.samplers, directory="diagnostics"),
        run=replace(case.run, steps=steps),
    )
    root = tutorial.parents[2]
    paths = [Path(__file__), tutorial / "setup.py"]
    paths += sorted((root / "source/solvers/vpm").rglob("*.py"))
    (directory / "source_hashes.json").write_text(
        json.dumps(
            {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
            indent=2,
        )
        + "\n"
    )
    setup.vpm.VPMSolver(case).run()


if __name__ == "__main__":
    main()
