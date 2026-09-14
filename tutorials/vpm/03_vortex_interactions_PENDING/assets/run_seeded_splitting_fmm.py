#!/usr/bin/env python3
"""Run the seeded CS/LES ring pair with filament splitting and CPU FMM.

Fresh initial conditions only. Supply the CPU thread limit externally.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import openonda.vpm as vpm
from openonda.tutorial_runner import load_case_module

setup_les = load_case_module(Path(__file__).resolve().parents[1], "assets.legacy_les")

CASE_NAME = "cs_breakdown_splitting_fmm_cpu_root_fresh"
PARTICLE_CAPACITY = 600_000


def build_case(*, steps: int = 1200, wall_minutes: float = 180, capacity: int = PARTICLE_CAPACITY):
    """Preserve the seeded tree case apart from induction, capacity and timing."""
    name = (
        CASE_NAME
        if capacity == PARTICLE_CAPACITY
        else f"cs_breakdown_splitting_fmm_cpu_root_{capacity}"
    )
    case = setup_les.build_case(
        "splitting",
        scenario="seeded_breakdown",
        compute_device="CPU",
        steps=steps,
        wall_minutes=wall_minutes,
        case_name=name,
    )
    numerics = case.numerics
    refinement = replace(
        numerics.stabilization.filament_refinement,
        max_n_particles=capacity,
    )
    return replace(
        case,
        numerics=replace(
            numerics,
            induction=vpm.FMMInduction(stretching_scheme="TRANSPOSED"),
            max_n_particles=capacity,
            stabilization=replace(numerics.stabilization, filament_refinement=refinement),
            diagnostics=replace(numerics.diagnostics, detailed_timing=True),
        ),
        run=replace(
            case.run,
            resource_limits=replace(
                case.run.resource_limits,
                max_particles=min(case.run.resource_limits.max_particles, capacity),
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--wall-minutes", type=float, default=180)
    parser.add_argument("--capacity", type=int, default=PARTICLE_CAPACITY)
    args = parser.parse_args()
    vpm.VPMSolver(
        build_case(steps=args.steps, wall_minutes=args.wall_minutes, capacity=args.capacity)
    ).run()
