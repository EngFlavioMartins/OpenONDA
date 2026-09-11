#!/usr/bin/env python3
"""Qualify the fresh seeded ring pair with GBD, LES and CPU FMM.

Supply six CPU threads externally. A restart uses the ordinary strict VPM
backup contract and writes to a distinct continuation namespace.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import h5py

import openonda.vpm as vpm
from openonda.tutorial_runner import load_case_module

setup_les = load_case_module(Path(__file__).resolve().parents[1], "setup_les")
CASE_NAME = "gbd_breakdown_fmm_cpu_root_200000_qualification"
PARTICLE_CAPACITY = 200_000


def build_case(*, steps: int = 200, wall_minutes: float = 30, name: str = CASE_NAME):
    """Keep the seeded physical case and qualify one bounded GBD alternative."""
    case = setup_les.build_case(
        "baseline",
        scenario="seeded_breakdown",
        compute_device="CPU",
        steps=steps,
        wall_minutes=wall_minutes,
        case_name=name,
    )
    numerics = case.numerics
    viscous = vpm.ViscousConfig.gbd(
        particle_spacing=0.06,
        padding=3,
        threshold=1e-4,
        threshold_mode="budget",
        kinematic_viscosity=numerics.viscous.kinematic_viscosity,
        max_nodes=PARTICLE_CAPACITY,
        core_radius_ratio=1,
        remeshing_kernel="M4_PRIME",
    )
    return replace(
        case,
        numerics=replace(
            numerics,
            induction=vpm.FMMInduction(stretching_scheme="TRANSPOSED"),
            viscous=viscous,
            max_n_particles=PARTICLE_CAPACITY,
            diagnostics=replace(numerics.diagnostics, detailed_timing=True),
        ),
        run=replace(
            case.run,
            resource_limits=replace(case.run.resource_limits, max_particles=PARTICLE_CAPACITY),
        ),
    )


def build_continuation(start_step: int, *, until_step: int = 1200, wall_minutes: float = 180):
    """Count additional steps from the saved clock without changing numerics."""
    return build_case(
        steps=until_step - start_step,
        wall_minutes=wall_minutes,
        name=f"gbd_breakdown_fmm_cpu_root_200000_from{start_step:06d}_to{until_step:06d}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--wall-minutes", type=float, help="Default: 30 fresh, 180 on restart")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--until-step", type=int, default=1200)
    args = parser.parse_args()
    if args.restart is None:
        case = build_case(
            steps=args.steps, wall_minutes=30 if args.wall_minutes is None else args.wall_minutes
        )
    else:
        with h5py.File(args.restart, "r") as checkpoint:
            start_step = int(checkpoint["solver"].attrs["step"])
        case = build_continuation(
            start_step,
            until_step=args.until_step,
            wall_minutes=180 if args.wall_minutes is None else args.wall_minutes,
        )
    solver = vpm.VPMSolver(case)
    if args.restart is not None:
        solver.load_backup(args.restart)
    solver.run()
