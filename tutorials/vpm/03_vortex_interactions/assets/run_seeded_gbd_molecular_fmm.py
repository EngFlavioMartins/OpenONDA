#!/usr/bin/env python3
"""Qualify a fresh seeded ring pair with molecular-only GBD and CPU FMM.

Supply six CPU threads externally. No LES checkpoint is used.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import openonda.vpm as vpm
from openonda.tutorial_runner import load_case_module

gbd = load_case_module(Path(__file__).resolve().parents[1], "assets.run_seeded_gbd_fmm")
CASE_NAME = "gbd_molecular_fmm_cpu_root_200000_qualification"


def build_case(*, steps: int = 200, wall_minutes: float = 30):
    """Remove only the SGS closure from the matched fresh GBD case."""
    case = gbd.build_case(steps=steps, wall_minutes=wall_minutes, name=CASE_NAME)
    return replace(case, numerics=replace(case.numerics, turbulence=vpm.TurbulenceConfig.dns()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--wall-minutes", type=float, default=30)
    args = parser.parse_args()
    vpm.VPMSolver(build_case(steps=args.steps, wall_minutes=args.wall_minutes)).run()
