#!/usr/bin/env python3
"""Bounded CS core-redistribution accuracy controls; no damping or projection.

The fixed LES filter is the initial strength-weighted RMS of volume^(1/3).
Control and redistribution therefore use the same closure scale even though
the remap changes particle quadrature volumes. ``build_frozen_case`` records
the selected physical baseline; the general builder retains the earlier
controlled-experiment defaults.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import h5py
import openonda.vpm as vpm
from openonda.tutorial_runner import load_case_module

ROOT = Path(__file__).resolve().parents[1]
setup_les = load_case_module(ROOT, "assets.legacy_les")
FILTER_WIDTH = 0.05299057526524513


def build_case(*, remap, name, steps=800, wall_minutes=24, spacing=.06,
               reset_radius=.06, trigger_radius=.12, tail_budget=.0003,
               initial_resolution=None, scenario="kinematic"):
    case = setup_les.build_case(
        "halfdt", steps=steps, wall_minutes=wall_minutes,
        case_name=name, particle_capacity=120000,
        particle_spacing=initial_resolution,
        particle_core_radius=initial_resolution,
        scenario=scenario,
    )
    stabilization = vpm.StabilizationConfig.disabled()
    if remap:
        stabilization = vpm.StabilizationConfig(
            regularization_interval_steps=20,
            regularization_grid_spacing=spacing,
            regularization_core_radius=reset_radius,
            regularization_core_radius_trigger=trigger_radius,
            regularization_tail_budget=tail_budget,
            regularization_max_particles=120000,
            regularization_transfer_only=True,
            regularization_solenoidal_remesh=False,
            regularization_divergence_trigger=None,
            regularization_misalignment_trigger=None,
            # Transfer guards, not LBM agreement tolerances. Failed transfers
            # stop the run; the operator must not add damping to force a pass.
            regularization_total_kinetic_energy_dissipation_limit=.01,
            regularization_total_enstrophy_dissipation_limit=.01,
        )
    return replace(case, numerics=replace(
        case.numerics,
        turbulence=replace(case.numerics.turbulence, filter_width=FILTER_WIDTH),
        stabilization=stabilization,
    ))


def build_frozen_case(*, name, steps=2400, wall_minutes=24):
    """Reproduce the historical unperturbed control selected on 2026-09-13.

    The existing conservative transfer is particle representation maintenance.
    Damping, adaptive broadening, projection, relaxation and splitting are off.
    The user selected the original, unperturbed Fig. 5 comparison. This control
    does not qualify the paper's separate seeded Fig. 3 breakdown problem.
    Only output name and observation budget vary in this factory.
    """
    return build_case(
        remap=True, name=name, steps=steps, wall_minutes=wall_minutes,
        initial_resolution=.05, spacing=.05, reset_radius=.05,
        trigger_radius=.1, tail_budget=.003,
    )


def build_seeded_reference_case(*, name, steps=1200, wall_minutes=24):
    """Qualify the same representation against Cheng et al. Fig. 3.

    The existing seeded scenario supplies Re=3415, axial mode eight and
    amplitude .05 R0. Both rings use the existing equal phase of zero;
    the paper does not specify a relative phase. No stabilizer is enabled.
    This is a qualification candidate, not an already accepted baseline.
    """
    return build_case(
        remap=True, name=name, steps=steps, wall_minutes=wall_minutes,
        initial_resolution=.05, spacing=.05, reset_radius=.05,
        trigger_radius=.1, tail_budget=.003, scenario="seeded_breakdown",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "redistribution"), required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--steps", type=int, default=800, help="Additional steps on restart")
    parser.add_argument("--wall-minutes", type=float, default=24)
    parser.add_argument("--initial-resolution", type=float,
                        help="Initial particle spacing and numerical core radius; remap options are separate")
    parser.add_argument("--spacing", type=float, default=.06, help="Remapping grid only")
    parser.add_argument("--reset-radius", type=float, default=.06)
    parser.add_argument("--trigger-radius", type=float, default=.12)
    parser.add_argument("--tail-budget", type=float, default=.0003)
    parser.add_argument("--restart", type=Path)
    args = parser.parse_args()
    case = build_case(
        remap=args.mode == "redistribution", name=args.case_name, steps=args.steps,
        wall_minutes=args.wall_minutes, spacing=args.spacing,
        reset_radius=args.reset_radius, trigger_radius=args.trigger_radius,
        tail_budget=args.tail_budget,
        initial_resolution=args.initial_resolution,
    )
    if args.restart is not None:
        case = replace(case, run=replace(case.run, initial_samples=False))
    solver = vpm.VPMSolver(case)
    try:
        if args.restart is not None:
            with h5py.File(args.restart, "r") as checkpoint:
                print("Strict restart from accepted step", checkpoint["solver"].attrs["step"])
            solver.load_backup(args.restart)
        solver.run()
    finally:
        solver.close()
