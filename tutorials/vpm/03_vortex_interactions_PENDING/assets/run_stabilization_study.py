#!/usr/bin/env python3
"""One existing stabilization strategy added to the frozen CS baseline.

Nominal values come from the existing configuration factories or leapfrogging
launchers. No physical or baseline-resolution parameter is exposed for tuning.
See leapfrogging_study.md for the acceptance gate and experiment ledger.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openonda.vpm as vpm
from openonda.tutorial_runner import load_case_module
from source.solvers.vpm.io.manifest import _case_configuration

ROOT = Path(__file__).resolve().parents[1]
base = load_case_module(ROOT, "assets.run_cs_redistribution")
METHODS = (
    "baseline", "stretching_viscosity", "pedrizzetti", "p_unnormalized",
    "p_moments", "splitting", "divergence_relaxation", "regularization",
    "solenoidal_remesh",
)


def build_case(method, *, name, steps=2400, wall_minutes=24, strength=None,
               reference="fig5"):
    """Keep the complete baseline, replacing only one mechanism's controls."""
    factory = {"fig5": base.build_frozen_case,
               "fig3": base.build_seeded_reference_case}[reference]
    case = factory(name=name, steps=steps, wall_minutes=wall_minutes)
    s = case.numerics.stabilization
    if method == "baseline":
        if strength is not None:
            raise ValueError("The frozen baseline has no tunable strength")
    elif method == "stretching_viscosity":
        s = replace(s, stretching_viscosity_coefficient=.5 if strength is None else strength)
    elif method in ("pedrizzetti", "p_unnormalized", "p_moments"):
        nominal = (.384684814725 * case.numerics.time_step_size
                   if method == "p_moments" else .3)
        s = replace(
            s, pedrizzetti_relaxation_factor=nominal if strength is None else strength,
            pedrizzetti_relaxation_preserve_vortex_strength=method == "pedrizzetti",
            pedrizzetti_relaxation_preserve_moments=method == "p_moments",
        )
    elif method == "splitting":
        factor = 2.0 if strength is None else strength
        peak = max(np.linalg.norm(ring.build().vortex_strength, axis=1).max()
                   for ring in case.initial_conditions)
        s = replace(s, filament_refinement=vpm.FilamentRefinementConfig.adaptive(
            interval_steps=5, max_vortex_strength_factor=factor,
            max_absolute_vortex_strength=factor * float(peak),
            offset_fraction=.25, max_n_particles=case.numerics.max_n_particles,
        ))
    elif method == "divergence_relaxation":
        s = replace(s, divergence_relaxation=vpm.DivergenceRelaxationConfig.constrained(
            interval_steps=25, start_step=25, grid_spacing=.05,
            regularization=.1 if strength is None else strength,
        ))
    elif method in ("regularization", "solenoidal_remesh"):
        # The built-in full remap includes its original guarded corrections.
        # Cadence, spatial resolution, core trigger, tail and capacity remain
        # frozen. Its factory's nominal dissipation allowance is a method
        # control, replacing the pure transfer's symmetric 1% error guard.
        allowance = .15 if strength is None else strength
        s = replace(
            s, regularization_transfer_only=False,
            regularization_solenoidal_remesh=method == "solenoidal_remesh",
            regularization_total_kinetic_energy_dissipation_limit=allowance,
            regularization_total_enstrophy_dissipation_limit=allowance,
        )
    else:
        raise ValueError(f"Unknown study method: {method}")
    return replace(case, numerics=replace(case.numerics, stabilization=s))


def configuration_record(case, reference="fig5"):
    """Record the exact one-method delta using the native manifest serializer."""
    actual = _case_configuration(SimpleNamespace(case=case))
    factory = {"fig5": base.build_frozen_case,
               "fig3": base.build_seeded_reference_case}[reference]
    control = _case_configuration(SimpleNamespace(case=factory(
        name=case.name, steps=case.run.steps,
        wall_minutes=case.run.wall_time_limit_seconds / 60,
    )))
    assert actual["initial_conditions"] == control["initial_conditions"]
    assert actual["run"] == control["run"]
    for key, value in control["numerics"].items():
        if key != "stabilization":
            assert actual["numerics"][key] == value, key
    before, after = [v["numerics"]["stabilization"] for v in (control, actual)]
    return {
        "reference": reference,
        "configuration": actual,
        "stabilization_delta": {k: {"baseline": before[k], "run": after[k]}
                                for k in before if before[k] != after[k]},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--reference", choices=("fig5", "fig3"), default="fig5",
                        help="Fig. 5 unperturbed control or Fig. 3 seeded qualification")
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--steps", type=int, default=2400,
                        help="Additional steps on a strict native restart")
    parser.add_argument("--wall-minutes", type=float, default=24)
    parser.add_argument("--strength", type=float,
                        help="Only the chosen stabilizer's principal strength")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--preflight", type=Path,
                        help="Write a native configuration and method delta; do not run")
    args = parser.parse_args()
    case = build_case(args.method, name=args.case_name, steps=args.steps,
                      wall_minutes=args.wall_minutes, strength=args.strength,
                      reference=args.reference)
    if args.preflight is not None:
        record = configuration_record(case, args.reference)
        args.preflight.write_text(json.dumps(record, indent=2))
        print(json.dumps(record["stabilization_delta"], indent=2))
    else:
        if args.restart is not None:
            case = replace(case, run=replace(case.run, initial_samples=False))
        solver = vpm.VPMSolver(case)
        try:
            if args.restart is not None:
                solver.load_backup(args.restart)
            solver.run()
        finally:
            solver.close()
