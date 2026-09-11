#!/usr/bin/env python3
"""Dry-check the held step-80 to step-300 seeded filter continuations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openonda.tutorial_runner import case_package

if not __package__:
    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

from .run_filter_pair_continuation import (  # noqa: E402
    CONTINUATION_STEPS,
    FINAL_STEP,
    FINAL_TIME,
    LEGS,
    THREADS,
    _installed_vpm_manifest,
    validate_leg,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "cs_filter_pair_step300_preflight"
PYTHON = "/opt/anaconda3/envs/OpenONDA/bin/python"


def _command(leg: str) -> str:
    return (
        f"env TI_CPU_MAX_NUM_THREADS={THREADS} {PYTHON} "
        f"assets/run_filter_pair_continuation.py --leg {leg}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    runtime = _installed_vpm_manifest()
    legs = {leg: validate_leg(leg) for leg in LEGS}
    expected_molecular_particle_core = float(np.sqrt(0.06**2 + 4.0 * (np.pi / 3415.0) * FINAL_TIME))
    expected_molecular_physical_core = float(np.sqrt(0.1**2 + 4.0 * (np.pi / 3415.0) * FINAL_TIME))
    result = {
        "status": "PREFLIGHT_PASSED_RELEASED_FOR_LAUNCH",
        "run_order": list(LEGS),
        "commands": {leg: _command(leg) for leg in LEGS},
        "threads": THREADS,
        "continuation_steps_per_leg": CONTINUATION_STEPS,
        "final_step": FINAL_STEP,
        "final_time": FINAL_TIME,
        "native_caps_minutes": {leg: info["wall_minutes"] for leg, info in LEGS.items()},
        "actual_aggregate_process_budget_minutes": 20.0,
        "runtime_source": runtime,
        "legs": legs,
        "expected_diagnostic": {
            "nondimensional_time_tGamma_over_R2": float(np.pi * FINAL_TIME),
            "molecular_particle_core": expected_molecular_particle_core,
            "molecular_physical_core": expected_molecular_physical_core,
            "interpretation": (
                "compare both complex mode components and their joint norm, both "
                "native-plane core widths, particle geometry, impulse and common-estimator "
                "integrals; coherent mode retention is not itself proof of breakdown"
            ),
        },
    }
    (output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")

    rows = [
        "# Held step-300 filter-pair continuation",
        "",
        "Status: **preflight passed; Delta process released; launch authorized**.",
        "",
        f"Installed VPM aggregate: `{runtime['aggregate_sha256']}` "
        f"({runtime['python_file_count']} Python files under `{runtime['package_root']}`).",
        "",
        "| leg | source step/time/N | checkpoint SHA-256 | config SHA-256 | additional steps | native cap | target |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for leg, record in legs.items():
        state = record["source_state"]
        rows.append(
            f"| {leg} | {state['step']}/{state['time']}/{state['n_particles_total']} | "
            f"`{record['checkpoint_sha256']}` | "
            f"`{state['numerical_configuration_sha256']}` | "
            f"{record['continuation_steps']} | {record['wall_minutes']:g} min | "
            f"step {record['final_step']}, t={record['final_time']:g} |"
        )
    rows += [
        "",
        "Exact sequential commands:",
        "",
        f"1. `{_command('control_cs020')}`",
        f"2. `{_command('molecular_cs000')}`",
        "",
        "The second command is conditional on first-leg completion/release review. "
        "Both commands use strict checkpoint identity validation with no timestep or "
        "configuration bypass and refuse occupied output namespaces.",
        "",
        "At `t=2.25 s` (`t Gamma0/R0^2 = 7.06858`), molecular spreading predicts "
        f"particle sigma `{expected_molecular_particle_core:.8f} R0` and physical-core "
        f"width `{expected_molecular_physical_core:.8f} R0`. The discriminating result "
        "is whether the two-plane core separation develops into phase-consistent joint "
        "mode retention while guards remain valid; this endpoint need not prove breakdown.",
    ]
    (output / "preflight.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
