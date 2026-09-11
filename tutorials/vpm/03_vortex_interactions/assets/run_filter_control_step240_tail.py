#!/usr/bin/env python3
"""Complete the approved Cs=.20 target from step 240 to step 300 once."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import h5py

import openonda.vpm as vpm
from openonda.tutorial_runner import case_package
from source.solvers.vpm.config.fingerprint import numerical_configuration


TUTORIAL = Path(__file__).resolve().parents[1]
__package__ = case_package(TUTORIAL) + ".assets"
from .. import setup_les
from .run_filter_pair_continuation import _installed_vpm_manifest, _sha256


SOURCE = (
    TUTORIAL
    / "solution"
    / "cs_breakdown_filter_cs020_cpu_t6_step300_continuation"
    / "vpm_000240.h5"
)
SOURCE_SHA256 = "f9f08a6a8f37b7b250c987c4886010d5e268829e32cf4ba8076c9e1f8bf646dc"
CONFIGURATION_SHA256 = "4b4d03f21a1bb308b25f158385f682dc086a2010fcd5f1e8e172eb24d97753e2"
CASE_NAME = "cs_breakdown_filter_cs020_cpu_t6_step300_tail"
OUTPUT = TUTORIAL / "figures" / "cs_filter_pair_step300_control_tail_preflight"


def main() -> None:
    if os.environ.get("TI_CPU_MAX_NUM_THREADS") != "6":
        raise RuntimeError("set TI_CPU_MAX_NUM_THREADS=6 before starting this process")
    runtime = _installed_vpm_manifest()
    if _sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("control step-240 checkpoint hash changed")
    with h5py.File(SOURCE, "r") as handle:
        solver_state = handle["solver"]
        observed = (
            int(solver_state.attrs["step"]),
            float(solver_state.attrs["time"]),
            int(solver_state.attrs["n_particles_total"]),
            int(solver_state.attrs["n_stabilization_events"]),
            int(solver_state.attrs["n_regularization_events"]),
        )
        stored_configuration = str(solver_state.attrs["numerical_configuration"])
        stored_hash = str(solver_state.attrs["numerical_configuration_sha256"])
    if observed != (240, 1.8, 16_104, 0, 0):
        raise RuntimeError(f"unexpected control tail source state: {observed}")
    if stored_hash != CONFIGURATION_SHA256:
        raise RuntimeError("control tail source configuration hash changed")

    case = setup_les.build_case(
        "baseline",
        scenario="seeded_breakdown",
        compute_device="CPU",
        steps=60,
        wall_minutes=5.0,
        particle_spacing=0.06,
        particle_core_radius=0.06,
        smagorinsky_coefficient=0.20,
        case_name=CASE_NAME,
    )
    case = replace(case, run=replace(case.run, initial_samples=False))
    candidate = json.dumps(
        numerical_configuration(case.numerics),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    if candidate != stored_configuration or candidate_hash != CONFIGURATION_SHA256:
        raise RuntimeError("control tail is not a strict numerical restart")
    if case.run.steps != 60 or case.run.wall_time_limit_seconds != 300.0:
        raise RuntimeError("control tail run plan changed")
    for path in (
        TUTORIAL / "solution" / CASE_NAME,
        TUTORIAL / "samples" / CASE_NAME,
    ):
        if path.exists():
            raise FileExistsError(f"control tail namespace is not fresh: {path}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "PREFLIGHT_PASSED_LAUNCHING",
        "command": (
            "env TI_CPU_MAX_NUM_THREADS=6 "
            "/opt/anaconda3/envs/OpenONDA/bin/python "
            "assets/run_filter_control_step240_tail.py"
        ),
        "source_checkpoint": str(SOURCE),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_state": {
            "step": 240,
            "time": 1.8,
            "n_particles_total": 16_104,
            "n_stabilization_events": 0,
            "n_regularization_events": 0,
        },
        "numerical_configuration_sha256": CONFIGURATION_SHA256,
        "runtime_package_root": runtime["package_root"],
        "runtime_python_file_count": runtime["python_file_count"],
        "runtime_aggregate_sha256": runtime["aggregate_sha256"],
        "continuation": {
            "case_name": CASE_NAME,
            "steps": 60,
            "target_step": 300,
            "target_time": 2.25,
            "native_cap_seconds": 300.0,
            "threads": 6,
        },
    }
    (OUTPUT / "preflight.json").write_text(json.dumps(provenance, indent=2) + "\n")

    solver = vpm.VPMSolver(case)
    try:
        solver.load_backup(SOURCE)
        if (solver.step, solver.time, solver.particles.n_particles_total) != (
            240,
            1.8,
            16_104,
        ):
            raise RuntimeError("loaded tail source differs from the captured checkpoint")
        solver.run()
    finally:
        solver.close()


if __name__ == "__main__":
    main()
