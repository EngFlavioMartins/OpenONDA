#!/usr/bin/env python3
"""Strictly continue one approved seeded filter leg from step 80 to step 300.

This runner refuses a changed checkpoint, installed VPM source generation,
numerical configuration, thread count, or occupied output namespace.  It uses
the native restart validator without any identity or timestep override.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py

import openonda.vpm as vpm
from openonda.tutorial_runner import case_package
import source.solvers.vpm as vpm_package
from source.solvers.vpm.config.fingerprint import numerical_configuration


TUTORIAL = Path(__file__).resolve().parents[1]
__package__ = case_package(TUTORIAL) + ".assets"
from . import legacy_les as setup_les


THREADS = 6
SOURCE_STEP = 80
SOURCE_TIME = 0.6
SOURCE_PARTICLES = 16_104
CONTINUATION_STEPS = 220
FINAL_STEP = 300
FINAL_TIME = 2.25
EXPECTED_INSTALLED_VPM_SHA256 = "338f353bbd681a67b2862c8d0e810ec92fcec4261a05a59607f3d223550e0303"
LEGS: dict[str, dict[str, Any]] = {
    "control_cs020": {
        "smagorinsky": 0.20,
        "wall_minutes": 10.0,
        "source_run": "cs_breakdown_filter_cs020_cpu_t6_step080",
        "source_h5_sha256": ("26cd3372c0f0364cf7bed79d2558c85024c95d5368b964d868cea6b0f07c2372"),
        "numerical_configuration_sha256": (
            "4b4d03f21a1bb308b25f158385f682dc086a2010fcd5f1e8e172eb24d97753e2"
        ),
        "case_name": "cs_breakdown_filter_cs020_cpu_t6_step300_continuation",
    },
    "molecular_cs000": {
        "smagorinsky": 0.0,
        "wall_minutes": 5.0,
        "source_run": "cs_breakdown_filter_cs000_cpu_t6_step080",
        "source_h5_sha256": ("f6a90f4ecd576ed35926de21803fc7143f5dc91ea6ca23eff5f5dce0820d1243"),
        "numerical_configuration_sha256": (
            "0e482b559ea93ca030209ac766fe9507a979996331a1af46d664d7e157bcdcbd"
        ),
        "case_name": "cs_breakdown_filter_cs000_cpu_t6_step300_continuation",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_configuration(configuration: dict[str, Any]) -> str:
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _installed_vpm_manifest() -> dict[str, Any]:
    package_root = Path(vpm_package.__file__).resolve().parent
    if "site-packages" not in package_root.parts:
        raise RuntimeError(
            f"continuation must use the installed VPM generation, resolved {package_root}"
        )
    files = sorted(package_root.rglob("*.py"))
    aggregate = hashlib.sha256()
    entries = []
    for path in files:
        relative = path.relative_to(package_root).as_posix()
        file_hash = _sha256(path)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(file_hash))
        entries.append({"path": relative, "sha256": file_hash})
    result = {
        "package_root": str(package_root),
        "python_file_count": len(entries),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": entries,
    }
    if result["aggregate_sha256"] != EXPECTED_INSTALLED_VPM_SHA256:
        raise RuntimeError(
            "installed VPM source generation changed: "
            f"{result['aggregate_sha256']} != {EXPECTED_INSTALLED_VPM_SHA256}"
        )
    return result


def _source_checkpoint(info: dict[str, Any]) -> Path:
    return TUTORIAL / "solution" / info["source_run"] / f"vpm_{SOURCE_STEP:06d}.h5"


def build_continuation(leg: str):
    info = LEGS[leg]
    case = setup_les.build_case(
        "baseline",
        scenario="seeded_breakdown",
        compute_device="CPU",
        steps=CONTINUATION_STEPS,
        wall_minutes=info["wall_minutes"],
        particle_spacing=0.06,
        particle_core_radius=0.06,
        smagorinsky_coefficient=info["smagorinsky"],
        case_name=info["case_name"],
    )
    return replace(case, run=replace(case.run, initial_samples=False))


def validate_leg(leg: str, *, require_fresh_namespace: bool = True) -> dict[str, Any]:
    info = LEGS[leg]
    checkpoint = _source_checkpoint(info)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"native source checkpoint is missing: {checkpoint}")
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != info["source_h5_sha256"]:
        raise RuntimeError(f"source checkpoint changed for {leg}: {checkpoint_hash}")

    with h5py.File(checkpoint, "r") as handle:
        solver = handle["solver"]
        state = {
            "step": int(solver.attrs["step"]),
            "time": float(solver.attrs["time"]),
            "n_particles_total": int(solver.attrs["n_particles_total"]),
            "n_stabilization_events": int(solver.attrs["n_stabilization_events"]),
            "n_regularization_events": int(solver.attrs["n_regularization_events"]),
            "numerical_configuration": str(solver.attrs["numerical_configuration"]),
            "numerical_configuration_sha256": str(solver.attrs["numerical_configuration_sha256"]),
        }
    expected_state = (SOURCE_STEP, SOURCE_TIME, SOURCE_PARTICLES, 0, 0)
    observed_state = (
        state["step"],
        state["time"],
        state["n_particles_total"],
        state["n_stabilization_events"],
        state["n_regularization_events"],
    )
    if observed_state != expected_state:
        raise RuntimeError(f"unexpected source checkpoint state for {leg}: {observed_state}")
    if state["numerical_configuration_sha256"] != info["numerical_configuration_sha256"]:
        raise RuntimeError(f"stored numerical-configuration hash changed for {leg}")

    case = build_continuation(leg)
    candidate_configuration = _canonical_configuration(numerical_configuration(case.numerics))
    candidate_hash = hashlib.sha256(candidate_configuration.encode("utf-8")).hexdigest()
    if candidate_configuration != state["numerical_configuration"]:
        raise RuntimeError(f"strict restart configuration differs for {leg}")
    if candidate_hash != info["numerical_configuration_sha256"]:
        raise RuntimeError(f"candidate numerical-configuration hash changed for {leg}")
    if case.run.steps != CONTINUATION_STEPS or case.run.initial_samples:
        raise RuntimeError(f"invalid continuation run plan for {leg}")
    if case.run.wall_time_limit_seconds != 60.0 * info["wall_minutes"]:
        raise RuntimeError(f"invalid continuation wall cap for {leg}")
    if SOURCE_STEP + case.run.steps != FINAL_STEP:
        raise RuntimeError(f"invalid final step for {leg}")
    if SOURCE_TIME + case.run.steps * case.numerics.time_step_size != FINAL_TIME:
        raise RuntimeError(f"invalid final time for {leg}")

    output_paths = (
        TUTORIAL / "solution" / info["case_name"],
        TUTORIAL / "samples" / info["case_name"],
    )
    if require_fresh_namespace:
        occupied = [str(path) for path in output_paths if path.exists()]
        if occupied:
            raise FileExistsError(f"continuation namespace is not fresh: {occupied}")
    return {
        "leg": leg,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "source_state": {
            key: value for key, value in state.items() if key != "numerical_configuration"
        },
        "case_name": info["case_name"],
        "smagorinsky_coefficient": info["smagorinsky"],
        "continuation_steps": case.run.steps,
        "final_step": FINAL_STEP,
        "final_time": FINAL_TIME,
        "wall_minutes": info["wall_minutes"],
        "output_paths": [str(path) for path in output_paths],
    }


def run(leg: str) -> None:
    if os.environ.get("TI_CPU_MAX_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"set TI_CPU_MAX_NUM_THREADS={THREADS} before starting this process")
    _installed_vpm_manifest()
    record = validate_leg(leg)
    solver = vpm.VPMSolver(build_continuation(leg))
    try:
        # No time_step_size override: the framework performs strict identity validation.
        solver.load_backup(record["checkpoint"])
        if (solver.step, solver.time, solver.particles.n_particles_total) != (
            SOURCE_STEP,
            SOURCE_TIME,
            SOURCE_PARTICLES,
        ):
            raise RuntimeError("loaded checkpoint state does not match the preflight")
        solver.run()
    finally:
        solver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", choices=tuple(LEGS), required=True)
    args = parser.parse_args()
    run(args.leg)


if __name__ == "__main__":
    main()
