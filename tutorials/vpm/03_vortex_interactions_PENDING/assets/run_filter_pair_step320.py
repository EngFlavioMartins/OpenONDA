#!/usr/bin/env python3
"""Preflight or run one held step-300 to step-320 filter discriminator leg."""

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
REPOSITORY = TUTORIAL.parents[2]
__package__ = case_package(TUTORIAL) + ".assets"
from . import legacy_les as setup_les


THREADS = 6
SOURCE_STEP = 300
SOURCE_TIME = 2.25
SOURCE_PARTICLES = 16_104
CONTINUATION_STEPS = 20
FINAL_STEP = 320
FINAL_TIME = 2.4
ACTUAL_PROCESS_BUDGET_SECONDS = 300.0
PYTHON = "/opt/anaconda3/envs/OpenONDA/bin/python"
OUTPUT = TUTORIAL / "figures" / "cs_filter_pair_step320_preflight"
PRELABEL_MANIFEST = (
    REPOSITORY
    / "docs"
    / "reviews"
    / "runtime_generations"
    / "2026-09-10-post-gaussian-pre-label-vpm-source.json"
)
PRELABEL_MANIFEST_SHA256 = "8426991d8f7110188843f27a43339d31edbdec2ed478d406fee8d4d7443623f2"
EXPECTED_INSTALLED_VPM_SHA256 = "ab4be73c378d83ce6ba169f0253b3c13a6288c22171872fa198dd42c810d956e"
EXPECTED_OPENONDA_VPM_SHA256 = "974b82e72aa9da9593352f545cf097606356e9c74c6f8150b1c0d1dabc218045"
EXPECTED_METADATA_ONLY_CHANGES = {
    "io/solver_io.py": {
        "prelabel": ("15835e8cf3973e14f373c59025d2c3a90ea277acfb85fcfda8da8f7aa1b1bd30"),
        "installed": ("7264896363b344d0971302ca00d2f4ccea1d917d5a849ea0d44aeab5240b450a"),
    },
    "physics/evaluation.py": {
        "prelabel": ("9b6754ebb4283d1ebcae6d3a564bed1c2823b28aeef1b5c753bbf964d758703d"),
        "installed": ("f4ea28644095170c6df28f8ce0482bf6498750b0b76ed3b2a1d688044dcf8339"),
    },
}
EXPECTED_FOURIER_SHA256 = "0368aa63e4dc3fdf3906a053c9fc0b53f0cc8f1ddb008f802ea8a3d70b0c4262"
LEGS: dict[str, dict[str, Any]] = {
    "control_cs020": {
        "smagorinsky": 0.20,
        "wall_minutes": 2.0,
        "source_run": "cs_breakdown_filter_cs020_cpu_t6_step300_tail",
        "source_h5_sha256": ("51ac824ce35a00a3b48b9ff3c8632e8ab6d20026c7ad009bdb061a637f8dfd10"),
        "numerical_configuration_sha256": (
            "4b4d03f21a1bb308b25f158385f682dc086a2010fcd5f1e8e172eb24d97753e2"
        ),
        "case_name": "cs_breakdown_filter_cs020_cpu_t6_step320_discriminator",
    },
    "molecular_cs000": {
        "smagorinsky": 0.0,
        "wall_minutes": 1.0,
        "source_run": "cs_breakdown_filter_cs000_cpu_t6_step300_continuation",
        "source_h5_sha256": ("ac8e1690c8b85d18af2eb7da59f1a5431fdbb72415bbc4679b5c893b9c7c2201"),
        "numerical_configuration_sha256": (
            "0e482b559ea93ca030209ac766fe9507a979996331a1af46d664d7e157bcdcbd"
        ),
        "case_name": "cs_breakdown_filter_cs000_cpu_t6_step320_discriminator",
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


def _runtime_identity() -> dict[str, Any]:
    package_root = Path(vpm_package.__file__).resolve().parent
    if "site-packages" not in package_root.parts:
        raise RuntimeError(f"expected installed VPM package, resolved {package_root}")
    installed = {}
    aggregate = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        file_hash = _sha256(path)
        installed[relative] = file_hash
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(file_hash))
    if aggregate.hexdigest() != EXPECTED_INSTALLED_VPM_SHA256:
        raise RuntimeError("installed reviewed VPM aggregate changed")

    if _sha256(PRELABEL_MANIFEST) != PRELABEL_MANIFEST_SHA256:
        raise RuntimeError("preserved pre-label manifest changed")
    manifest = json.loads(PRELABEL_MANIFEST.read_text())
    prelabel = {
        entry["path"].removeprefix("source/solvers/vpm/"): entry["sha256"]
        for entry in manifest["files"]
        if entry["path"].startswith("source/solvers/vpm/") and entry["path"].endswith(".py")
    }
    changed = {
        relative
        for relative in set(prelabel) | set(installed)
        if prelabel.get(relative) != installed.get(relative)
    }
    if changed != set(EXPECTED_METADATA_ONLY_CHANGES):
        raise RuntimeError(f"unexpected installed/pre-label source delta: {changed}")
    for relative, expected in EXPECTED_METADATA_ONLY_CHANGES.items():
        if (
            prelabel.get(relative) != expected["prelabel"]
            or installed.get(relative) != expected["installed"]
        ):
            raise RuntimeError(f"reviewed metadata-only file identity changed: {relative}")
    if installed.get("numerics/fourier_integrals.py") != EXPECTED_FOURIER_SHA256:
        raise RuntimeError("installed Fourier evaluator changed")
    openonda_vpm_path = Path(vpm.__file__).resolve()
    if _sha256(openonda_vpm_path) != EXPECTED_OPENONDA_VPM_SHA256:
        raise RuntimeError("installed openonda.vpm API changed")
    return {
        "package_root": str(package_root),
        "python_file_count": len(installed),
        "aggregate_sha256": aggregate.hexdigest(),
        "openonda_vpm_path": str(openonda_vpm_path),
        "openonda_vpm_sha256": EXPECTED_OPENONDA_VPM_SHA256,
        "prelabel_manifest": str(PRELABEL_MANIFEST),
        "prelabel_manifest_sha256": PRELABEL_MANIFEST_SHA256,
        "reviewed_metadata_only_changes": EXPECTED_METADATA_ONLY_CHANGES,
        "fourier_integrals_sha256": EXPECTED_FOURIER_SHA256,
    }


def _source_checkpoint(info: dict[str, Any]) -> Path:
    return TUTORIAL / "solution" / info["source_run"] / "vpm_000300.h5"


def _build_case(leg: str):
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


def _validate_leg(leg: str, *, require_fresh_namespace: bool = True) -> dict[str, Any]:
    info = LEGS[leg]
    checkpoint = _source_checkpoint(info)
    if not checkpoint.is_file() or _sha256(checkpoint) != info["source_h5_sha256"]:
        raise RuntimeError(f"reviewed source checkpoint changed for {leg}")
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
    observed = (
        state["step"],
        state["time"],
        state["n_particles_total"],
        state["n_stabilization_events"],
        state["n_regularization_events"],
    )
    if observed != (SOURCE_STEP, SOURCE_TIME, SOURCE_PARTICLES, 0, 0):
        raise RuntimeError(f"unexpected source state for {leg}: {observed}")
    if state["numerical_configuration_sha256"] != info["numerical_configuration_sha256"]:
        raise RuntimeError(f"stored numerical configuration changed for {leg}")

    case = _build_case(leg)
    candidate = _canonical_configuration(numerical_configuration(case.numerics))
    candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    if candidate != state["numerical_configuration"]:
        raise RuntimeError(f"strict restart configuration differs for {leg}")
    if candidate_hash != info["numerical_configuration_sha256"]:
        raise RuntimeError(f"candidate numerical configuration changed for {leg}")
    if (
        case.run.steps != CONTINUATION_STEPS
        or case.run.wall_time_limit_seconds != 60.0 * info["wall_minutes"]
        or case.run.initial_samples
    ):
        raise RuntimeError(f"invalid captured run plan for {leg}")
    if SOURCE_STEP + case.run.steps != FINAL_STEP:
        raise RuntimeError(f"invalid final step for {leg}")
    final_time = SOURCE_TIME + case.run.steps * case.numerics.time_step_size
    if abs(final_time - FINAL_TIME) > 1.0e-15:
        raise RuntimeError(f"invalid final time for {leg}: {final_time}")

    output_paths = (
        TUTORIAL / "solution" / info["case_name"],
        TUTORIAL / "samples" / info["case_name"],
    )
    if require_fresh_namespace:
        occupied = [str(path) for path in output_paths if path.exists()]
        if occupied:
            raise FileExistsError(f"step-320 namespace is not fresh: {occupied}")
    return {
        "leg": leg,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": info["source_h5_sha256"],
        "source_state": {
            key: value for key, value in state.items() if key != "numerical_configuration"
        },
        "case_name": info["case_name"],
        "smagorinsky_coefficient": info["smagorinsky"],
        "steps": case.run.steps,
        "target_step": FINAL_STEP,
        "target_time": FINAL_TIME,
        "native_cap_seconds": case.run.wall_time_limit_seconds,
        "output_paths": [str(path) for path in output_paths],
        "required_terminal_outputs": [
            str(output_paths[0] / "vpm_000320.h5"),
            str(output_paths[1] / "core_section_000320.vts"),
            str(output_paths[1] / "cross_section_000320.vts"),
        ],
    }


def _command(leg: str) -> str:
    return (
        f"env TI_CPU_MAX_NUM_THREADS={THREADS} {PYTHON} "
        f"assets/run_filter_pair_step320.py --leg {leg}"
    )


def _write_preflight() -> None:
    runtime = _runtime_identity()
    legs = {leg: _validate_leg(leg) for leg in LEGS}
    result = {
        "status": "PREFLIGHT_PASSED_HELD_FOR_COMPUTE_RELEASE",
        "threads": THREADS,
        "commands": {leg: _command(leg) for leg in LEGS},
        "run_order": list(LEGS),
        "actual_process_budget_seconds": ACTUAL_PROCESS_BUDGET_SECONDS,
        "native_cap_seconds": {leg: record["native_cap_seconds"] for leg, record in legs.items()},
        "runtime_identity": runtime,
        "legs": legs,
        "analysis_plan": {
            "group_0": (
                "compare cross-plane local mirror asymmetry, two-plane morphology "
                "and complex centered modes 4, 8 and 12 at exact clocks"
            ),
            "group_1": "compare complex centered mode 8 separately",
            "claim_boundary": (
                "one short-interval persistence result screens the next decision; "
                "it does not prove physical instability or breakdown"
            ),
            "planned_artifact": str(
                TUTORIAL / "figures" / "cs_filter_pair_step320" / "comparison.json"
            ),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")

    rows = [
        "# Held matched step-320 filter discriminator",
        "",
        "Status: **preflight passed; held for explicit compute release**.",
        "",
        f"Reviewed installed VPM aggregate: `{runtime['aggregate_sha256']}` "
        f"({runtime['python_file_count']} Python files). Relative to the preserved "
        "pre-label manifest, exactly `io/solver_io.py` and "
        "`physics/evaluation.py` changed for metadata labels. The Fourier evaluator "
        f"remains `{runtime['fourier_integrals_sha256']}`; dynamics are unchanged.",
        "",
        "| leg | source SHA-256 | config SHA-256 | namespace | cap | target |",
        "|---|---|---|---|---:|---|",
    ]
    for leg, record in legs.items():
        rows.append(
            f"| {leg} | `{record['source_checkpoint_sha256']}` | "
            f"`{record['source_state']['numerical_configuration_sha256']}` | "
            f"`{record['case_name']}` | {record['native_cap_seconds'] / 60:g} min | "
            f"{record['target_step']}/{record['target_time']} |"
        )
    rows += [
        "",
        "Exact sequential commands after explicit compute release:",
        "",
        f"1. `{_command('control_cs020')}`",
        f"2. `{_command('molecular_cs000')}`",
        "",
        "The driver strictly validates current installed identity, the preserved "
        "two-file metadata-only delta, source HDF5 hashes/states, unchanged numerical "
        "configurations, six threads, caps and fresh output namespaces. It calls the "
        "public restart API without a timestep or identity override.",
        "",
        "Required terminal outputs are `vpm_000320.h5`, "
        "`core_section_000320.vts` and `cross_section_000320.vts` for each leg. "
        "Offline comparison will track group-0 cross-plane local mirror asymmetry, "
        "complex modes 4/8/12 and morphology on both native planes, separately from "
        "group-1 mode 8. Persistence over this interval is a screening result, not "
        "proof of physical instability.",
    ]
    (OUTPUT / "preflight.md").write_text("\n".join(rows) + "\n")


def _run_leg(leg: str) -> None:
    if os.environ.get("TI_CPU_MAX_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"set TI_CPU_MAX_NUM_THREADS={THREADS} before launch")
    _runtime_identity()
    record = _validate_leg(leg)
    solver = vpm.VPMSolver(_build_case(leg))
    try:
        solver.load_backup(record["source_checkpoint"])
        if (solver.step, solver.time, solver.particles.n_particles_total) != (
            SOURCE_STEP,
            SOURCE_TIME,
            SOURCE_PARTICLES,
        ):
            raise RuntimeError("loaded checkpoint differs from captured source state")
        solver.run()
    finally:
        solver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--leg", choices=tuple(LEGS))
    args = parser.parse_args()
    if args.preflight:
        _write_preflight()
    else:
        _run_leg(args.leg)


if __name__ == "__main__":
    main()
