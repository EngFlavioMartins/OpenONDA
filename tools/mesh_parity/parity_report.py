"""CLI and report orchestration for the cfMesh differential oracle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .cfmesh_oracle import (
    CfMeshExecutionError,
    CfMeshUnavailableError,
    OpenONDAUnsupportedForParityError,
    ParitySpec,
    load_parity_spec,
    run_cfmesh,
    run_openonda,
)
from .compare_meshes import ComparisonOptions, compare_meshes

DEFAULT_WORKFLOW_STAGES = (
    "templateGeneration",
    "surfaceTopology",
    "surfaceProjection",
    "patchAssignment",
    "edgeExtraction",
    "boundaryLayerGeneration",
    "meshOptimisation",
    "boundaryLayerRefinement",
)

_SURFACE_UNTANGLING_STAGES = frozenset({"surfaceProjection", "patchAssignment"})
_SURFACE_OPTIMISATION_STAGES = frozenset({"edgeExtraction"})
_WRAPPER_STAGES = frozenset({"boundaryLayerGeneration"})
_FINAL_OPTIMISATION_STAGES = frozenset({"meshOptimisation", "boundaryLayerRefinement"})

# cfMesh's surface optimizer intentionally terminates at a relative objective
# tolerance of 1e-3. Symmetric curved-surface vertices can therefore select
# different, equally valid local minima. These profiles are measured envelopes
# for the coarse curved-cylinder oracle; Level-A/B/C topology remains exact.
CFMESH_SURFACE_UNTANGLING_OPTIONS = ComparisonOptions(
    centroid_relative_tolerance=1.0e-3,
    centroid_absolute_tolerance=1.0e-10,
    volume_relative_tolerance=1.0e-2,
    face_normal_angle_tolerance_degrees=1.01,
)
CFMESH_SURFACE_OPTIMISATION_OPTIONS = ComparisonOptions(
    centroid_relative_tolerance=5.0e-3,
    centroid_absolute_tolerance=1.0e-10,
    volume_relative_tolerance=5.0e-2,
    face_normal_angle_tolerance_degrees=0.65,
    candidate_limit=64,
    assignment_component_limit=2048,
)
CFMESH_WRAPPER_OPTIONS = ComparisonOptions(
    centroid_relative_tolerance=5.0e-4,
    centroid_absolute_tolerance=1.0e-10,
    # Wrapper cells are thinner than their Cartesian parents, so small surface
    # displacements produce a larger relative volume envelope.
    volume_relative_tolerance=7.0e-2,
    face_normal_angle_tolerance_degrees=0.65,
)
CFMESH_FINAL_OPTIMISATION_OPTIONS = ComparisonOptions(
    centroid_relative_tolerance=2.0e-5,
    centroid_absolute_tolerance=1.0e-10,
    volume_relative_tolerance=2.5e-4,
    face_normal_angle_tolerance_degrees=0.032,
)


def comparison_options_for_stage(stage: str | None) -> ComparisonOptions:
    """Select the audited geometry profile for a cfMesh workflow stage."""
    if stage in _SURFACE_UNTANGLING_STAGES:
        return CFMESH_SURFACE_UNTANGLING_OPTIONS
    if stage in _SURFACE_OPTIMISATION_STAGES:
        return CFMESH_SURFACE_OPTIMISATION_OPTIONS
    if stage in _WRAPPER_STAGES:
        return CFMESH_WRAPPER_OPTIONS
    if stage in _FINAL_OPTIMISATION_STAGES:
        return CFMESH_FINAL_OPTIMISATION_OPTIONS
    return ComparisonOptions()


def _openonda_commit() -> str | None:
    """Return the current source revision without making a report depend on Git."""
    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _summary(payload: dict[str, Any]) -> str:
    status = payload["status"].upper()
    stage = payload.get("stage") or "final"
    lines = [f"{status}  {stage}"]
    if payload["status"] == "pass":
        lines.append("Topology and geometry parity checks passed.")
        return "\n".join(lines) + "\n"
    reason = payload.get("reason") or payload.get("comparison", {}).get("first_failure")
    if reason:
        lines.append(f"First failure: {reason}")
    comparison = payload.get("comparison")
    if comparison:
        invariant_differences = comparison.get("invariant_differences", {})
        if invariant_differences:
            lines.append(
                "Level-A invariant differences: " + ", ".join(sorted(invariant_differences))
            )
        topology = comparison.get("topology", {})
        for name in (
            "adjacency_mismatches",
            "patch_incidence_mismatches",
            "face_topology_mismatches",
        ):
            if topology.get(name) is not None:
                lines.append(f"{name}: {topology[name]}")
    if payload.get("message"):
        lines.append(str(payload["message"]))
    return "\n".join(lines) + "\n"


def _write_payload(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "parity_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (directory / "parity_summary.txt").write_text(_summary(payload), encoding="utf-8")
    return payload


def run_parity(
    spec: ParitySpec,
    output_directory: Path | str,
    *,
    cfmesh_executable: Path | str | None = None,
    cfmesh_launcher: Path | str | None = None,
    stop_after: str | None = None,
    options: ComparisonOptions | None = None,
) -> dict[str, Any]:
    """Run one cfMesh/OpenONDA comparison and retain all audit artefacts.

    The caller supplies a fresh directory.  Refusing a non-empty directory is
    intentional: an old log or polyMesh must never be mistaken for this run.
    """
    output_directory = Path(output_directory).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty parity output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    effective_options = options or comparison_options_for_stage(stop_after)
    profile_name = (
        "caller_supplied"
        if options is not None
        else "cfmesh_surface_untangling"
        if stop_after in _SURFACE_UNTANGLING_STAGES
        else "cfmesh_surface_optimisation"
        if stop_after in _SURFACE_OPTIMISATION_STAGES
        else "cfmesh_wrapper"
        if stop_after in _WRAPPER_STAGES
        else "cfmesh_final_optimisation"
        if stop_after in _FINAL_OPTIMISATION_STAGES
        else "strict"
    )
    payload: dict[str, Any] = {
        "schema": "openonda-cfmesh-parity-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "error",
        "stage": stop_after,
        "effective_configuration": spec.effective_config(),
        "comparison_profile": profile_name,
        "comparison_options": effective_options.to_dict(),
        "openonda_git_commit": _openonda_commit(),
    }
    try:
        cfmesh_run, triangles = run_cfmesh(
            spec,
            output_directory / "cfmesh",
            executable=cfmesh_executable,
            launcher=cfmesh_launcher,
            stop_after=stop_after,
        )
    except CfMeshUnavailableError as error:
        payload.update(
            {
                "status": "blocked",
                "reason": "cfmesh_unavailable",
                "message": str(error),
            }
        )
        return _write_payload(output_directory, payload)
    except (CfMeshExecutionError, OSError, ValueError) as error:
        payload.update(
            {
                "status": "error",
                "reason": "cfmesh_execution",
                "message": str(error),
            }
        )
        return _write_payload(output_directory, payload)
    payload["cfmesh"] = {
        **dict(cfmesh_run.metadata),
        "case_directory": str(cfmesh_run.directory),
        "poly_mesh_directory": str(cfmesh_run.poly_mesh_directory),
        "log": str(cfmesh_run.log_path),
    }
    try:
        openonda_run = run_openonda(spec, output_directory / "openonda", stop_after=stop_after)
    except OpenONDAUnsupportedForParityError as error:
        payload.update(
            {
                "status": "partial",
                "reason": "openonda_checkpoint_or_feature_unavailable",
                "message": str(error),
            }
        )
        return _write_payload(output_directory, payload)
    except (OSError, ValueError, RuntimeError) as error:
        payload.update(
            {
                "status": "error",
                "reason": "openonda_execution",
                "message": str(error),
            }
        )
        return _write_payload(output_directory, payload)
    payload["openonda"] = {
        **dict(openonda_run.metadata),
        "case_directory": str(openonda_run.directory),
        "poly_mesh_directory": str(openonda_run.poly_mesh_directory),
    }
    comparison = compare_meshes(
        cfmesh_run.mesh,
        openonda_run.mesh,
        surface_triangles=triangles,
        options=effective_options,
    )
    payload["comparison"] = comparison.to_dict()
    payload["status"] = "pass" if comparison.passed else "fail"
    payload["reason"] = comparison.first_failure
    return _write_payload(output_directory, payload)


def run_stage_ladder(
    spec: ParitySpec,
    output_directory: Path | str,
    *,
    cfmesh_executable: Path | str | None = None,
    cfmesh_launcher: Path | str | None = None,
    stages: Sequence[str] = DEFAULT_WORKFLOW_STAGES,
    options: ComparisonOptions | None = None,
) -> dict[str, Any]:
    """Stop at the first failed checkpoint; never progress downstream silently."""
    output_directory = Path(output_directory).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty parity output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for stage in stages:
        record = run_parity(
            spec,
            output_directory / stage,
            cfmesh_executable=cfmesh_executable,
            cfmesh_launcher=cfmesh_launcher,
            stop_after=stage,
            options=options,
        )
        records.append(
            {"checkpoint": stage, "status": record["status"], "reason": record.get("reason")}
        )
        if record["status"] != "pass":
            break
    first_bad = next((item for item in records if item["status"] != "pass"), None)
    payload = {
        "schema": "openonda-cfmesh-parity-ladder-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "effective_configuration": spec.effective_config(),
        "openonda_git_commit": _openonda_commit(),
        "checkpoints": records,
        "first_bad_stage": first_bad["checkpoint"] if first_bad is not None else None,
        "reason": first_bad["reason"] if first_bad is not None else None,
        "status": "pass"
        if len(records) == len(stages) and records[-1]["status"] == "pass"
        else "fail",
    }
    return _write_payload(output_directory, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON parity specification")
    parser.add_argument("--output", required=True, type=Path, help="fresh artefact directory")
    parser.add_argument("--cfmesh-executable", type=Path, help="absolute cartesianMesh executable")
    parser.add_argument(
        "--cfmesh-launcher",
        type=Path,
        help="optional environment launcher (for example OpenFOAM.app's versioned openfoam command)",
    )
    parser.add_argument("--stop-after", help="cfMesh workflowControls.stopAfter checkpoint")
    parser.add_argument(
        "--checkpoint-ladder",
        dest="stage_ladder",
        action="store_true",
        help="run named checkpoints and stop at the first non-pass result",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = _parser().parse_args(argv)
    try:
        spec = load_parity_spec(args.spec)
        if args.stage_ladder:
            payload = run_stage_ladder(
                spec,
                args.output,
                cfmesh_executable=args.cfmesh_executable,
                cfmesh_launcher=args.cfmesh_launcher,
            )
        else:
            payload = run_parity(
                spec,
                args.output,
                cfmesh_executable=args.cfmesh_executable,
                cfmesh_launcher=args.cfmesh_launcher,
                stop_after=args.stop_after,
            )
    except (FileExistsError, OSError, ValueError) as error:
        print(f"parity-report: {error}", file=sys.stderr)
        return 2
    print(_summary(payload), end="")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())


__all__ = [
    "CFMESH_SURFACE_OPTIMISATION_OPTIONS",
    "CFMESH_FINAL_OPTIMISATION_OPTIONS",
    "CFMESH_WRAPPER_OPTIONS",
    "DEFAULT_WORKFLOW_STAGES",
    "comparison_options_for_stage",
    "main",
    "run_stage_ladder",
    "run_parity",
]
