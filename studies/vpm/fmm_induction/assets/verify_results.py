#!/usr/bin/env python3
"""Read-only verification of the generated FMM qualification records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
REQUIRED_KERNELS = {
    "GAUSSIAN",
    "HIGH_ORDER_GAUSSIAN",
    "SUPER_GAUSSIAN",
    "WINCKELMANS",
}
REQUIRED_COUNTS = {1000, 4000, 14080, 35000, 70200}
REQUIRED_DISTRIBUTIONS = {
    "uniform",
    "clustered",
    "elongated",
    "ring",
    "two_rings",
    "leapfrog",
    "rotor",
}
ACCURACY_LIMITS = {
    "velocity_relative_l2": 5.0e-3,
    "gradient_relative_l2": 1.0e-2,
    "rate_relative_l2": 1.5e-2,
    "rate_particle_p95": 3.0e-2,
    "raw_rate_defect": 1.0e-3,
}


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path.name}: {error}") from error


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except OSError as error:
        raise ValueError(f"cannot read {path.name}: {error}") from error


def _is_false(value: object) -> bool:
    return value is False or value == "False" or value == "false"


def _is_true(value: object) -> bool:
    return value is True or value == "True" or value == "true"


def _as_float(row: dict[str, object], field: str) -> float | None:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return None


def _as_int(row: dict[str, object], field: str) -> int | None:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        return None


def _check_result_provenance(
    results_dir: Path, source_commit: str, failures: list[str]
) -> tuple[dict[Path, object], dict[Path, list[dict[str, str]]]]:
    json_results: dict[Path, object] = {}
    csv_results: dict[Path, list[dict[str, str]]] = {}
    result_paths = sorted(results_dir.rglob("*.json")) + sorted(results_dir.rglob("*.csv"))
    for path in result_paths:
        if path.name == "manifest.json":
            continue
        if path.suffix == ".json":
            try:
                value = _read_json(path)
            except ValueError as error:
                failures.append(str(error))
                continue
            json_results[path] = value
            if not isinstance(value, dict):
                failures.append(f"{path.name} must contain a JSON object")
                continue
            if value.get("source_commit") != source_commit:
                failures.append(f"{path.name} has a mismatched source_commit")
            if "source_dirty" in value and not _is_false(value["source_dirty"]):
                failures.append(f"{path.name} has source_dirty=true")
        else:
            try:
                rows = _read_csv(path)
            except ValueError as error:
                failures.append(str(error))
                continue
            csv_results[path] = rows
            if not rows:
                failures.append(f"{path.name} contains no result rows")
            for row_number, row in enumerate(rows, start=2):
                if row.get("source_commit") != source_commit:
                    failures.append(f"{path.name} row {row_number} has a mismatched source_commit")
                if "source_dirty" in row and not _is_false(row["source_dirty"]):
                    failures.append(f"{path.name} row {row_number} has source_dirty=true")
    return json_results, csv_results


def _check_accuracy_rows(
    csv_results: dict[Path, list[dict[str, str]]], failures: list[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    accuracy_rows: list[dict[str, str]] = []
    scaling_rows: list[dict[str, str]] = []
    for path, rows in csv_results.items():
        if path.name == "accuracy.csv":
            accuracy_rows.extend(row for row in rows if row.get("method", "").upper() == "FMM")
        if path.name == "scaling.csv":
            scaling_rows.extend(row for row in rows if row.get("method", "").upper() == "FMM")
    if not accuracy_rows:
        failures.append("no FMM accuracy rows found")
    if not scaling_rows:
        failures.append("no FMM scaling rows found")

    kernels = {row.get("kernel", "").upper() for row in accuracy_rows}
    missing_kernels = REQUIRED_KERNELS - kernels
    if missing_kernels:
        failures.append(f"missing FMM kernels: {sorted(missing_kernels)}")
    backends = {row.get("backend", "").upper() for row in accuracy_rows}
    for backend in ("CPU", "VULKAN"):
        if backend not in backends:
            failures.append(f"missing FMM {backend} kernel-accuracy rows")

    counts = {_as_int(row, "count") for row in scaling_rows}
    missing_counts = REQUIRED_COUNTS - counts
    if missing_counts:
        failures.append(f"missing FMM scaling counts: {sorted(missing_counts)}")
    distributions = {row.get("distribution", "") for row in accuracy_rows}
    missing_distributions = REQUIRED_DISTRIBUTIONS - distributions
    if missing_distributions:
        failures.append(f"missing FMM distributions: {sorted(missing_distributions)}")

    for row in accuracy_rows + scaling_rows:
        identity = f"{row.get('backend', '?')}/{row.get('kernel', '?')}/{row.get('count', '?')}"
        for field, limit in ACCURACY_LIMITS.items():
            value = _as_float(row, field)
            if value is None:
                failures.append(f"FMM accuracy row {identity} has no valid {field}")
            elif value > limit:
                failures.append(f"FMM accuracy row {identity} exceeds {field}={limit:g}")
        for field in ("host_particle_transfers", "direct_strength_rate_fallbacks"):
            value = _as_int(row, field)
            if value is None:
                failures.append(f"FMM accuracy row {identity} has no valid {field}")
            elif value != 0:
                failures.append(f"FMM accuracy row {identity} has nonzero {field}")
    return accuracy_rows, scaling_rows


def _check_comparisons(json_results: dict[Path, object], failures: list[str]) -> None:
    direct_paths = sorted(path for path in json_results if path.name.startswith("direct_fmm_"))
    if not direct_paths:
        failures.append("direct-versus-FMM trajectory comparison is missing")
    for path in direct_paths:
        value = json_results[path]
        if not isinstance(value, dict):
            continue
        for field in (
            "comparison_gate_passed",
            "fmm_host_particle_transfers",
            "fmm_direct_strength_rate_fallbacks",
        ):
            if field not in value:
                failures.append(f"{path.name} is missing {field}")
        if not _is_true(value.get("comparison_gate_passed")):
            failures.append(f"{path.name} comparison_gate_passed is not true")
        if value.get("fmm_host_particle_transfers") != 0:
            failures.append(f"{path.name} has nonzero fmm_host_particle_transfers")
        if value.get("fmm_direct_strength_rate_fallbacks") != 0:
            failures.append(f"{path.name} has nonzero fmm_direct_strength_rate_fallbacks")

    required_comparisons = {
        "coupled_vlm_comparison.json": (
            "comparison_gate_passed",
            "fmm_zero_host_transfer_passed",
            "fmm_zero_fallback_passed",
            "fmm_scheduled_output_passed",
        ),
        "coupled_fvm_comparison.json": (
            "comparison_gate_passed",
            "finite_fields",
            "fmm_host_particle_transfers",
            "fmm_direct_strength_rate_fallbacks",
        ),
    }
    for filename, fields in required_comparisons.items():
        paths = [path for path in json_results if path.name == filename]
        if not paths:
            failures.append(f"{filename} is missing")
            continue
        value = json_results[paths[0]]
        if not isinstance(value, dict):
            continue
        for field in fields:
            if field not in value:
                failures.append(f"{filename} is missing {field}")
        if not _is_true(value.get("comparison_gate_passed")):
            failures.append(f"{filename} comparison_gate_passed is not true")
        for field in (
            fields[1:] if filename == "coupled_vlm_comparison.json" else ("finite_fields",)
        ):
            if not _is_true(value.get(field)):
                failures.append(f"{filename} {field} is not true")
        if filename == "coupled_fvm_comparison.json":
            if value.get("fmm_host_particle_transfers") != 0:
                failures.append(f"{filename} has nonzero fmm_host_particle_transfers")
            if value.get("fmm_direct_strength_rate_fallbacks") != 0:
                failures.append(f"{filename} has nonzero fmm_direct_strength_rate_fallbacks")


def verify_results(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Return every failed verification condition without modifying any file."""
    failures: list[str] = []
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    try:
        manifest = _read_json(manifest_path)
    except ValueError as error:
        return [str(error)]
    if not isinstance(manifest, dict):
        return ["manifest.json must contain a JSON object"]
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not SHA_PATTERN.fullmatch(source_commit):
        failures.append("manifest source_commit is not a 40-character hexadecimal Git SHA")
        source_commit = ""
    if not _is_false(manifest.get("source_dirty")):
        failures.append("manifest source_dirty is not false")
    json_results, csv_results = _check_result_provenance(results_dir, source_commit, failures)
    _check_accuracy_rows(csv_results, failures)
    _check_comparisons(json_results, failures)
    return failures


def main(results_dir: Path = RESULTS_DIR) -> int:
    failures = verify_results(results_dir)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("FMM result verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
