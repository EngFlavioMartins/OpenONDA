#!/usr/bin/env python3
"""Preserve one validated reference variant before reusing its work directory."""

from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path
import shutil


EXPECTED = {
    "g0": {"grid": "g0", "domain": "baseline", "dt_scale": 1.0},
    "g1": {"grid": "g1", "domain": "baseline", "dt_scale": 1.0},
    "g2": {"grid": "g2", "domain": "baseline", "dt_scale": 1.0},
    "g1_half_dt": {"grid": "g1", "domain": "baseline", "dt_scale": 0.5},
    "g1_large_domain": {"grid": "g1", "domain": "large", "dt_scale": 1.0},
}

ARTIFACTS = {
    "forces_history.csv": Path("samples/forces_history.csv"),
    "performance.jsonl": Path("solution/performance.jsonl"),
    "metadata.json": Path("solution/benchmark_metadata.json"),
    "diagnostics.json": Path("solution/reference_diagnostics.json"),
    "sample_quality.json": Path("solution/sample_quality.json"),
}


def _validated(case: Path, tag: str) -> None:
    diagnostics = json.loads(
        (case / "solution" / "reference_diagnostics.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (case / "solution" / "sample_quality.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (case / "solution" / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    expected = EXPECTED[tag]
    actual = {
        "grid": metadata["mesh"]["grid"],
        "domain": metadata["mesh"]["domain"],
        "dt_scale": float(metadata["time"]["dt_scale"]),
    }
    if actual != expected:
        raise ValueError(f"{tag} metadata mismatch: expected {expected}, found {actual}")
    if diagnostics.get("status") != "statistically_ready":
        raise ValueError(f"{tag} force history is not statistically ready")
    if quality.get("status") != "passed":
        raise ValueError(f"{tag} sample audit has not passed")


def preserve(case: Path, destination: Path, tag: str) -> list[Path]:
    """Copy the compact convergence authority, refusing conflicting overwrites."""
    _validated(case, tag)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, relative_source in ARTIFACTS.items():
        source = case / relative_source
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / f"{tag}_{suffix}"
        if target.exists() and not filecmp.cmp(source, target, shallow=False):
            raise FileExistsError(
                f"Refusing to replace different verification evidence: {target}"
            )
        if not target.exists():
            shutil.copy2(source, target)
        written.append(target)
    return written


def main() -> None:
    tutorial = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", choices=tuple(EXPECTED))
    parser.add_argument(
        "--case",
        type=Path,
        default=tutorial / "reference_flow",
        help="validated case containing solution/ and samples/",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=tutorial / "reference_flow" / "solution" / "verification",
    )
    args = parser.parse_args()
    outputs = preserve(args.case.resolve(), args.destination.resolve(), args.tag)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
