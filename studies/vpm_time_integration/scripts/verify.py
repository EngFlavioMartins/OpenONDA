#!/usr/bin/env python3
"""Build or verify the VPM time-integration study provenance manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

REPOSITORY = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "manifest.json"

COPIES = (
    ("docs/reviews/vpm-stability-theory/results.json", "data/fixed_core/results.json"),
    ("docs/reviews/vpm-stability-theory/temporal.csv", "data/fixed_core/temporal.csv"),
    ("docs/reviews/vpm-stability-theory/geometry.csv", "data/fixed_core/geometry.csv"),
    ("docs/reviews/vpm-stability-theory/tangent.csv", "data/fixed_core/tangent.csv"),
    (
        "docs/reviews/vpm-stability-theory/derivative_checks.csv",
        "data/fixed_core/derivative_checks.csv",
    ),
    ("docs/reviews/vpm-time-integration-assessment/results.json", "data/comparator/results.json"),
    ("docs/reviews/vpm-time-integration-assessment/results.csv", "data/comparator/results.csv"),
    (
        "docs/reviews/vpm-stability-theory/probe.py",
        "provenance/frozen_scripts/fixed_core_probe.py",
    ),
    (
        "docs/reviews/vpm-time-integration-assessment/assessment.py",
        "provenance/frozen_scripts/comparator_assessment.py",
    ),
    (
        "docs/reviews/2026-09-vpm-stability-theory.md",
        "provenance/reviews/fixed_core_review.md",
    ),
    (
        "docs/reviews/2026-09-vpm-time-integration-assessment.md",
        "provenance/reviews/comparator_review.md",
    ),
    (
        "docs/reviews/vpm-stability-theory/theorem-source-ledger.md",
        "provenance/reviews/fixed_core_theorem_source_ledger.md",
    ),
    (
        "docs/reviews/vpm-stability-theory/equation-map.md",
        "provenance/reviews/fixed_core_equation_map.md",
    ),
)

AUTHORED = (
    ".gitignore",
    "README.md",
    "METHODS.md",
    "EQUATIONS.md",
    "SOURCES.md",
    "APPENDIX_FIGURE_LIST.md",
    "WRITER_BRIEF.md",
    "provenance/kernel_snapshot.json",
    "scripts/style.py",
    "scripts/plot_results.py",
    "scripts/run_fixed_core.py",
    "scripts/run_comparator.py",
    "scripts/verify.py",
)

FIGURE_INPUT = {
    "fixed_core_temporal_convergence": ["data/fixed_core/temporal.csv"],
    "fixed_core_order_summary": ["data/fixed_core/temporal.csv"],
    "geometry_moment_defects": ["data/fixed_core/geometry.csv"],
    "tangent_excess": ["data/fixed_core/tangent.csv"],
    "comparator_convergence": ["data/comparator/results.json"],
    "oscillator_stability": ["data/comparator/results.json", "scripts/plot_results.py"],
}

EXPECTED_ROWS = {
    "data/fixed_core/temporal.csv": 144,
    "data/fixed_core/geometry.csv": 12,
    "data/fixed_core/tangent.csv": 12,
    "data/fixed_core/derivative_checks.csv": 8,
    "data/comparator/results.csv": 42,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def figure_entries() -> list[dict]:
    entries = []
    for stem, inputs in FIGURE_INPUT.items():
        for suffix in ("png", "pdf", "svg"):
            relative = f"figures/{stem}.{suffix}"
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append(
                {
                    "path": relative,
                    "sha256": sha256(path),
                    "generated_by": "scripts/plot_results.py",
                    "inputs": inputs,
                }
            )
    return entries


def build_manifest() -> dict:
    exact = []
    for source_relative, packaged_relative in COPIES:
        source = REPOSITORY / source_relative
        packaged = ROOT / packaged_relative
        source_hash = sha256(source)
        packaged_hash = sha256(packaged)
        if source_hash != packaged_hash:
            raise RuntimeError(f"copy differs from source: {packaged_relative}")
        exact.append(
            {
                "original": source_relative,
                "original_sha256": source_hash,
                "packaged": packaged_relative,
                "packaged_sha256": packaged_hash,
                "byte_identical": True,
            }
        )
    authored = [{"path": relative, "sha256": sha256(ROOT / relative)} for relative in AUTHORED]
    return {
        "schema_version": 1,
        "study": "VPM time integration",
        "packaged_date": "2026-09-10",
        "repository_head_at_packaging": git_head(),
        "scientific_status": "accepted bounded accuracy and conditional perturbation study",
        "execution_boundary": "frozen production-convention standalone NumPy probes; not native production-solver execution",
        "post_study_source_note": (
            "Repository Gaussian-kernel source files were corrected after the accepted fixed-core study. "
            "Later kernel-defect and patch-impact diagnostics are outside this package and are not inputs."
        ),
        "exact_copies": exact,
        "authored_files": authored,
        "derived_figures": figure_entries(),
    }


def validate_signatures() -> None:
    for stem in FIGURE_INPUT:
        png = ROOT / "figures" / f"{stem}.png"
        pdf = ROOT / "figures" / f"{stem}.pdf"
        svg = ROOT / "figures" / f"{stem}.svg"
        if not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"invalid PNG signature: {png}")
        if not pdf.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError(f"invalid PDF signature: {pdf}")
        if "<svg" not in svg.read_text(encoding="utf-8", errors="strict")[:1000]:
            raise RuntimeError(f"invalid SVG header: {svg}")


def validate_results() -> None:
    for relative, expected in EXPECTED_ROWS.items():
        with (ROOT / relative).open(newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count != expected:
            raise RuntimeError(f"{relative}: expected {expected} rows, found {count}")
    fixed = json.loads((ROOT / "data/fixed_core/results.json").read_text())
    if len(fixed["temporal"]) != 144 or len(fixed["geometry"]) != 12 or len(fixed["tangent"]) != 12:
        raise RuntimeError("fixed-core JSON row counts do not match the accepted payload")
    comparator = json.loads((ROOT / "data/comparator/results.json").read_text())
    if len(comparator["rows"]) != 42:
        raise RuntimeError("comparator JSON does not contain 42 accepted rows")


def verify_manifest(manifest: dict) -> None:
    for entry in manifest["exact_copies"]:
        packaged = ROOT / entry["packaged"]
        if sha256(packaged) != entry["packaged_sha256"]:
            raise RuntimeError(f"packaged copy hash changed: {entry['packaged']}")
        if entry["original_sha256"] != entry["packaged_sha256"]:
            raise RuntimeError(f"copy mismatch in manifest: {entry['packaged']}")
    for group in ("authored_files", "derived_figures"):
        for entry in manifest[group]:
            if sha256(ROOT / entry["path"]) != entry["sha256"]:
                raise RuntimeError(f"hash changed: {entry['path']}")
    validate_results()
    validate_signatures()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write the manifest after checking all source copies",
    )
    args = parser.parse_args()
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=2) + "\n")
        print(f"wrote {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    verify_manifest(manifest)
    print(
        "verified 13 exact copies, 13 authored files, 18 figure files, "
        "and accepted raw-result row counts"
    )


if __name__ == "__main__":
    main()
