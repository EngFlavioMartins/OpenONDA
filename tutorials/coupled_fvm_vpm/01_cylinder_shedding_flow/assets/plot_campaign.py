#!/usr/bin/env python3
"""Regenerate figures from an explicit or most recently updated campaign report."""

import argparse
import json
from pathlib import Path

from openonda.tutorial_runner import load_case_module

CASE_DIR = Path(__file__).resolve().parents[1]


def campaign_report(root: Path, run_dir: Path | None) -> tuple[Path, dict]:
    if run_dir is None:
        candidates = list(root.glob("*/pipeline_manifest.json"))
        if not candidates:
            raise FileNotFoundError(f"no campaign reports below {root}; provide --run-dir")
        path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    else:
        path = run_dir / "pipeline_manifest.json"
    report = json.loads(path.read_text())
    if report.get("pilot") or "reference" not in report:
        raise ValueError(f"{path} has no production grid report; finish the campaign first")
    coupled = any(row["kind"] == "coupled" for row in report.get("runs", []))
    if coupled and not all(key in report for key in ("coupled_grids", "span_profiles")):
        raise ValueError(f"{path} has an incomplete coupled comparison")
    return path.parent, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CASE_DIR / "study_results" / "cylinder")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    directory, report = campaign_report(args.root, args.run_dir)
    output = directory / "figures"
    output.mkdir(parents=True, exist_ok=True)
    if "coupled_grids" in report:
        pipeline = load_case_module(Path(__file__).parent, "run_pipeline")
        pipeline.plot_results(report, output, args.format)
    else:
        reference = load_case_module(CASE_DIR / "reference_flow", "postprocess_grid_study")
        reference.plot_forces(
            report["reference"]["grids"], output / f"reference_grid_forces.{args.format}"
        )
    print(f"Campaign: {directory}\nFigures: {output}")


if __name__ == "__main__":
    main()
