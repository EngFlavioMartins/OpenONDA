#!/usr/bin/env python3
"""Export validated cube comparison fields and their portable plotting inputs."""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from . import postprocess as util


def export_inputs(destination: Path) -> list[str]:
    """Copy exact accepted fields and validation inputs without full FVM volumes."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Plot input destination must be empty")
    util.prepare_comparison_fields()
    util.validate_plot_inputs()
    reference = util.reference_run()
    manifest = util.comparison_manifest()
    coupled_mesh = util._fvm_artifacts(util.SOLUTION)[1]
    reference_mesh = util._fvm_artifacts(reference.solution)[1]
    files = {
        util.SOLUTION / "run_metadata.json",
        util.SOLUTION / "fvm_metadata.json",
        util.SOLUTION / "coupler_diagnostics.jsonl",
        coupled_mesh,
        reference.solution / "fvm_metadata.json",
        reference.solution / "diagnostics.jsonl",
        reference_mesh,
        reference.samples / "forces_history.csv",
        util.SAMPLES / "forces_history.csv",
        util.SAMPLES / "vpm_centreline.csv",
        util.SAMPLES / "vpm_offaxis_y075.csv",
    }
    for root in (util.SAMPLES, util.CASE_DIR / "reference_flow/samples"):
        files.update(path for path in root.rglob("*") if path.is_file())
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(files):
        relative = source.relative_to(util.CASE_DIR)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str) and value.startswith(str(util.CASE_DIR) + "/"):
            return str(Path(value).relative_to(util.CASE_DIR))
        return value

    output_manifest = destination / util.COMPARISON.relative_to(util.CASE_DIR) / "manifest.json"
    records = []
    for target in sorted(destination.rglob("*")):
        if target.is_file():
            records.append(
                {
                    "path": target.relative_to(destination).as_posix(),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
    marker = {
        "schema_version": 1,
        "method": util.PREPARATION_METHOD,
        "comparison": portable(manifest),
        "reference": {
            "name": reference.name,
            "solution": reference.solution.relative_to(util.CASE_DIR).as_posix(),
            "samples": reference.samples.relative_to(util.CASE_DIR).as_posix(),
            "target_spacing": reference.target_spacing,
        },
        "meshes": {
            "coupled": coupled_mesh.relative_to(util.CASE_DIR).as_posix(),
            "reference": reference_mesh.relative_to(util.CASE_DIR).as_posix(),
        },
        "files": records,
    }
    marker_path = output_manifest.with_name("prepared_inputs.json")
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return sorted(
        [row["path"] for row in records] + [marker_path.relative_to(destination).as_posix()]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    files = export_inputs(args.destination)
    print(f"Exported {len(files)} plotting inputs to {args.destination}")


if __name__ == "__main__":
    main()
