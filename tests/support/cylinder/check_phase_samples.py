#!/usr/bin/env python3
"""Audit recorded samples from the current ordinary cylinder case."""

import argparse
import json
from pathlib import Path

from .audit_saved_samples import audit_normal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("reference", "coupled"))
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
        ),
        help="ordinary coupled case directory; reference uses its reference_flow child",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="observed horizon (default: last force sample); slower samplers need only scheduled events",
    )
    parser.add_argument(
        "--output", type=Path, help="optional new JSON report; existing files are never overwritten"
    )
    options = parser.parse_args()
    if options.output is not None and options.output.exists():
        parser.error(f"Report already exists: {options.output}")
    result = audit_normal(options.case_dir, options.kind, options.end)
    if options.output is not None:
        with options.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, allow_nan=False))
