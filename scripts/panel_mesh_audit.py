#!/usr/bin/env python3
"""Run the panel-solver STL preflight audit on a body mesh from the command line.

Loads an STL file, runs
:func:`source.solvers.vpm.boundary_elements.panels.geometry.stl_audit.audit_stl_mesh`,
prints a summary, and optionally writes the machine-readable JSON report.
Exit status is non-zero when the audit fails (or warns, under ``--strict``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from source.solvers.vpm.boundary_elements.panels.geometry.stl_audit import (  # noqa: E402
    StlAuditError,
    audit_stl_mesh,
    write_audit_report_json,
)
from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import load_stl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", type=Path, help="Path to the STL file to audit")
    parser.add_argument(
        "--max-panels", type=int, default=None, help="Reject above this triangle count"
    )
    parser.add_argument(
        "--expected-components",
        type=int,
        default=None,
        help="Required number of disconnected watertight bodies (default: 1)",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the JSON report here")
    parser.add_argument(
        "--strict", action="store_true", help="Treat a passing audit with warnings as failure"
    )
    args = parser.parse_args()

    vertex_position, _ = load_stl(str(args.stl))
    try:
        report = audit_stl_mesh(
            vertex_position,
            max_panels=args.max_panels,
            expected_components=args.expected_components,
        )
    except StlAuditError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"{args.stl}: {report['disposition']}")
    print(f"  triangles             : {report['n_triangles']}")
    print(f"  components            : {report['component_count']}")
    print(f"  signed volume(s)      : {report['component_signed_volumes']}")
    print(
        f"  area min/mean/max     : {report['area_min']:.3e} / {report['area_mean']:.3e} / {report['area_max']:.3e}"
    )
    print(f"  max aspect ratio      : {report['aspect_ratio_max']:.1f}")
    for warning in report["warnings"]:
        print(f"  warning: {warning}")

    if args.json is not None:
        write_audit_report_json(report, str(args.json))
        print(f"  report written to     : {args.json}")

    if args.strict and report["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
