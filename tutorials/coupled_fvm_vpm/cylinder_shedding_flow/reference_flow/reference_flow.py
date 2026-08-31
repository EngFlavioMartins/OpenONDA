#!/usr/bin/env python3
"""Launch one isolated member of the cylinder reference grid study.

Usage:
    python reference_flow.py --dx 0.041666666666666664 -name coarse

The solver's native case layout is preserved through two symlinks, while the
actual user-facing data go directly to ``solution/<name>/`` and
``samples/<name>/``.  A pre-existing case is deliberately rejected so a rerun
cannot append to or overwrite a completed grid-study member.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re


SOURCE_DIR = Path(__file__).resolve().parent
CASE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", required=True, type=float, help="Cylinder wall cell size in D")
    parser.add_argument("-name", "--name", required=True, dest="name", help="Output case name")
    return parser.parse_args()


def _ensure_case_directories(name: str) -> Path:
    if not CASE_NAME_PATTERN.fullmatch(name):
        raise SystemExit(
            "Case name must contain only letters, numbers, '.', '_', or '-' and may not start "
            "with punctuation."
        )
    solution = SOURCE_DIR / "solution" / name
    samples = SOURCE_DIR / "samples" / name
    for output in (solution, samples):
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise SystemExit(
                f"Refusing to overwrite existing grid-study output: {output}. "
                "Remove that named case explicitly before rerunning it."
            )

    case_root = SOURCE_DIR / ".grid-study-cases" / name
    case_root.mkdir(parents=True, exist_ok=True)
    for link_name, target in (("solution", solution), ("samples", samples)):
        link = case_root / link_name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise SystemExit(f"Grid-study case layout is inconsistent: {link}")
        else:
            try:
                link.symlink_to(target.resolve(), target_is_directory=True)
            except FileExistsError:
                if not link.is_symlink() or link.resolve() != target.resolve():
                    raise SystemExit(f"Grid-study case layout is inconsistent: {link}") from None
    return case_root


def main() -> None:
    arguments = _parse_arguments()
    if arguments.dx <= 0.0:
        raise SystemExit("--dx must be positive")
    case_root = _ensure_case_directories(arguments.name)

    # These values select the exact study mesh.  Keep the ordinary reference
    # setup as the single solver implementation rather than duplicating it.
    os.environ["OPENONDA_GRID_STUDY_DX"] = f"{arguments.dx:.17g}"
    os.environ["OPENONDA_GRID_STUDY_NAME"] = arguments.name
    os.environ["OPENONDA_REFERENCE_CASE_DIR"] = str(case_root)
    os.environ["OPENONDA_ENFORCE_SPANWISE_INVARIANCE"] = "0"

    from reference_flow_setup import main as run_reference

    run_reference()


if __name__ == "__main__":
    main()
