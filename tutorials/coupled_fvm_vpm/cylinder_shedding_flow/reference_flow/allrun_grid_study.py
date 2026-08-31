#!/usr/bin/env python3
"""Run the four fixed body-fitted cylinder meshes, one after another.

Each command intentionally names its exact wall spacing.  Do not replace this
with an adaptive label: ``reference_flow.py`` records the requested
``dx, 2dx, 4dx, 12dx`` mesh contract in the corresponding metadata file.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SOURCE_DIR = Path(__file__).resolve().parent
CASES = (
    ("very_coarse", 1.0 / 12.0),
    ("coarse", 1.0 / 24.0),
    ("medium", 1.0 / 36.0),
    ("fine", 1.0 / 48.0),
)
MPI_RANKS = 6


def main() -> None:
    for name, dx in CASES:
        command = [
            "mpiexec",
            "--bind-to",
            "none",
            "-n",
            str(MPI_RANKS),
            sys.executable,
            "-u",
            "reference_flow.py",
            "--dx",
            f"{dx:.17g}",
            "-name",
            name,
        ]
        print("\n===== GRID STUDY: " + " ".join(command) + " =====\n", flush=True)
        subprocess.run(command, cwd=SOURCE_DIR, check=True)

    subprocess.run(
        [sys.executable, "../assets/analyse_grid_study.py", "--reference-dir", str(SOURCE_DIR)],
        cwd=SOURCE_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
