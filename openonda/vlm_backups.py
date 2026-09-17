"""Rebuild VLM ParaView frames from existing VPM checkpoints.

Run ``python -m openonda.vlm_backups path/to/solution`` to rebuild
``solution/vlm/`` frames and the root-level ``solution/vlm.pvd`` collection.
New simulations publish these companions at the VPM output cadence.
"""

import argparse
from pathlib import Path

from source.solvers.vpm.io.logging import Logging
from source.solvers.vpm.io.vlm_backup import export_vlm_backup


def main():
    """Export one checkpoint or all checkpoints beneath a solution directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="checkpoint .h5 file or solution directory")
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f"path does not exist: {args.path}")
    checkpoints = sorted(args.path.rglob("vpm_*.h5")) if args.path.is_dir() else [args.path]
    count = sum(export_vlm_backup(path) is not None for path in checkpoints)
    Logging.record(
        "VLM surface output", ("checkpoints", count), ("time series", "vlm.pvd"), flush=True
    )


if __name__ == "__main__":
    main()
