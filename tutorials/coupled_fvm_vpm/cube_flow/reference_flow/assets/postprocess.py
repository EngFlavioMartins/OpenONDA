#!/usr/bin/env python3
"""Rebuild reference_flow sampler products from archived FVM snapshots.

This script never constructs or evolves a live solver.  It replays the archive
under ``solution/`` through the samplers declared in ``setup``
and replaces only ``samples/``.
"""

from __future__ import annotations

from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

from setup import FVM_MESH, FVM_SETUP, SAMPLERS
from source.solvers.fvm.sampling.postprocess import PostProcess


def main() -> None:
    processed = PostProcess(
        case_dir=CASE_DIR,
        config=FVM_SETUP,
        samplers=SAMPLERS,
        mesh=FVM_MESH,
        overwrite=True,
    ).run()
    print(f"Rebuilt samples for {len(processed)} archived snapshots in {CASE_DIR / 'samples'}")


if __name__ == "__main__":
    main()
