#!/usr/bin/env python3
"""Rebuild referenceFlow sampler products from archived FVM snapshots.

This script never constructs or evolves a live solver.  It replays the archive
under ``solution/`` through the samplers declared in ``referenceFlow_setup``
and replaces only ``samples/``.
"""

from __future__ import annotations

from pathlib import Path

from referenceFlow_setup import FVM_MESH, FVM_SETUP, SAMPLERS
from source.solvers.FVM.sampling.postprocess import PostProcess


def main() -> None:
    case_dir = Path(__file__).resolve().parent
    processed = PostProcess(
        case_dir=case_dir,
        config=FVM_SETUP,
        samplers=SAMPLERS,
        mesh=FVM_MESH,
        overwrite=True,
    ).run()
    print(f"Rebuilt samples for {len(processed)} archived snapshots in {case_dir / 'samples'}")


if __name__ == "__main__":
    main()