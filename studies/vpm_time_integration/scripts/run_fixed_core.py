#!/usr/bin/env python3
"""Rerun the exact frozen fixed-core probe into a disposable output folder."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "openonda-vpm-time-integration-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "openonda-vpm-time-integration-xdg"))

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "provenance" / "frozen_scripts" / "fixed_core_probe.py"
OUTPUT = ROOT / "data" / "fixed_core" / "reproduced"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("frozen_fixed_core_probe", FROZEN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {FROZEN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = OUTPUT
    module.main()
    print(f"reproduced frozen probe under {OUTPUT}")


if __name__ == "__main__":
    main()
