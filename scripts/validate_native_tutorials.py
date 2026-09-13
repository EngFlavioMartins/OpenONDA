"""Run maintained native FVM-VPM tutorials from isolated directories.

Validates that installed wheels can drive the tutorial launcher for native
solver combinations from a clean, unmodified checkout location.  Mirrors the
installed-wheel tutorial checks in ``.github/workflows/ci.yml``; the nightly
schedule executes this script against a freshly installed Conda environment.

Usage:

    python scripts/validate_native_tutorials.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

# Each entry is a catalog name plus a distinguishing workspace suffix.
_TUTORIALS = [
    ("coupled_fvm_vpm/cube_flow", "hybrid-flow"),
]


def run_one(tutorial: str, workspace: Path) -> None:
    executable = shutil.which("openonda")
    if executable is None:
        raise RuntimeError("openonda CLI not found on PATH")
    subprocess.run(
        [executable, "tutorial", "run", tutorial, "--workspace", str(workspace)],
        check=True,
    )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep isolated workspaces")
    args = parser.parse_args(arguments)

    base = Path(tempfile.mkdtemp(prefix="openonda-native-"))
    try:
        for tutorial, suffix in _TUTORIALS:
            workspace = base / suffix
            workspace.mkdir()
            print(f"running {tutorial} below {workspace}")
            run_one(tutorial, workspace)
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
