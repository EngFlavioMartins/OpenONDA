#!/usr/bin/env python3
"""Install OpenONDA into this Python environment and verify it outside the checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main(arguments=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gaussian-mesh-cuda12",
        action="store_true",
        help="install the optional CUDA 12 Gaussian mesh backend and verify its GPU runtime",
    )
    parser.add_argument(
        "--dev", action="store_true", help="also install the lint and packaging tools"
    )
    parser.add_argument("--with-environment", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        parser.error(
            f"OpenONDA requires CPython 3.11 (this is {sys.implementation.name} "
            f"{'.'.join(map(str, sys.version_info[:3]))}); run 'source install.sh' to create a "
            "3.11 environment"
        )
    root = Path(__file__).resolve().parent
    install = [sys.executable, "-m", "pip", "install"]
    extras = (["dev"] if args.dev else []) + (
        ["gaussian-mesh-cuda12"] if args.gaussian_mesh_cuda12 else []
    )
    target = str(root) + ("[" + ",".join(extras) + "]" if extras else "")
    install += ["-e", target]
    verify = [sys.executable, "-I", "-m", "openonda.verify_install"]
    if args.gaussian_mesh_cuda12:
        verify.append("--with-gaussian-mesh")
    if args.with_environment:
        verify.append("--with-environment")
    # Neither the current directory nor PYTHONPATH can make verification pass
    # by accidentally importing the source checkout instead of the installation.
    with tempfile.TemporaryDirectory(prefix="openonda-install-") as directory:
        for command in (install, [sys.executable, "-m", "pip", "check"], verify):
            print(f"+ {shlex.join(command)}", flush=True)
            result = subprocess.run(command, cwd=directory, check=False)
            if result.returncode:
                return result.returncode
    print(f"OpenONDA is installed and verified for {sys.executable}.")
    print("Run a first case: cd tutorials/fvm/taylor_green && ./allrun.sh && ./allplot.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
