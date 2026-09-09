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
        "--dev", action="store_true", help="editable install with development tools"
    )
    args = parser.parse_args(arguments)
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        parser.error("OpenONDA requires Python 3.11–3.13; use Python 3.11 on Intel macOS")
    root = Path(__file__).resolve().parent
    install = [sys.executable, "-m", "pip", "install"]
    install += ["-e", f"{root}[dev]"] if args.dev else [str(root)]
    verify = [sys.executable, "-I", "-m", "openonda.verify_install"]
    if not args.dev:
        verify.append("--require-site-packages")
    # Neither the current directory nor PYTHONPATH can make verification pass
    # by accidentally importing the source checkout instead of the installation.
    with tempfile.TemporaryDirectory(prefix="openonda-install-") as directory:
        for command in (install, [sys.executable, "-m", "pip", "check"], verify):
            print(f"+ {shlex.join(command)}", flush=True)
            result = subprocess.run(command, cwd=directory, check=False)
            if result.returncode:
                return result.returncode
    print(f"OpenONDA is installed and verified for {sys.executable}.")
    print("Use this environment's python to run your case: python setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
