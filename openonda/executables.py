"""Discover optional rendering tools outside user-authored tutorial scripts."""

import os
from pathlib import Path
import shutil
import sys


def find_executable(name: str, explicit: str | None = None) -> str:
    """Find a tool in PATH or its conventional macOS application installation."""
    if explicit:
        found = shutil.which(explicit)
        if found:
            return found
        raise FileNotFoundError(f"Requested {name} executable does not exist: {explicit}")
    if name == "pvpython":
        override = os.environ.get("OPENONDA_PARAVIEW_PYTHON") or os.environ.get("PVPYTHON")
        if override:
            return find_executable(name, override)
    found = shutil.which(name)
    if found:
        return found
    if sys.platform == "darwin":
        if name == "pvpython":
            for root in (Path("/Applications"), Path.home() / "Applications"):
                candidates = sorted(root.glob("ParaView*.app/Contents/bin/pvpython"), reverse=True)
                for candidate in candidates:
                    if os.access(candidate, os.X_OK):
                        return str(candidate)
        if name in ("pdflatex", "latex", "dvipng"):
            candidate = Path("/Library/TeX/texbin") / name
            if os.access(candidate, os.X_OK):
                return str(candidate)
    raise FileNotFoundError(
        f"Optional rendering tool {name!r} is not installed or available in PATH"
    )
