"""Run a module from a user-owned tutorial without changing Python paths.

Usage: ``python -m openonda.tutorial_runner CASE_DIRECTORY MODULE [ARGS...]``.
The case directory is a local package, so relative imports resolve its edited
configuration and assets rather than the immutable installed template.
"""

from __future__ import annotations

import argparse
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import runpy
import sys
from types import ModuleType


def load_case_module(directory: str | Path, module: str = "setup") -> ModuleType:
    """Load a local tutorial module, retaining ordinary package-relative imports."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Tutorial directory does not exist: {directory}")
    # One namespace per case permits comparing several workspaces in a process.
    import hashlib

    name = "_openonda_case_" + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    return importlib.import_module(f"{name}.{module}" if module else name)


def case_package(directory: str | Path) -> str:
    """Register a local package for a directly executed tutorial script."""
    return load_case_module(directory, "").__name__


def main(arguments: list[str] | None = None) -> int:
    """Execute a case module with the remaining command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("module")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    # Register the package without importing the executable module twice.
    parent, _, _ = args.module.rpartition(".")
    package = load_case_module(args.directory, parent) if parent else None
    if package is None:
        # An empty module selects the namespace package itself.
        package = load_case_module(args.directory, "")
    module_name = package.__name__ + "." + args.module.rsplit(".", 1)[-1]
    sys.argv = [str(args.directory / (args.module.replace(".", "/") + ".py")), *args.arguments]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
