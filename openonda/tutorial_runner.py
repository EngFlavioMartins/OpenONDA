"""Run a module from a user-owned tutorial without changing Python paths.

Usage: ``python -m openonda.tutorial_runner CASE_DIRECTORY MODULE [ARGS...]``.
The case directory is a local package, so relative imports resolve its edited
configuration and assets rather than the immutable installed template.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
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


def _archive_outputs(directory: Path) -> None:
    """Archive this case's output before a fresh setup, retaining its inputs."""
    from .runtime import detected_world_size

    if detected_world_size() != 1:
        raise RuntimeError("Launch --fresh outside mpiexec; the solver establishes MPI")
    paths = [directory / name for name in ("solution", "solutions", "samples", "figures")]
    paths.extend(sorted(directory.glob("*.log")))
    paths = [path for path in paths if path.exists() or path.is_symlink()]
    if not paths:
        return
    previous = directory / "previous_runs"
    previous.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=previous))
    moved = []
    try:
        for path in paths:
            path.rename(archive / path.name)
            moved.append(path)
    except OSError:
        for path in reversed(moved):
            (archive / path.name).rename(path)
        archive.rmdir()
        raise
    print(f"Previous outputs archived in {archive}", flush=True)


def _run_setup(directory: Path, arguments: list[str]) -> int:
    """Keep a portable case lock while the setup launches and waits for MPI."""
    import fcntl

    with (directory / ".openonda-run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"OpenONDA case is already running: {directory}", file=sys.stderr)
            return 75
        if "--fresh" in arguments:
            _archive_outputs(directory)
            arguments = [argument for argument in arguments if argument != "--fresh"]
        with subprocess.Popen(
            [sys.executable, str(directory / "setup.py"), *arguments], cwd=directory
        ) as child:
            while True:
                try:
                    return child.wait()
                except KeyboardInterrupt:
                    # The child receives the same terminal interrupt. Retain
                    # ownership until its solver finishes stopping safely.
                    continue


def main(arguments: list[str] | None = None) -> int:
    """Execute a case module with the remaining command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("module")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.module == "setup":
        return _run_setup(args.directory.resolve(), args.arguments)
    # Register the package without importing the executable module twice.
    parent, _, _ = args.module.rpartition(".")
    package = load_case_module(args.directory, parent)
    module_name = package.__name__ + "." + args.module.rsplit(".", 1)[-1]
    sys.argv = [str(args.directory / (args.module.replace(".", "/") + ".py")), *args.arguments]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
