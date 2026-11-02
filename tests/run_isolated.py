"""Run every test file in its own pytest process.

Taichi, CUDA and the floating-point environment are process-global, so the
suite runs one file per process. Arguments that name test files, test
directories or test node ids below ``tests/`` select what runs (every test
file when there are none); the other arguments go to pytest::

    python tests/run_isolated.py -m "not gpu and not slow"
    python tests/run_isolated.py --junit-dir results -m qualification
    python tests/run_isolated.py tests/fvm/test_mesh_motion.py tests/tutorials
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

NO_TESTS_SELECTED = 5


def split_targets(arguments: list[str], root: Path) -> tuple[list[str], list[str]]:
    """Return the test targets (one per process) and the remaining pytest arguments.

    A target is an argument naming an existing test file, a node id in one, or a
    directory below ``root``; a directory expands to its test files. Without
    targets every test file below ``root`` runs.
    """
    targets = []
    pytest_arguments = []
    for argument in arguments:
        path = Path(argument.split("::")[0]).resolve()
        if argument.startswith("-") or not path.exists() or not path.is_relative_to(root):
            pytest_arguments.append(argument)
        elif path.is_dir():
            targets.extend(str(item) for item in sorted(path.rglob("test_*.py")))
        else:
            targets.append(str(path) + argument[len(argument.split("::")[0]) :])
    if not targets:
        targets = [str(item) for item in sorted(root.rglob("test_*.py"))]
    return targets, pytest_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--junit-dir", type=Path, help="write one JUnit XML file per test file")
    options, pytest_arguments = parser.parse_known_args()

    root = Path(__file__).resolve().parent
    targets, pytest_arguments = split_targets(pytest_arguments, root)
    failed = []
    for target in targets:
        path = Path(target.split("::")[0])
        command = [sys.executable, "-m", "pytest", target, "-p", "no:cacheprovider"]
        if options.junit_dir is not None:
            options.junit_dir.mkdir(parents=True, exist_ok=True)
            name = "_".join(path.relative_to(root).with_suffix(".xml").parts)
            command.append(f"--junitxml={options.junit_dir / name}")
        start = time.perf_counter()
        status = subprocess.run(command + pytest_arguments).returncode
        elapsed = time.perf_counter() - start
        print(
            f"[{'ok' if status in (0, NO_TESTS_SELECTED) else 'FAILED'}] "
            f"{Path(target).relative_to(root.parent)} ({elapsed:.0f} s)",
            flush=True,
        )
        if status not in (0, NO_TESTS_SELECTED):
            failed.append(Path(target).relative_to(root.parent))

    print(f"\n{len(targets) - len(failed)} of {len(targets)} test files passed")
    for path in failed:
        print(f"  failed: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
