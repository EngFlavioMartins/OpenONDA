"""Run every test file in its own pytest process.

Taichi, CUDA and the floating-point environment are process-global, so the
suite runs one file per process. Arguments after the options go to pytest::

    python tests/run_isolated.py -m "not gpu and not slow"
    python tests/run_isolated.py --junit-dir results -m qualification
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

NO_TESTS_SELECTED = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--junit-dir", type=Path, help="write one JUnit XML file per test file")
    options, pytest_arguments = parser.parse_known_args()

    root = Path(__file__).resolve().parent
    files = sorted(root.rglob("test_*.py"))
    failed = []
    for path in files:
        command = [sys.executable, "-m", "pytest", str(path), "-p", "no:cacheprovider"]
        if options.junit_dir is not None:
            options.junit_dir.mkdir(parents=True, exist_ok=True)
            name = "_".join(path.relative_to(root).with_suffix(".xml").parts)
            command.append(f"--junitxml={options.junit_dir / name}")
        start = time.perf_counter()
        status = subprocess.run(command + pytest_arguments).returncode
        elapsed = time.perf_counter() - start
        print(
            f"[{'ok' if status in (0, NO_TESTS_SELECTED) else 'FAILED'}] "
            f"{path.relative_to(root.parent)} ({elapsed:.0f} s)",
            flush=True,
        )
        if status not in (0, NO_TESTS_SELECTED):
            failed.append(path.relative_to(root.parent))

    print(f"\n{len(files) - len(failed)} of {len(files)} test files passed")
    for path in failed:
        print(f"  failed: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
