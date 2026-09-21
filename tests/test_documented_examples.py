"""Run the README and solver-guide examples verbatim in an empty directory."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "document,solver,directory,steps,time",
    [
        ("README.md", "fvm", "fvm-example", 5, 0.05),
        ("README.md", "vpm", "vpm-example", 5, 0.05),
        ("docs/fvm.md", "fvm", "first-fvm-case", 100, 0.1),
        ("docs/vpm.md", "vpm", "first-vpm-case", 2, 0.02),
    ],
)
def test_documented_example(tmp_path, document, solver, directory, steps, time):
    root = Path(__file__).resolve().parents[1]
    blocks = re.findall(r"```python\n(.*?)```", (root / document).read_text(), re.DOTALL)
    code = next(block for block in blocks if f"from openonda import {solver}\n" in block)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(root),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        TI_OFFLINE_CACHE_FILE_PATH=str(tmp_path / "taichi-cache"),
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    solution = tmp_path / directory / "solution"
    assert (solution / f"{solver}.pvd").is_file()
    metadata = json.loads((solution / f"{solver}_metadata.json").read_text())
    assert metadata["lifecycle"]["status"] == {"fvm": "complete", "vpm": "completed"}[solver]
    assert metadata["state"]["step"] == steps
    assert metadata["state"]["time"] == pytest.approx(time)
