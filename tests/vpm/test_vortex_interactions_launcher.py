"""Regression tests for the vortexInteractions study launcher."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def test_completed_staged_case_replaces_stale_rejected_result(tmp_path: Path) -> None:
    tutorial = Path(__file__).parents[2] / "tutorials/VPM/vortexInteractions"
    run_root = tmp_path / "solution"
    stale_case = run_root / "leapfrog_baseline"
    stale_case.mkdir(parents=True)
    (stale_case / "rejected_state.h5").touch()

    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
shift
output_root=""
case_name=""
while (($#)); do
    case "$1" in
        --output-root) output_root="$2"; shift 2 ;;
        --name) case_name="$2"; shift 2 ;;
        *) shift ;;
    esac
done
mkdir -p "$output_root/$case_name"
printf '%s\\n' \\
    '{"status": "completed", "completed_steps": 1, "requested_steps": 1}' \\
    > "$output_root/$case_name/run_manifest.json"
"""
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "OPENONDA_PYTHON": str(fake_python),
            "RUN_ROOT": str(run_root),
            "FIGURES_ROOT": str(tmp_path / "figures"),
            "METHODS": "baseline",
            "RUN_FAMILIES": "leapfrog",
            "LF_STEPS": "1",
            "RUN_PLOTS": "0",
        }
    )
    subprocess.run(
        ["bash", str(tutorial / "allrun.sh")],
        cwd=tutorial,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((stale_case / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert not (stale_case / "rejected_state.h5").exists()
    assert not (run_root / ".failed").exists()
