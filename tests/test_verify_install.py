"""The installed-package check that the installer runs passes outside the checkout."""

import json
import os
import subprocess
import sys


def test_installation_check_solves_and_restarts_every_solver(tmp_path):
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "-I", "-m", "openonda.verify_install"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    report = json.loads(completed.stdout[completed.stdout.index("{") :])
    assert report["native_fvm"]["max_continuity_error"] < 1e-6
    assert report["native_vpm"]["restart_step"] == 2
    assert (report["native_coupled"]["fvm_step"], report["native_coupled"]["vpm_step"]) == (4, 2)
