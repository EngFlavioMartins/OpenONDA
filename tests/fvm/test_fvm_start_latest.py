"""Fresh, resumed, and completed FVM runs share a single entry point."""

from dataclasses import replace
import json

import numpy as np

from openonda import fvm
from source.solvers.fvm.mesh.cartesian import structured_box
from tests.fvm.test_restart_and_diagnostics import _setup


def _solver(directory):
    setup = _setup()
    setup.time = replace(
        setup.time, end_time=0.03, output_schedule=fvm.RunSchedule(every_n_steps=1)
    )
    setup.backup = fvm.BackupConfig(schedule=fvm.RunSchedule(every_n_steps=2), write_at_end=True)
    return fvm.FVMSolver(setup, str(directory), mesh_data=structured_box(2, 2, 2))


def test_latest_restores_bdf_history_and_reconciles_unsaved_diagnostics(tmp_path):
    reference = _solver(tmp_path / "reference")
    reference.run(start_from="latest")
    expected = reference.velocity.copy()
    directory = tmp_path / "continued"
    interrupted = _solver(directory)
    interrupted.advance()
    interrupted.save_state(directory / "solution/backup")
    interrupted.advance()
    interrupted.close()
    logs = (directory / "solution/fvm.log").read_text()

    resumed = _solver(directory)
    resumed.run(start_from="latest")
    np.testing.assert_allclose(resumed.velocity, expected, rtol=0, atol=1e-13)
    assert resumed.step == 3
    history = directory / "solution/diagnostics.jsonl"
    times = [json.loads(line)["time"] for line in history.read_text().splitlines()]
    assert times == [0.01, 0.02, 0.03]
    assert (directory / "solution/fvm.log").read_text().startswith(logs)
    before = history.read_bytes()
    completed = _solver(directory)
    completed.run(start_from="latest")
    assert completed.step == 3
    assert history.read_bytes() == before
