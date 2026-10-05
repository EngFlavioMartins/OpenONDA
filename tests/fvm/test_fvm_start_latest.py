"""Fresh, resumed, and completed FVM runs share a single entry point."""

from dataclasses import replace
import json

import numpy as np

from openonda import fvm
from tests.fvm.test_restart_and_diagnostics import _setup
from tests.support.fvm_mesh import structured_box


def _solver(directory, *, end_time=0.03):
    setup = _setup()
    setup.time = replace(
        setup.time, end_time=end_time, output_schedule=fvm.RunSchedule(every_n_steps=1)
    )
    setup.backup = fvm.BackupConfig(schedule=fvm.RunSchedule(every_n_steps=2), write_at_end=True)
    return fvm.FVMSolver(setup, str(directory), mesh_data=structured_box(2, 2, 2))


def test_latest_restores_bdf_history_and_aligns_unsaved_diagnostics(tmp_path):
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


def test_initial_replaces_old_native_history_and_latest_finds_new_run(tmp_path):
    directory = tmp_path / "fresh_again"
    previous = _solver(directory)
    previous.run(start_from="latest")
    assert previous.step == 3

    fresh = _solver(directory, end_time=0.01)
    fresh.run(start_from="initial")
    assert fresh.step == 1

    times = [
        json.loads(line)["time"]
        for line in (directory / "solution/diagnostics.jsonl").read_text().splitlines()
    ]
    assert times == [0.01]
    resumed = _solver(directory, end_time=0.01)
    assert resumed.start_from("latest")
    assert resumed.step == 1
    resumed.close()


def test_initial_ignores_corrupt_old_backup(tmp_path):
    directory = tmp_path / "corrupt_old"
    backup = directory / "solution/backup"
    backup.parent.mkdir(parents=True)
    backup.write_text("not a native backup")
    solver = _solver(directory, end_time=0.01)
    solver.run(start_from="initial")
    assert solver.step == 1
    assert list((directory / "solution/restart_history").glob("initial-before-*/backup/backup"))


def test_latest_without_backup_replaces_prior_output_history(tmp_path):
    previous = _solver(tmp_path)
    previous.run(start_from="latest")
    (tmp_path / "solution/backup").unlink()
    fresh = _solver(tmp_path, end_time=0.01)
    fresh.run(start_from="latest")
    assert fresh.step == 1
    history = tmp_path / "solution/diagnostics.jsonl"
    assert [json.loads(row)["time"] for row in history.read_text().splitlines()] == [0.01]
    assert list((tmp_path / "solution/restart_history").glob("initial-before-*"))


def test_none_start_preserves_in_memory_state(tmp_path):
    solver = _solver(tmp_path)
    solver.advance()
    expected = solver.velocity.copy()
    assert not solver.start_from(None)
    assert solver.step == 1
    np.testing.assert_array_equal(solver.velocity, expected)
    solver.close()
