"""Coupled runs continue from their latest numbered backup, including coupling step zero."""

import json

import numpy as np

from openonda import coupler
from tests.support.coupler import FREESTREAM_VELOCITY, uniform_stream_cases


def _solver(directory):
    flow, particles = uniform_stream_cases(
        directory, fvm_time_step_size=0.01, end_time=0.08
    )
    config = coupler.CouplerConfig(
        transfer_inset_spacings=0.0,
        blend_end_depth=0.0,
        freestream_velocity=FREESTREAM_VELOCITY,
        backup=coupler.Backup(schedule=coupler.EverySteps(2)),
    )
    return coupler.CoupledSolver(flow, particles, config)


def _history(directory):
    history = directory / "solution/coupler_diagnostics.jsonl"
    return [json.loads(row)["step"] for row in history.read_text().splitlines()]


def test_latest_continues_an_interrupted_run_and_stops_at_the_end(tmp_path):
    # Interrupt after three shared FVM/VPM steps; the last backup is step two.
    with _solver(tmp_path) as first:
        first.start_from("latest")
        for _ in range(3):
            first.advance()
    metadata = json.loads((tmp_path / "solution/coupler_metadata.json").read_text())
    assert metadata["run_status"] == {"status": "partial"}
    assert _history(tmp_path) == [1, 2, 3]

    with _solver(tmp_path) as resumed:
        resumed.run(start_from="latest")
        expected = resumed.fvm_solver.velocity.copy()
    assert _history(tmp_path) == list(range(1, 9))
    assert (tmp_path / "solution/coupled/coupled_000008/checkpoint_info.json").is_file()

    # A completed run continues nowhere and writes nothing new.
    before = (tmp_path / "solution/coupler_diagnostics.jsonl").read_bytes()
    with _solver(tmp_path) as completed:
        completed.run(start_from="latest")
        np.testing.assert_allclose(completed.fvm_solver.velocity, expected, atol=1e-13, rtol=0)
    assert (tmp_path / "solution/coupler_diagnostics.jsonl").read_bytes() == before

    with _solver(tmp_path / "reference") as reference:
        reference.run(start_from="initial")
        np.testing.assert_allclose(reference.fvm_solver.velocity, expected, atol=1e-12, rtol=0)


def test_backup_at_coupling_step_zero_does_not_repeat_the_initial_transfer(tmp_path):
    with _solver(tmp_path) as first:
        first.save_backup()
    assert (tmp_path / "solution/coupled/coupled_000000/checkpoint_info.json").is_file()
    with _solver(tmp_path) as resumed:
        assert resumed.start_from("latest")
        resumed.advance()
        # One transfer at the initial time, then one per coupling step.
        assert resumed.vorticity_transfer.step == 2


def test_initial_replaces_the_previous_run(tmp_path):
    with _solver(tmp_path) as previous:
        previous.start_from("latest")
        for _ in range(2):
            previous.advance()
    stale_frame = tmp_path / "solution/vpm/vpm_999999.h5"
    stale_frame.write_bytes(b"frame of an earlier run")

    with _solver(tmp_path) as fresh:
        assert not fresh.start_from("initial")
        fresh.advance()
    assert not stale_frame.exists()
    assert not (tmp_path / "solution/coupled").exists()
    assert _history(tmp_path) == [1]

    with _solver(tmp_path) as resumed:
        assert not resumed.start_from("latest")
        resumed.advance()
    assert _history(tmp_path) == [1]
