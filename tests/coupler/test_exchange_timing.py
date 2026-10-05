"""Accepted exchange accounting closes across output and native backups."""

import json
import logging
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from source.coupler import reporting


@pytest.mark.parametrize("backup_due", [False, True])
def test_exchange_total_includes_state_checks_reporting_and_backup_once(
    tmp_path, monkeypatch, backup_due
):
    clock = SimpleNamespace(value=20.0)
    monkeypatch.setattr(reporting.time, "perf_counter", lambda: clock.value)

    def diagnostics(*args):
        clock.value += 2.0
        return {}

    monkeypatch.setattr(reporting, "compute_diagnostics", diagnostics)
    monkeypatch.setattr(reporting, "finish_coupling_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(reporting, "flush_log", lambda *args: None)
    backups = []

    def backup(path, *, coupling_step):
        backups.append(coupling_step)
        # This complete checkpoint must precede the end-to-end timing record.
        assert not (tmp_path / "coupler_diagnostics.jsonl").exists()
        clock.value += 7.0

    coupler = SimpleNamespace(
        _is_master=True,
        _log_stop_step=10,
        _step_transfer_stats={"donor_gather_seconds": 0.5},
        fvm_solver=SimpleNamespace(step=5, time=0.04, max_courant_number=0.5),
        n_fvm_substeps=5,
        vpm_time_step_size=0.04,
        setup=SimpleNamespace(backup_interval_steps=1 if backup_due else 0),
        solution_dir=tmp_path,
        coupling_diagnostics=[],
        save_backup=backup,
    )
    reporting.record_step(
        coupler,
        1,
        0.04,
        (6.0, 2.0, 3.0, 1.0),
        None,
        logger=logging.getLogger("test.exchange_timing"),
        exchange_started=0.0,
        state_checks_and_sampling_seconds=5.0,
    )
    row = json.loads((tmp_path / "coupler_diagnostics.jsonl").read_text())
    times = row["timing_seconds"]
    assert times["total"] == 22.0 + (7.0 if backup_due else 0.0)
    assert times["evolution_total"] == 12.0
    assert times["state_checks_and_samplers"] == 5.0
    assert times["reporting"] == 2.0
    assert times["backup"] == (7.0 if backup_due else 0.0)
    assert times["coupling_control_and_wait"] == 3.0
    assert times["last_sweep_donor_gather"] == 0.5  # not added a second time
    assert times["total"] == sum(
        times[name]
        for name in (
            "vpm",
            "vpm_boundary_condition",
            "fvm",
            "transfer",
            "state_checks_and_samplers",
            "reporting",
            "backup",
            "coupling_control_and_wait",
        )
    )
    assert backups == ([1] if backup_due else [])
    assert "before timing output" in row["timing_scope"]


def _failure_coupler(directory, backup, *, master=True):
    return SimpleNamespace(
        _is_master=master,
        _log_stop_step=10,
        _step_transfer_stats={},
        fvm_solver=SimpleNamespace(step=5, time=0.04, max_courant_number=0.5),
        n_fvm_substeps=5,
        vpm_time_step_size=0.04,
        setup=SimpleNamespace(backup_interval_steps=1),
        solution_dir=directory,
        coupling_diagnostics=[],
        save_backup=backup,
    )


def _record(coupler, *, comm=None):
    reporting.record_step(
        coupler,
        1,
        0.04,
        (0.01, 0.01, 0.01, 0.01),
        None,
        logger=logging.getLogger("test.exchange_timing"),
        comm=comm,
    )


def _patch_reporting(monkeypatch):
    monkeypatch.setattr(reporting, "compute_diagnostics", lambda *args: {"accepted": True})
    monkeypatch.setattr(reporting, "finish_coupling_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(reporting, "flush_log", lambda *args: None)


def test_failed_backup_preserves_one_accepted_row_and_original_error(tmp_path, monkeypatch):
    _patch_reporting(monkeypatch)
    failure = OSError("checkpoint device is full")
    checkpoint_info = tmp_path / "last-committed-checkpoint_info.json"
    checkpoint_info.write_text('{"step":0}\n')

    def fail_backup(*args, **kwargs):
        raise failure

    coupler = _failure_coupler(tmp_path, fail_backup)
    with pytest.raises(OSError, match="checkpoint device is full") as caught:
        _record(coupler)
    assert caught.value is failure
    lines = (tmp_path / "coupler_diagnostics.jsonl").read_text().splitlines()
    assert len(lines) == len(coupler.coupling_diagnostics) == 1
    row = json.loads(lines[0])
    assert row["accepted"] and row["step"] == 1 and row["time"] == 0.04
    assert row["backup_phase"] == {
        "scheduled": True,
        "status": "failed",
        "error": "OSError: checkpoint device is full",
    }
    assert row["timing_seconds"]["backup"] >= 0
    assert checkpoint_info.read_text() == '{"step":0}\n'


def test_diagnostic_write_failure_does_not_mask_original_backup_failure(tmp_path, monkeypatch):
    _patch_reporting(monkeypatch)
    failure = OSError("primary checkpoint failure")

    def fail_backup(*args, **kwargs):
        raise failure

    original_open = Path.open

    def fail_diagnostics(path, *args, **kwargs):
        if path.name == "coupler_diagnostics.jsonl":
            raise PermissionError("diagnostic filesystem unavailable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_diagnostics)
    coupler = _failure_coupler(tmp_path, fail_backup)
    with pytest.raises(OSError, match="primary checkpoint failure") as caught:
        _record(coupler)
    assert caught.value is failure
    assert len(coupler.coupling_diagnostics) == 1
    assert any("PermissionError" in note for note in failure.__notes__)


def test_backup_failure_keeps_all_ranks_on_the_same_output_collectives(tmp_path, monkeypatch):
    """Bounded threaded communicators catch a skipped collective as a timeout."""
    _patch_reporting(monkeypatch)
    barrier = threading.Barrier(2, timeout=3)
    summaries = {}
    communicators = []
    results = [None, None]
    failure = OSError("rank zero checkpoint write failed")

    class Comm:
        def __init__(self, rank):
            self.rank, self.calls = rank, 0

        def Get_size(self):
            return 2

        def Ibarrier(self):
            return SimpleNamespace(Test=lambda: True)

        def allgather(self, value):
            turn = self.calls
            summaries[(turn, self.rank)] = value
            barrier.wait()
            result = [summaries[(turn, rank)] for rank in range(2)]
            barrier.wait()
            self.calls += 1
            return result

    def worker(rank):
        def backup(*args, **kwargs):
            if rank == 0:
                raise failure

        comm = Comm(rank)
        communicators.append(comm)
        coupler = _failure_coupler(tmp_path, backup, master=rank == 0)
        try:
            _record(coupler, comm=comm)
        except BaseException as error:
            results[rank] = error

    workers = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    assert not any(worker.is_alive() for worker in workers)
    assert results[0] is failure
    assert isinstance(results[1], RuntimeError)
    assert "rank 0" in str(results[1]) and "checkpoint write failed" in str(results[1])
    assert [comm.calls for comm in communicators] == [3, 3]
    assert len((tmp_path / "coupler_diagnostics.jsonl").read_text().splitlines()) == 1
