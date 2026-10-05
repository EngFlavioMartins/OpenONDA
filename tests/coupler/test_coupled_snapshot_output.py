"""Coupled checkpoints retain matching FVM/VPM visualization times."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py
import pytest

from source.coupler import solver
from source.coupler.parallel import collective_phase
from source.solvers.fvm.io.vtk_exporter import PVDManager


@pytest.mark.parametrize("buffered", [False, True])
def test_checkpoint_times_match_both_collections_without_duplicate_fvm_writes(
    tmp_path, monkeypatch, buffered
):
    output = tmp_path / "solution"
    output.mkdir()
    fvm_frames = output / "fvm"
    fvm_frames.mkdir()
    pvd = PVDManager(str(output / "fvm.pvd"))
    fvm = SimpleNamespace(step=0, time=0.0, _state_revision=0)
    pending = []
    written = []

    def publish_frame(step, time):
        frame = fvm_frames / f"fvm_{step:06d}.vtu"
        frame.write_text("<VTKFile/>\n")
        pvd.add_step(time, str(frame))

    def write_vtk():
        written.append(fvm.step)
        state = (fvm.step, fvm.time, fvm._state_revision)
        if buffered:
            pending.append(state[:2])
        else:
            publish_frame(*state[:2])
        fvm._last_vtk_state = state

    def flush_output():
        while pending:
            publish_frame(*pending.pop(0))

    fvm.write_vtk = write_vtk
    fvm.flush_output = flush_output
    coupler = SimpleNamespace(fvm_solver=fvm, solution_dir=output, _comm=None, _is_master=True)

    def checkpoint(case, directory, *, coupling_step):
        backup = Path(directory)
        generation = backup / f"checkpoint-{coupling_step:06d}"
        generation.mkdir(parents=True, exist_ok=True)
        basename = f"vpm_{coupling_step:06d}"
        with h5py.File(generation / f"{basename}.h5", "w") as archive:
            group = archive.create_group("solver")
            group.attrs["time"] = case.fvm_solver.time
        (generation / f"{basename}.vtu").write_text("<VTKFile/>\n")
        (backup / "checkpoint_info.json").write_text(
            json.dumps(
                {
                    "checkpoint_files": {
                        "vpm": f"{generation.name}/{basename}.h5",
                        "vpm_vtu": f"{generation.name}/{basename}.vtu",
                    }
                }
            )
        )
        return backup

    monkeypatch.setattr(solver, "save_coupled_backup", checkpoint)
    original_publish = solver.publish_vpm_snapshot

    def publish_vpm(backup, directory):
        # Buffered FVM output must be visible before VPM output succeeds.
        assert not pending
        assert fvm.time in [
            float(row.attrib["timestep"])
            for row in ET.parse(output / "fvm.pvd").findall(".//DataSet")
        ]
        return original_publish(backup, directory)

    monkeypatch.setattr(solver, "publish_vpm_snapshot", publish_vpm)
    for exchange_step in (0, 25, 100, 101):
        fvm.step, fvm.time, fvm._state_revision = (
            5 * exchange_step,
            0.04 * exchange_step,
            exchange_step,
        )
        if exchange_step == 100:
            # The ordinary four-second FVM schedule already emits this frame.
            fvm.write_vtk()
        assert (
            solver.FVMVPMCoupler.save_backup(
                coupler, output / "backups", coupling_step=exchange_step
            )
            == output / "backups"
        )
    # Saving an unchanged endpoint reuses its retained FVM frame.
    solver.FVMVPMCoupler.save_backup(coupler, output / "backups", coupling_step=101)

    collections = [
        ET.parse(output / f"{component}.pvd").findall(".//DataSet") for component in ("fvm", "vpm")
    ]
    assert [[float(row.attrib["timestep"]) for row in rows] for rows in collections] == [
        [0.0, 1.0, 4.0, 4.04],
        [0.0, 1.0, 4.0, 4.04],
    ]
    for rows in collections:
        assert all((output / row.attrib["file"]).is_file() for row in rows)
    assert written == [0, 125, 500, 505]


def _parallel_output_write(
    tmp_path, monkeypatch, *, root_written=False, worker_written=False, failure_phase=None
):
    barrier = threading.Barrier(2, timeout=3)
    values = {}
    results = [None, None]
    communicators = [None, None]
    writes, snapshot_writes = [], []
    original_failure = OSError(f"{failure_phase} write failed")

    class Comm:
        def __init__(self, rank):
            self.rank, self.calls = rank, 0

        def Get_size(self):
            return 2

        def Ibarrier(self):
            return SimpleNamespace(Test=lambda: True)

        def allgather(self, value):
            turn = self.calls
            values[(turn, self.rank)] = value
            barrier.wait()
            result = [values[(turn, rank)] for rank in range(2)]
            barrier.wait()
            self.calls += 1
            return result

        def bcast(self, value, root):
            return self.allgather(value)[root]

    def checkpoint(case, directory, **kwargs):
        if failure_phase == "backup" and case._is_master:
            raise original_failure
        return Path(directory)

    def publish(*args):
        snapshot_writes.append(0)
        if failure_phase == "publish":
            raise original_failure

    monkeypatch.setattr(solver, "save_coupled_backup", checkpoint)
    monkeypatch.setattr(solver, "publish_vpm_snapshot", publish)

    def worker(rank):
        comm = communicators[rank] = Comm(rank)
        fvm = SimpleNamespace(step=125, time=1.0, _state_revision=125)
        if root_written if rank == 0 else worker_written:
            fvm._last_vtk_state = (125, 1.0, 125)

        def write_vtk():
            # Like the real FVM writer, every rank enters an error envelope.
            with collective_phase(comm, "FVM visualization output"):
                writes.append(rank)
                if failure_phase == "vtk" and rank == 1:
                    raise original_failure
            fvm._last_vtk_state = (125, 1.0, 125)

        def flush_output():
            if failure_phase == "flush" and rank == 1:
                raise original_failure

        fvm.write_vtk, fvm.flush_output = write_vtk, flush_output
        case = SimpleNamespace(
            fvm_solver=fvm, solution_dir=tmp_path, _comm=comm, _is_master=rank == 0
        )
        try:
            solver.FVMVPMCoupler.save_backup(case, tmp_path / "backups", coupling_step=25)
        except BaseException as error:
            results[rank] = error

    workers = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in (0, 1)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join(timeout=5)
    assert not any(worker_thread.is_alive() for worker_thread in workers)
    assert communicators[0].calls == communicators[1].calls
    return results, writes, snapshot_writes, original_failure


@pytest.mark.parametrize("root_written,worker_written", [(False, True), (True, False)])
def test_fvm_duplicate_decision_is_shared_by_all_ranks(
    tmp_path, monkeypatch, root_written, worker_written
):
    results, writes, snapshot_writes, _ = _parallel_output_write(
        tmp_path, monkeypatch, root_written=root_written, worker_written=worker_written
    )
    assert results == [None, None]
    assert sorted(writes) == ([] if root_written else [0, 1])
    assert snapshot_writes == [0]


@pytest.mark.parametrize("phase", ["backup", "vtk", "flush", "publish"])
def test_snapshot_failure_propagates_before_the_next_collective_stage(tmp_path, monkeypatch, phase):
    results, writes, snapshot_writes, failure = _parallel_output_write(
        tmp_path, monkeypatch, failure_phase=phase
    )
    failing_rank = 0 if phase in ("backup", "publish") else 1
    assert results[failing_rank] is failure
    assert isinstance(results[1 - failing_rank], RuntimeError)
    assert f"rank {failing_rank}" in str(results[1 - failing_rank])
    assert "write failed" in str(results[1 - failing_rank])
    assert sorted(writes) == ([] if phase == "backup" else [0, 1])
    assert snapshot_writes == ([0] if phase == "publish" else [])
