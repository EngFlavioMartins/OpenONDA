"""Collective checkpoint clock validation without advancing a physical solver."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.parametrize("ranks", [2, 4])
def test_backup_clock_validation_and_output_are_collective(tmp_path, ranks):
    if find_spec("mpi4py") is None:
        pytest.skip("mpi4py is required")
    launcher = Path(sys.executable).with_name("mpiexec")
    if not launcher.is_file():
        found = shutil.which("mpiexec")
        if found is None:
            pytest.skip("mpiexec is required")
        launcher = Path(found)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    result = subprocess.run(
        [
            str(launcher),
            "-n",
            str(ranks),
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout[-10000:] + result.stderr[-10000:]
    reports = [
        line.removeprefix("SYNCHRONIZED_BACKUP_MPI ")
        for line in result.stdout.splitlines()
        if line.startswith("SYNCHRONIZED_BACKUP_MPI ")
    ]
    assert len(reports) == 1, result.stdout
    assert json.loads(reports[0]) == {
        "ranks": ranks,
        "rejected_before_writes": ["nonroot_fvm_time", "nonroot_fvm_step", "root_vpm_time"],
        "rejected_before_flush": ["nonroot_fvm_candidate", "nonroot_fvm_failed"],
        "complete_output_write": True,
    }


def _mpi_worker(root):
    from types import SimpleNamespace

    import h5py
    from mpi4py import MPI
    import numpy as np

    from source.coupler.backup import checkpoint_path_hash, save_coupled_backup
    from source.coupler.config.types import CouplerSetup
    from source.solvers.fvm.config.types import FVMSetup, TimeConfig
    from source.solvers.fvm.core.solver import FVMSolver
    from source.solvers.vpm.config.case import Numerics

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    class FakeFVM:
        def __init__(self):
            self.step = 10
            self.time = 0.2
            self.time_step_size = 0.02
            self._accepted_time_step_size = 0.02
            self._previous_time_step_size = 0.02
            self._n_committed_time_steps = 10
            self._step_phase = "accepted"
            self._evolution_failure = None
            self.setup = FVMSetup(
                case_name="fake_backup_clock", time=TimeConfig(time_step_size=0.02, end_time=1.0)
            )
            self._resolved_setup = self.setup
            self._time_config = self.setup.time
            self.parallel = SimpleNamespace(
                comm=comm,
                is_partitioned=True,
                is_root=rank == 0,
                rank=rank,
                size=size,
            )
            self.writes = 0
            self.flushes = 0

        def _ensure_evolution_usable(self):
            FVMSolver._ensure_evolution_usable(self)

        def flush_output(self):
            self.flushes += 1
            raise AssertionError("Rejected FVM state reached output flushing")

        def save_state(self, directory):
            self.writes += 1
            directory = Path(directory)
            if rank == 0:
                directory.mkdir(parents=True)
            comm.Barrier()
            with (directory / f"rank-{rank:05d}.npz").open("wb") as stream:
                np.savez(stream, time=np.asarray(self.time), step=np.asarray(self.step))
            comm.Barrier()
            if rank == 0:
                (directory / "checkpoint_info.json").write_text(
                    json.dumps(
                        {
                            "format_version": 8,
                            "n_ranks": size,
                            "files": [f"rank-{index:05d}.npz" for index in range(size)],
                        }
                    )
                )
            comm.Barrier()

    class FakeVPM:
        def __init__(self):
            self.step = 5
            self.time = 0.2
            self.time_step_size = 0.04
            self.setup = Numerics(time_step_size=0.04, compute_device="CPU", verbose=False)
            self.writes = 0

        def _save_backup_to(self, filename):
            self.writes += 1
            with h5py.File(filename + ".h5", "w") as archive:
                solver = archive.create_group("solver")
                solver.attrs["step"] = self.step
                solver.attrs["time"] = self.time
            Path(filename + ".vtu").write_text("fake complete visualization checkpoint_file\n")

    def make_coupler():
        return SimpleNamespace(
            _comm=comm,
            _is_master=rank == 0,
            fvm_solver=FakeFVM(),
            vpm_solver=FakeVPM() if rank == 0 else None,
            n_fvm_substeps=2,
            fvm_time_step_size=0.02,
            vpm_time_step_size=0.04,
            setup=CouplerSetup(),
            vorticity_transfer=None,
            _velocity_boundary_condition_old=np.ones((3, 3)),
            _normal_velocity_boundary_condition_old=np.ones(3),
            _tangential_gradient_boundary_condition_old=np.zeros((3, 3)),
        )

    rejected = []
    for name in ("nonroot_fvm_time", "nonroot_fvm_step", "root_vpm_time"):
        coupler = make_coupler()
        if name == "nonroot_fvm_time" and rank == size - 1:
            coupler.fvm_solver.time = 0.201
        elif name == "nonroot_fvm_step" and rank == size - 1:
            coupler.fvm_solver.step = 11
        elif name == "root_vpm_time" and rank == 0:
            coupler.vpm_solver.time = 0.201
        error = None
        target = root / name
        try:
            save_coupled_backup(coupler, target, coupling_step=5)
        except Exception as failure:
            error = str(failure)
        errors = comm.allgather(error)
        assert all(errors), (name, errors)
        writes = comm.allgather(
            (
                coupler.fvm_solver.writes,
                0 if coupler.vpm_solver is None else coupler.vpm_solver.writes,
            )
        )
        assert writes == [(0, 0)] * size, (name, writes)
        assert not target.exists() or not any(target.rglob("*")), name
        rejected.append(name)

    rejected_before_flush = []
    for name in ("nonroot_fvm_candidate", "nonroot_fvm_failed"):
        fvm = FakeFVM()
        if rank == size - 1:
            if name == "nonroot_fvm_candidate":
                fvm._step_phase = "candidate"
            else:
                fvm._evolution_failure = RuntimeError("injected physical step failure")
        target = root / name
        error = None
        try:
            FVMSolver.save_state(fvm, target)
        except Exception as failure:
            error = str(failure)
        errors = comm.allgather(error)
        assert all(errors), (name, errors)
        expected = (
            "uncommitted FVM candidate" if name.endswith("candidate") else "terminally invalid"
        )
        assert all(expected in message for message in errors), (name, errors)
        assert comm.allgather((fvm.flushes, fvm.writes)) == [(0, 0)] * size
        assert not target.exists(), name
        rejected_before_flush.append(name)

    coupler = make_coupler()
    target = root / "complete"
    returned = save_coupled_backup(coupler, target, coupling_step=5)
    assert Path(returned) == target
    comm.Barrier()
    checkpoint_info = json.loads((target / "checkpoint_info.json").read_text())
    assert {
        key: checkpoint_info[key]
        for key in ("coupling_step", "fvm_step", "vpm_step", "n_fvm_substeps", "time")
    } == {
        "coupling_step": 5,
        "fvm_step": 10,
        "vpm_step": 5,
        "n_fvm_substeps": 2,
        "time": 0.2,
    }
    assert set(checkpoint_info["checkpoint_files"]) == {
        "fvm",
        "vpm",
        "vpm_vtu",
        "vpm_boundary_condition",
    }
    assert set(checkpoint_info["file_sha256"]) == set(checkpoint_info["checkpoint_files"])
    for name, relative in checkpoint_info["checkpoint_files"].items():
        checkpoint_file = target / relative
        assert checkpoint_file.exists()
        assert checkpoint_path_hash(checkpoint_file) == checkpoint_info["file_sha256"][name]
    fvm_directory = target / checkpoint_info["checkpoint_files"]["fvm"]
    inner = json.loads((fvm_directory / "checkpoint_info.json").read_text())
    assert inner["n_ranks"] == size and len(inner["files"]) == size
    for filename in inner["files"]:
        with np.load(fvm_directory / filename, allow_pickle=False) as state:
            assert float(state["time"]) == checkpoint_info["time"]
            assert int(state["step"]) == checkpoint_info["fvm_step"]
    with h5py.File(target / checkpoint_info["checkpoint_files"]["vpm"], "r") as archive:
        assert archive["solver"].attrs["time"] == checkpoint_info["time"]
        assert archive["solver"].attrs["step"] == checkpoint_info["vpm_step"]
    writes = comm.allgather(
        (coupler.fvm_solver.writes, 0 if coupler.vpm_solver is None else coupler.vpm_solver.writes)
    )
    assert writes == [(1, 1)] + [(1, 0)] * (size - 1)
    if rank == 0:
        print(
            "SYNCHRONIZED_BACKUP_MPI "
            + json.dumps(
                {
                    "ranks": size,
                    "rejected_before_writes": rejected,
                    "rejected_before_flush": rejected_before_flush,
                    "complete_output_write": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--worker"
    _mpi_worker(Path(sys.argv[2]).resolve())
