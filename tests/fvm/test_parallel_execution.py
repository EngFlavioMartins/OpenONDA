"""Execution-mode safety and replicated-field contract tests."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from source.solvers.fvm import ComputeConfig, FVMSetup
from source.solvers.fvm.core.parallel import ParallelContext, detected_world_size


class _FakeComm:
    def __init__(self, rank=0, size=2):
        self._rank = rank
        self._size = size

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size


class _FakeMPI:
    SUM = "sum"
    MAX = "max"
    LAND = "land"


def test_serial_context_is_default():
    context = ParallelContext.create(ComputeConfig())
    assert context.rank == 0
    assert context.size == 1
    assert context.is_root


def test_serial_context_rejects_accidental_mpi_launch(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert detected_world_size() == 4
    with pytest.raises(RuntimeError, match="launched with 4 MPI ranks"):
        ParallelContext.create(ComputeConfig())


def test_replicated_mode_requires_petsc_backend():
    execution = ComputeConfig(parallel_mode="petsc_replicated", linear_backend="scipy")
    with pytest.raises(ValueError, match="requires linear_backend='petsc'"):
        ParallelContext.create(execution)


def test_petsc_context_records_injected_communicator(monkeypatch):
    fake_package = types.ModuleType("petsc4py")
    fake_package.PETSc = object()
    monkeypatch.setitem(sys.modules, "petsc4py", fake_package)
    context = ParallelContext.create(
        ComputeConfig.petsc_replicated(),
        comm=_FakeComm(rank=1, size=2),
        mpi=_FakeMPI,
    )
    assert context.rank == 1
    assert context.size == 2
    assert not context.is_root
    assert context.root_view(np.ones((4, 3)), trailing_shape=(3,)).shape == (0, 3)


def test_execution_config_round_trip(tmp_path):
    path = tmp_path / "fvm.json"
    config = FVMSetup(
        case_name="parallel",
        execution=ComputeConfig.petsc_replicated(),
    )
    config.save(path)
    loaded = FVMSetup.load(path)
    assert loaded.execution == config.execution
