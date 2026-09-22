"""The coupled factory owns rank decisions and the solvers it constructs."""

from types import SimpleNamespace

import pytest

from source.coupler import CouplerSetup, create_coupler
from source.solvers.fvm import FVMSetup
import source.solvers.fvm.factory as fvm_factory
import source.solvers.vpm as vpm_api
from source.solvers.vpm import Numerics, VPMCase


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("fails", [False, True])
def test_factory_constructs_vpm_only_on_owner_and_closes_resources(
    tmp_path, monkeypatch, rank, fails
):
    events = []

    class FVM:
        case_dir = tmp_path
        parallel = SimpleNamespace(
            rank=rank, is_root=rank == 0, comm=None, bcast=lambda value: value
        )

        def __enter__(self):
            return self

        def __exit__(self, _type, error, _tb):
            events.append(("fvm_closed", error is not None))

    def build_vpm(case):
        assert rank == 0, "A worker must never construct a GPU solver"
        assert case.directory == tmp_path
        events.append("vpm_created")
        return SimpleNamespace(close=lambda: events.append("vpm_closed"))

    monkeypatch.setattr(fvm_factory, "create_fvm_solver", lambda *args, **kwargs: FVM())
    monkeypatch.setattr(vpm_api, "VPMSolver", build_vpm)
    case = VPMCase(numerics=Numerics(compute_device="CPU"), directory=tmp_path)
    driver = create_coupler(FVMSetup(case_name="factory"), case, CouplerSetup(), mesh={})
    try:
        with driver:
            assert (driver._injected_vpm is not None) == (rank == 0)
            if fails:
                raise ValueError("deliberate run failure")
    except ValueError as error:
        assert fails and str(error) == "deliberate run failure"
    driver.close()
    expected = ["vpm_created", "vpm_closed"] if rank == 0 else []
    assert events == [*expected, ("fvm_closed", fails)]
    if driver._log_handler is not None:
        assert driver._log_handler.stream is None


@pytest.mark.parametrize("rank", [0, 1])
def test_factory_propagates_owner_construction_failure(tmp_path, monkeypatch, rank):
    events = []

    class FVM:
        parallel = SimpleNamespace(
            is_root=rank == 0,
            comm=None,
            bcast=lambda value: "ValueError: bad particle configuration",
        )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("closed")

    def fail(case):
        assert rank == 0
        raise ValueError("bad particle configuration")

    monkeypatch.setattr(fvm_factory, "create_fvm_solver", lambda *args, **kwargs: FVM())
    monkeypatch.setattr(vpm_api, "VPMSolver", fail)
    case = VPMCase(numerics=Numerics(compute_device="CPU"), directory=tmp_path)
    with pytest.raises((ValueError, RuntimeError), match="bad particle configuration"):
        create_coupler(FVMSetup(case_name="factory"), case, CouplerSetup())
    assert events == ["closed"]


def test_factory_does_not_close_supplied_solvers(tmp_path):
    def unexpected_close():
        pytest.fail("externally supplied solver closed")

    fvm = SimpleNamespace(case_dir=tmp_path, close=unexpected_close)
    vpm = SimpleNamespace(close=unexpected_close)
    with create_coupler(fvm, vpm, CouplerSetup()) as driver:
        assert driver._injected_fvm is fvm
        assert driver._injected_vpm is vpm


def test_driver_uses_the_supplied_fvm_communicator(tmp_path, monkeypatch):
    # Communicator-local ownership must not be replaced by a global MPI rank.
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    comm = SimpleNamespace(Get_rank=lambda: 1, Get_size=lambda: 2)
    fvm = SimpleNamespace(case_dir=tmp_path, parallel=SimpleNamespace(rank=1, comm=comm))
    with create_coupler(fvm, None, CouplerSetup()) as driver:
        assert driver._comm is comm
        assert driver._mpi_rank == 1
        assert not driver._is_master


def test_worker_receives_local_phase_failure():
    from source.coupler.parallel import collective_phase

    comm = SimpleNamespace(
        Get_size=lambda: 2,
        allgather=lambda value: ["HealthError: strain limit exceeded", value],
    )
    with (
        pytest.raises(RuntimeError, match="rank 0.*strain limit exceeded"),
        collective_phase(comm, "VPM output"),
    ):
        pass
