"""Collective phase ordering, bounded waiting and interruption propagation."""

from contextlib import contextmanager
import time

import pytest


@pytest.fixture
def phase(monkeypatch):
    from source.simulation import parallel

    original_wait = parallel._wait_completion

    @contextmanager
    def production(comm, description, *, sleep=time.sleep):
        with monkeypatch.context() as patch:
            patch.setattr(parallel, "_wait_completion", lambda req: original_wait(req, sleep=sleep))
            with parallel.collective_phase(comm, description):
                yield

    return production


class Request:
    def __init__(self, events, polls=0):
        self.events, self.polls, self.complete = events, polls, False

    def Test(self):
        self.events.append("test")
        if self.polls:
            self.polls -= 1
            return False
        self.complete = True
        return True


class Comm:
    def __init__(self, *, polls=0, summaries=None):
        self.events, self.summaries = [], summaries
        self.request = Request(self.events, polls)

    def Get_size(self):
        return 4

    def Ibarrier(self):
        self.events.append("ibarrier")
        return self.request

    def allgather(self, summary):
        assert self.request.complete
        self.events.append(("allgather", summary))
        return [summary] * 4 if self.summaries is None else self.summaries


def test_immediate_completion_has_no_sleep_and_keeps_collective_order(phase):
    comm = Comm()
    with phase(comm, "work", sleep=lambda _: pytest.fail("slept")):
        comm.events.append("body")
    assert comm.events == ["body", "ibarrier", "test", ("allgather", None)]


def test_bounded_polling_then_capped_adaptive_sleep(phase):
    comm, sleeps = Comm(polls=20), []
    with phase(comm, "work", sleep=sleeps.append):
        pass
    assert len(sleeps) == 13
    assert sleeps[:6] == [5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1e-3]
    assert sleeps[6:] == [1e-3] * 7


@pytest.mark.parametrize("error", [ValueError("bad"), KeyboardInterrupt(), SystemExit(17)])
def test_original_local_base_exception_object_is_preserved(error, phase):
    comm = Comm(polls=9)
    with (
        pytest.raises(type(error)) as raised,
        phase(comm, "work", sleep=lambda _: None),
    ):
        raise error
    assert raised.value is error
    assert comm.events[-1] == ("allgather", f"{type(error).__name__}: {error}")


def test_remote_failure_uses_first_failing_rank(phase):
    comm = Comm(summaries=[None, "ValueError: first", None, "SystemExit: 4"])
    with (
        pytest.raises(RuntimeError, match="work failed on rank 1: ValueError: first"),
        phase(comm, "work"),
    ):
        pass


def test_wait_interrupt_finishes_request_and_is_then_propagated(phase):
    comm = Comm(polls=11)
    error = KeyboardInterrupt("during sleep")
    calls = []

    def sleep(delay):
        calls.append(delay)
        if len(calls) == 1:
            raise error

    with (
        pytest.raises(KeyboardInterrupt) as raised,
        phase(comm, "work", sleep=sleep),
    ):
        pass
    assert raised.value is error and comm.request.complete
    assert len(calls) == 4
    assert comm.events[-1] == ("allgather", "KeyboardInterrupt: during sleep")


def test_body_failure_takes_precedence_over_wait_interrupt(phase):
    comm, primary = Comm(polls=9), ValueError("body")

    def sleep(_):
        raise KeyboardInterrupt("later")

    with (
        pytest.raises(ValueError) as raised,
        phase(comm, "work", sleep=sleep),
    ):
        raise primary
    assert raised.value is primary


@pytest.mark.parametrize("comm", [None, type("Serial", (), {"Get_size": lambda self: 1})()])
def test_serial_path_never_requests_collectives_or_sleeps(comm, phase):
    with phase(comm, "work", sleep=lambda _: pytest.fail("slept")):
        pass


def test_mpi_request_error_is_not_swallowed_or_followed_by_mismatched_allgather(phase):
    comm = Comm()

    def broken():
        raise RuntimeError("broken MPI request")

    comm.request.Test = broken
    with (
        pytest.raises(RuntimeError, match="broken MPI request"),
        phase(comm, "work"),
    ):
        pass
    assert comm.events == ["ibarrier"]
