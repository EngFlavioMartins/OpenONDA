"""Request.Test interruption hardening; fake MPI, no numerical solver imports.

The prototype and production path share the same focused qualifications.
It does not claim recovery from a failed communicator or a killed rank. The
MPI standard permits retesting an already-null request and returns true:
https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node74.htm .
"""

import pytest


def _candidate_wait_completion(request, *, sleep):
    """Same bounded waits, deferring Python interrupts from Test as well.

    Only KeyboardInterrupt/SystemExit from the request/progress path are
    deferred. MPI.Exception derives from Exception, not either interruption
    type, and must escape immediately. Existing local sleep-error deferral is
    unchanged. The SAME request is used, including its completed/null state.
    """
    failure, delay, tests = None, 5e-5, 0
    while True:
        try:
            if request.Test():
                return failure
            tests += 1
            if tests < 8:
                continue
            try:
                sleep(delay)
            except BaseException as error:
                if failure is None:
                    failure = error
            delay = min(2*delay, 1e-3)
        except (KeyboardInterrupt, SystemExit) as error:
            if failure is None:
                failure = error


@pytest.fixture(params=["prototype", "production"])
def wait_completion(request):
    if request.param == "prototype":
        return _candidate_wait_completion
    from source.simulation.parallel import _wait_completion

    return _wait_completion


class FakeRequest:
    """Null/completed requests must remain valid for repeated Test calls."""

    def __init__(self, events, *, interruptions=(), interrupt_after_completion=False, polls=0):
        self.events = events
        self.interruptions = list(interruptions)
        self.interrupt_after_completion = interrupt_after_completion
        self.polls = polls
        self.complete = False

    def Test(self):
        self.events.append("test")
        if self.interruptions:
            if self.interrupt_after_completion:
                self.complete = True  # MPI_Test nullified the nonpersistent handle.
            raise self.interruptions.pop(0)
        if self.complete:
            self.events.append("test-null")
            return True
        if self.polls:
            self.polls -= 1
            return False
        self.complete = True
        return True


@pytest.mark.parametrize("error", [KeyboardInterrupt("test"), SystemExit(19)])
@pytest.mark.parametrize("already_completed", [False, True])
def test_interrupted_test_retries_same_live_or_null_request(error, already_completed, wait_completion):
    events = []
    request = FakeRequest(events, interruptions=[error], interrupt_after_completion=already_completed)
    returned = wait_completion(request, sleep=lambda _: pytest.fail("unexpected sleep"))
    assert returned is error
    assert request.complete
    assert events == (["test", "test", "test-null"] if already_completed else ["test", "test"])


def test_first_interrupt_object_preserved_across_repeated_test_and_sleep_interruptions(wait_completion):
    first = KeyboardInterrupt("first Test")
    events, sleeps = [], []
    request = FakeRequest(events, interruptions=[first, SystemExit(7)], polls=9)

    def interrupted_sleep(delay):
        sleeps.append(delay)
        raise KeyboardInterrupt("later sleep")

    returned = wait_completion(request, sleep=interrupted_sleep)
    assert returned is first
    assert request.complete
    assert sleeps == [5e-5, 1e-4]


@pytest.mark.parametrize("first_failure", [None, KeyboardInterrupt("earlier")])
def test_genuine_request_error_is_fatal_without_polling_or_summary(first_failure, wait_completion):
    class FakeMPIError(Exception):
        pass

    error = FakeMPIError("MPI communicator failed")
    events = []
    request = FakeRequest(events, interruptions=([first_failure] if first_failure else [])+[error])
    with pytest.raises(FakeMPIError) as raised:
        wait_completion(request, sleep=lambda _: pytest.fail("unexpected sleep"))
    assert raised.value is error
    assert not request.complete
    assert len(events) == (2 if first_failure else 1)


@pytest.mark.parametrize("body_failure", [None, ValueError("body failed first")])
def test_summary_only_after_completion_and_body_failure_keeps_precedence(body_failure, wait_completion):
    events = []
    interrupt = KeyboardInterrupt("inside request")
    request = FakeRequest(events, interruptions=[interrupt], polls=1)
    wait_failure = wait_completion(request, sleep=lambda _: None)
    failure = body_failure if body_failure is not None else wait_failure
    assert request.complete
    events.append(("allgather", f"{type(failure).__name__}: {failure}"))
    assert failure is (body_failure if body_failure is not None else interrupt)
    assert events[-1][0] == "allgather"
    assert events.count("test") == 3


def test_uninterrupted_poll_sleep_schedule_is_unchanged(wait_completion):
    events, sleeps = [], []
    request = FakeRequest(events, polls=20)
    assert wait_completion(request, sleep=sleeps.append) is None
    assert len(sleeps) == 13
    assert sleeps[:6] == [5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1e-3]
    assert sleeps[6:] == [1e-3]*7
