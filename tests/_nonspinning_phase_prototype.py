"""UNWIRED qualification prototype for low-CPU collective phase completion.

The original error-summary allgather remains authoritative. A nonblocking
barrier first establishes that every local-only body has finished; bounded
polling lets otherwise idle ranks sleep while a long owner-only task runs.
No production module imports this prototype.
"""

from contextlib import contextmanager
import time


def _wait_completion(request, *, sleep=time.sleep, spin_tests=8, first_sleep=5e-5, max_sleep=1e-3):
    """Return a deferred Python interruption after completing the rendezvous.

Do not cancel/free a live collective or abandon it on a local timeout: either
would let ranks issue different collective sequences. MPI Test failures are
not recoverable here and deliberately propagate, just as MPI failures in the
original allgather do. An interruption raised by sleep is deferred until every
rank reaches the existing error-summary exchange.
"""
    failure = None
    delay = first_sleep
    tests = 0
    while not request.Test():
        tests += 1
        if tests < spin_tests:
            continue
        try:
            sleep(delay)
        except BaseException as error:
            if failure is None:
                failure = error
        delay = min(2 * delay, max_sleep)
    return failure


@contextmanager
def cooperative_collective_phase(comm, description, *, sleep=time.sleep):
    """Match collective_phase exception semantics without long busy-waits.

Serial execution and legacy/fake communicators lacking Ibarrier retain their
original path. MPI-3 communicators use exactly Ibarrier then allgather on every
rank. This still requires an intact communicator and a local-only body; it
cannot recover a hung rank, a process exit, or an unmatched MPI operation.
"""
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
    if comm is not None and comm.Get_size() > 1:
        begin = getattr(comm, "Ibarrier", None)
        if callable(begin):
            interruption = _wait_completion(begin(), sleep=sleep)
            if failure is None:
                failure = interruption
        summary = None if failure is None else f"{type(failure).__name__}: {failure}"
        summaries = comm.allgather(summary)
        first = next(((rank, value) for rank, value in enumerate(summaries) if value), None)
        if first is not None and failure is None:
            rank, message = first
            raise RuntimeError(f"{description} failed on rank {rank}: {message}")
    if failure is not None:
        raise failure
