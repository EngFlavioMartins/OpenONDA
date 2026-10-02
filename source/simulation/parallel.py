"""Error propagation around local work in a collective solver lifecycle."""

from contextlib import contextmanager
import time


def _wait_completion(request, *, sleep=time.sleep):
    """Complete a phase rendezvous without spending the wait busy-polling.

    A few immediate tests keep short phases cheap. Longer waits sleep for
    50 microseconds initially, increasing to at most one millisecond between
    progress tests. A Python interruption during a progress test or sleep is
    deferred until the collective completes, so every rank can enter the
    error-summary exchange. Retesting the same request is valid even when an
    interrupted Test already completed it and set its handle to REQUEST_NULL.
    MPI request errors still propagate: a broken communicator is not recoverable
    here. Never abandon, cancel, or time out a live collective locally.
    """
    failure = None
    delay = 5e-5
    tests = 0
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
            delay = min(2 * delay, 1e-3)
        except (KeyboardInterrupt, SystemExit) as error:
            # Do not catch MPI.Exception or other request failures: continuing
            # a broken communicator cannot restore collective ordering.
            if failure is None:
                failure = error


@contextmanager
def collective_phase(comm, description: str):
    """Finish local work on every rank before propagating any failure.

    All ranks enter in the same order. The body must contain local work only;
    this cannot recover a failure inside an MPI collective. Exception summaries
    are exchanged rather than potentially unpickleable exception objects.
    MPI-3 communicators first rendezvous with bounded polling and sleeping, so
    idle ranks do not busy-spin while another rank performs long local work.
    Serial execution does not enter collectives.
    """
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
    if comm is not None and comm.Get_size() > 1:
        interruption = _wait_completion(comm.Ibarrier())
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
