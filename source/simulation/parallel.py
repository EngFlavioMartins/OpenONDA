"""Error propagation around local work in a collective solver lifecycle."""

from contextlib import contextmanager


@contextmanager
def collective_phase(comm, description: str):
    """Finish local work on every rank before propagating any failure.

    All ranks enter in the same order. The body must contain local work only;
    this cannot recover a failure inside an MPI collective. Exception summaries
    are exchanged rather than potentially unpickleable exception objects.
    """
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
    if comm is not None and comm.Get_size() > 1:
        summary = None if failure is None else f"{type(failure).__name__}: {failure}"
        summaries = comm.allgather(summary)
        first = next(((rank, value) for rank, value in enumerate(summaries) if value), None)
        if first is not None and failure is None:
            rank, message = first
            raise RuntimeError(f"{description} failed on rank {rank}: {message}")
    if failure is not None:
        raise failure
