"""Bounded MPI-only production Request.Test interruption qualification.

Explicit usage: mpiexec -n 4 python tests/_phase_test_interrupt_mpi.py
No solver, Taichi or GPU is imported. A proxy injects a Python interruption
before the first Test, or after a real successful Test has nullified the owned
request. Every phase must reach its error-summary collective and then another
clean phase. This is deterministic failure injection, not a signal stress test.
"""

import hashlib
import json
from pathlib import Path
import sys
import time

from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.simulation import parallel


class InjectedRequest:
    def __init__(self, request, error, after_completion):
        self.request = request
        self.error = error
        self.after_completion = after_completion
        self.injected = False
        self.retested_null = False

    def Test(self):
        if self.injected and self.request == MPI.REQUEST_NULL:
            self.retested_null = True
        if not self.injected and not self.after_completion:
            self.injected = True
            raise self.error
        result = self.request.Test()
        if result and not self.injected:
            assert self.request == MPI.REQUEST_NULL
            self.injected = True
            raise self.error
        return result


def main():
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 4
    rank = comm.Get_rank()
    original_wait = parallel._wait_completion
    cases = [
        (1, KeyboardInterrupt("before Test"), False, None),
        (2, SystemExit(17), False, None),
        (3, KeyboardInterrupt("after completed Test"), True, None),
        (0, SystemExit(19), True, None),
        (1, KeyboardInterrupt("secondary wait"), False, 1),
        (2, SystemExit(23), True, 1),
    ]
    verified_null_retests = 0
    for interrupted_rank, interrupt, after_completion, body_rank in cases:
        requests = []
        body_error = ValueError("earlier body failure") if rank == body_rank else None

        def injected_wait(
            request, fault_rank=interrupted_rank, fault=interrupt,
            completed=after_completion, captures=requests,
        ):
            if rank == fault_rank:
                proxy = InjectedRequest(request, fault, completed)
                captures.append(proxy)
                return original_wait(proxy)
            return original_wait(request)

        parallel._wait_completion = injected_wait
        local_error = body_error if body_error is not None else (
            interrupt if rank == interrupted_rank else None
        )
        error_ranks = {interrupted_rank}
        if body_rank is not None:
            error_ranks.add(body_rank)
        try:
            with parallel.collective_phase(comm, "injected Test"):
                if rank == 0:
                    time.sleep(0.02)
                if body_error is not None:
                    raise body_error
        except BaseException as error:
            if local_error is not None:
                assert error is local_error
            else:
                assert isinstance(error, RuntimeError)
                assert f"failed on rank {min(error_ranks)}:" in str(error)
        else:
            raise AssertionError("injected interruption did not propagate")
        finally:
            parallel._wait_completion = original_wait
        if requests:
            assert requests[0].injected
            assert requests[0].request == MPI.REQUEST_NULL
            if after_completion:
                assert requests[0].retested_null
                verified_null_retests += 1
        with parallel.collective_phase(comm, "clean phase after interruption"):
            pass
        assert comm.allgather(rank) == [0, 1, 2, 3]
    # A real mpi4py exception injected into Test remains fatal locally. It is
    # tested without creating an MPI request/collective that could be abandoned.
    mpi_error = MPI.Exception(MPI.ERR_OTHER)

    class FailedRequest:
        def Test(self):
            raise mpi_error

    try:
        original_wait(FailedRequest())
    except MPI.Exception as error:
        assert error is mpi_error
    else:
        raise AssertionError("MPI error was swallowed")
    null_retests = comm.reduce(verified_null_retests, op=MPI.SUM, root=0)
    if rank == 0:
        source = Path(parallel.__file__)
        print(json.dumps({
            "status": "passed", "ranks": 4, "injected_phase_cases": len(cases),
            "verified_null_retests": null_retests, "real_mpi_exception_preserved": True,
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }), flush=True)


if __name__ == "__main__":
    main()
