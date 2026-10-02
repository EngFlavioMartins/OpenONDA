"""Explicitly launched, bounded four-rank CPU/ordering qualification.

Usage: mpiexec -n 4 python tests/_nonspinning_phase_mpi.py
This does not import any solver, FVM, Taichi, or GPU module. JSON goes to stdout.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

from _nonspinning_phase_prototype import cooperative_collective_phase
from mpi4py import MPI


@contextmanager
def legacy_phase(comm, description):
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
    summary = None if failure is None else f"{type(failure).__name__}: {failure}"
    summaries = comm.allgather(summary)
    first = next(((rank, value) for rank, value in enumerate(summaries) if value), None)
    if first is not None and failure is None:
        rank, message = first
        raise RuntimeError(f"{description} failed on rank {rank}: {message}")
    if failure is not None:
        raise failure


def measure(comm, phase, *, repetitions, owner_delay):
    comm.Barrier()
    started, cpu = time.perf_counter(), time.process_time()
    for _index in range(repetitions):
        with phase(comm, "owner-only"):
            if comm.Get_rank() == 0 and owner_delay:
                time.sleep(owner_delay)
    row = {
        "rank": comm.Get_rank(),
        "wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu,
    }
    rows = comm.gather(row, root=0)
    return {"repetitions": repetitions, "owner_delay": owner_delay, "ranks": rows}


def qualify_failures(comm, phase):
    rank = comm.Get_rank()
    cases = [
        {0: ValueError("owner")},
        {2: ValueError("worker")},
        {1: KeyboardInterrupt("interrupt")},
        {3: SystemExit(17)},
        {0: ValueError("first"), 2: SystemExit(29)},
    ]
    for errors in cases:
        local = errors.get(rank)
        try:
            with phase(comm, "failure"):
                if rank == 0:
                    time.sleep(0.01)
                if local is not None:
                    raise local
        except BaseException as caught:
            if local is not None:
                assert caught is local
            else:
                first = min(errors)
                value = errors[first]
                assert type(caught) is RuntimeError
                assert str(caught) == (
                    f"failure failed on rank {first}: {type(value).__name__}: {value}"
                )
        else:
            raise AssertionError("failure did not propagate")
        # Proves all ranks completed precisely the same collective sequence.
        with phase(comm, "after failure"):
            pass
        assert comm.allgather(rank) == list(range(4))
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    phase = cooperative_collective_phase
    module_path = Path(__file__).with_name("_nonspinning_phase_prototype.py")
    if args.production:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from source.simulation.parallel import collective_phase

        phase = collective_phase
        module_path = Path(__file__).resolve().parents[1] / "source/simulation/parallel.py"
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 4, "Qualification requires exactly four MPI ranks"
    output = args.output
    admissible = None
    if comm.Get_rank() == 0:
        admissible = output is None or (output.parent.is_dir() and not output.exists())
    assert comm.bcast(admissible, root=0), "Output must be new and its directory must exist"
    failures = qualify_failures(comm, phase)
    results = {}
    for label, implementation in (("legacy", legacy_phase), ("cooperative", phase)):
        results[label] = {
            "long_owner": measure(comm, implementation, repetitions=4, owner_delay=0.2),
            "empty_body": measure(comm, implementation, repetitions=2000, owner_delay=0),
        }
    if comm.Get_rank() == 0:
        report = {
            "status": "passed",
            "implementation": "production" if args.production else "prototype",
            "module": str(module_path),
            "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
            "failure_cases": failures,
            "results": results,
        }
        if output is not None:
            with output.open("x") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
