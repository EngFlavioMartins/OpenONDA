"""Two-rank benchmark trace instrumentation; no FVM, VPM, or GPU imports."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np

ASSETS = (
    Path(__file__).resolve().parents[2]
    / "tests/support/cylinder"
)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ASSETS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scenario(comm, output_dir, kind):
    capture = load("capture_interface_traces")
    master = comm.Get_rank() == 0
    owner = SimpleNamespace(
        _is_master=master,
        fvm_solver=SimpleNamespace(step=5, time=0.2, parallel=SimpleNamespace(comm=comm)),
        vpm_solver=SimpleNamespace(step=2, time=0.4) if master else None,
    )
    if master:
        for suffix, value in (("", 2.0), ("_old", 1.0)):
            owner.__dict__["_velocity_boundary_condition" + suffix] = np.full((2, 3), value)
            owner.__dict__["_normal_velocity_boundary_condition" + suffix] = np.full(2, value)
            owner.__dict__["_tangential_gradient_boundary_condition" + suffix] = np.full(
                (2, 3), value
            )
    module = SimpleNamespace()
    calls = []

    def advance(current, *args):
        assert current is owner
        calls.append("advance")
        comm.Barrier()
        owner.fvm_solver.step += 5
        owner.fvm_solver.time = 0.4

    def refresh(current, *args):
        assert current is owner
        calls.append("refresh")
        comm.Barrier()
        if master:
            owner._velocity_boundary_condition_old[:] = 3
            owner._normal_velocity_boundary_condition_old[:] = 3
            owner._tangential_gradient_boundary_condition_old[:] = 3

    def iterate(current, geometry, candidate):
        module.advance_fvm(current, *geometry, None, candidate)
        module.update_boundary_history_after_replacement(current, *geometry)
        if master:
            owner._last_interface_iteration_diagnostics = {
                "sweeps": 1,
                "accepted_sweep": 1,
                "converged": True,
                "residuals": [{"sweep": 1, "accepted": True}],
            }
        return 41

    module.advance_iterated_interface = iterate
    module.advance_fvm = advance
    module.update_boundary_history_after_replacement = refresh
    originals = vars(module).copy()
    original_save = capture.np.savez
    path = output_dir / f"{kind}-step000002.npz"
    if master and kind == "admission":
        path.write_bytes(b"preserve me")
    if master and kind == "write_failure":

        def fail_save(*args, **kwargs):
            raise OSError("injected master NPZ write failure")

        capture.np.savez = fail_save
    comm.Barrier()
    reports = []
    error = None
    result = None
    try:
        with capture.capture_interface_traces(
            owner, output_dir / kind, reports, max_exchanges=1, iteration_module=module
        ):
            result = module.advance_iterated_interface(
                owner,
                (np.zeros((2, 3)), np.ones((2, 3)), np.ones(2)),
                owner._velocity_boundary_condition if master else np.empty((0, 3)),
            )
    except RuntimeError as exc:
        error = str(exc)
    finally:
        capture.np.savez = original_save
    assert vars(module) == originals
    gathered = comm.allgather(
        {
            "rank": comm.Get_rank(),
            "error": error,
            "calls": calls,
            "reports": reports,
            "result": result,
        }
    )
    if kind == "success":
        assert all(row["error"] is None and row["result"] == 41 for row in gathered), gathered
        assert len(gathered[0]["reports"]) == 1 and not gathered[1]["reports"]
        if master:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"]))
                assert metadata["status"] == "complete"
                assert metadata["accepted_sweep"] == 1
                assert [row["label"] for row in metadata["events"]] == [
                    "old_physical_endpoint",
                    "raw_predictor",
                    "trial_input",
                    "trial_output",
                    "accepted_endpoint",
                ]
    else:
        assert gathered[0]["error"] == gathered[1]["error"] and error is not None, gathered
        expected = [] if kind == "admission" else ["advance", "refresh"]
        assert all(row["calls"] == expected for row in gathered)
        if kind == "admission" and master:
            assert path.read_bytes() == b"preserve me"
    comm.Barrier()
    return gathered


def qualify_preflight(comm, output_dir):
    benchmark = load("benchmark_coupled_checkpoint")
    case = output_dir / "case"
    solution = case / "solution"
    if comm.Get_rank() == 0:
        solution.mkdir(parents=True)
    comm.Barrier()
    report = solution / "fresh.json"
    prefix = solution / "fresh-trace"
    # Rank1 is forbidden from consulting shared existence. After the single
    # collective root decision, fast root writes before the simulated slow
    # rank finishes. No second per-rank filesystem admission is permitted.
    original_exists = Path.exists
    original_glob = Path.glob
    if comm.Get_rank() != 0:

        def forbidden(*args, **kwargs):
            raise AssertionError("Worker must not recheck shared output admission")

        Path.exists = forbidden
        Path.glob = forbidden
    try:
        benchmark._collective_output_preflight(comm, case, report, prefix, 1)
        if comm.Get_rank() == 0:
            report.write_text("reserved by root")
            (solution / "fresh-trace-step000002.npz").write_bytes(b"root trace")
        comm.Barrier()
        error = None
        try:
            benchmark._collective_output_preflight(comm, case, report, prefix, 1)
        except ValueError as exc:
            error = str(exc)
        errors = comm.allgather(error)
        assert errors[0] == errors[1] and "new report path" in errors[0], errors
    finally:
        Path.exists = original_exists
        Path.glob = original_glob


def main():
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    output_dir = Path(sys.argv[1])
    results = [
        scenario(comm, output_dir, kind) for kind in ("success", "admission", "write_failure")
    ]
    qualify_preflight(comm, output_dir)
    if comm.Get_rank() == 0:
        print("INTERFACE_TRACE_MPI_QUALIFIED " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
