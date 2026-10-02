"""Two-rank speculative seed admission, rollback, and cold disable checks."""

import json
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np

from source.coupler import interface_iteration as iteration
from source.coupler import interface_prediction as prediction


def run_case(comm, *, correction, disable_worker=False, failure_at=None, failure_rank=0):
    master = comm.Get_rank() == 0
    fvm = SimpleNamespace(
        step=10,
        time=0.5,
        value=float(comm.Get_rank()),
        parallel=SimpleNamespace(comm=comm, is_parallel=True),
        boundaries=[{"name": "outer", "normal_velocity_field": np.zeros(2)}],
    )
    transfer = SimpleNamespace(step=4, last_interface_flow={"marker": 0})
    vpm = SimpleNamespace(strength=3.0) if master else None
    c = SimpleNamespace(
        fvm_solver=fvm,
        vpm_solver=vpm,
        vorticity_transfer=transfer,
        _is_master=master,
        setup=SimpleNamespace(
            coupling_patch="outer",
            interface_iterations=3,
            interface_normal_tolerance=1e-5,
            interface_gradient_tolerance=1e-5,
            interface_acceleration="none",
        ),
        interface_predictor=prediction.SafeguardedInterfacePredictor(),
    )
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    geometry = np.zeros((2, 3)), normals, np.ones(2)

    def trace(value):
        u = np.tile([value, 0.0, 0.0], (2, 1))
        return u, np.einsum("ij,ij->i", u, normals), np.tile([0.0, value, 0.0], (2, 1))

    old, raw = trace(0.3 if master else 0.0), trace(1.0 if master else 0.0)
    iteration._write_trace(c, old, old=True)
    iteration._write_trace(c, raw)
    operator = object()
    prediction._identity = lambda owner, geo: ("exact-fixture-controls", (operator,))
    c.interface_predictor._history = {
        "identity": prediction._identity(c, geometry),
        "clock": (10, 0.5),
        "endpoint": prediction._copy_trace(old),
        "correction": trace(correction),
    }
    if disable_worker and not master:
        c.interface_predictor.enabled = False
    starts, inputs, outputs = [], [], []
    iteration.capture_restart_payload = lambda local: (local.step, local.time, local.value)

    def restore(local, state):
        local.step, local.time, local.value = state

    iteration.publish_restart_payload = restore
    iteration._particle_state_snapshot = lambda owner: owner.strength
    iteration._restore_particle_state = lambda owner, state: setattr(owner, "strength", state)

    def advance(owner, *args):
        value = comm.bcast(float(args[-1][0, 0]) if master else None, root=0)
        if master:
            assert np.array_equal(args[-2], old[0])
            assert np.array_equal(owner._normal_velocity_boundary_condition, trace(value)[1])
            assert np.array_equal(owner._tangential_gradient_boundary_condition, trace(value)[2])
        starts.append((fvm.step, fvm.time, fvm.value, dict(transfer.last_interface_flow)))
        inputs.append(value)
        fvm.step += 1
        fvm.time += 0.05
        fvm.value = comm.Get_rank() + value
        fvm.boundaries[0]["normal_velocity_field"] = trace(value)[1]
        return 0.0

    iteration.advance_fvm = advance

    def transfer_step(*args):
        if master:
            vpm.strength += 2.0
        transfer.step += 1
        transfer.last_interface_flow = {"marker": len(inputs)}
        return transfer.step, 0.0

    c._transfer_vorticity_to_vpm = transfer_step

    def update(owner, *args):
        if master:
            iteration._write_trace(owner, trace(0.01 * inputs[-1]), old=True)

    iteration.update_boundary_history_after_replacement = update
    fvm.write_accepted_step_output = lambda: outputs.append((fvm.step, fvm.time, fvm.value))
    patches = []

    def patch(owner, name, value):
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    failure = MemoryError(f"injected {failure_at} on rank {failure_rank}")

    def fail(*args, **kwargs):
        raise failure

    if failure_at is not None and comm.Get_rank() == failure_rank:
        hooks = {
            "initial_capture": (iteration, "capture_restart_payload"),
            "eligibility": (prediction, "_clock"),
            "seed_algebra": (prediction, "_same_arrays"),
            "worker_placeholder": (prediction, "_copy_trace"),
            "seed_capture": (iteration, "_capture_trial_fallback"),
            "seed_install": (iteration, "_write_trace"),
            "seed_restore": (iteration, "_restore_trial_fallback"),
            "stage": (prediction, "_copy_trace"),
        }
        patch(*hooks[failure_at], fail)
    caught = None
    try:
        iteration.advance_iterated_interface(c, geometry, raw[0])
    except BaseException as error:
        caught = error
    finally:
        for owner, name, value in patches:
            setattr(owner, name, value)
    if failure_at is not None:
        summaries = comm.allgather(None if caught is None else str(caught))
        assert all(
            summary is not None and f"injected {failure_at}" in summary for summary in summaries
        )
        if comm.Get_rank() == failure_rank:
            assert caught is failure
        assert c.interface_predictor._history is None
        assert c.interface_predictor._pending is None
        assert c.interface_predictor._active_identity is None
        assert len(inputs) == (1 if failure_at in ("seed_restore", "stage") else 0)
        assert len(outputs) == (1 if failure_at == "stage" else 0)
        c.interface_predictor.commit()
        assert c.interface_predictor._history is None
        return 1
    if caught is not None:
        raise caught
    if disable_worker:
        assert inputs == [1.0, 0.01, 0.0001]
        assert not c._last_interface_iteration_diagnostics["prediction"]["attempted"]
    elif correction == -1.0:
        assert inputs == [0.0]
        assert c._last_interface_iteration_diagnostics["prediction"]["accepted"]
    else:
        assert inputs == [2.0, 1.0, 0.01, 0.0001]
        assert starts[0] == starts[1]
        assert c._last_interface_iteration_diagnostics["prediction"]["fallback"]
    assert all(start[:2] == (10, 0.5) for start in starts)
    assert len(outputs) == 1 and fvm.step == 11 and transfer.step == 5
    if master:
        assert vpm.strength == 5.0
    records = comm.allgather(c._last_interface_iteration_diagnostics["residuals"])
    assert records[0] == records[1]
    return len(inputs)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    counts = [
        run_case(comm, correction=-1.0),
        run_case(comm, correction=1.0),
        run_case(comm, correction=-1.0, disable_worker=True),
    ]
    failed_cases = [
        ("initial_capture", 0),
        ("initial_capture", 1),
        ("eligibility", 0),
        ("eligibility", 1),
        ("seed_algebra", 0),
        ("worker_placeholder", 1),
        ("seed_capture", 0),
        ("seed_capture", 1),
        ("seed_install", 0),
        ("seed_install", 1),
        ("seed_restore", 0),
        ("seed_restore", 1),
        ("stage", 0),
    ]
    failures = sum(
        run_case(
            comm,
            correction=1.0 if where == "seed_restore" else -1.0,
            failure_at=where,
            failure_rank=rank,
        )
        for where, rank in failed_cases
    )
    if comm.Get_rank() == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "trial_counts": counts,
                    "ranks": comm.Get_size(),
                    "collective_failure_cases": failures,
                }
            )
        )
