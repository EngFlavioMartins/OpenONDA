"""Two-rank interface sweeps: workers follow the master's residual decision."""

import json
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np

from openonda.coupler import interface_iteration as iteration

NORMALS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


def trace(value):
    velocity = np.tile([value, 0.0, 0.0], (2, 1))
    return velocity, np.einsum("ij,ij->i", velocity, NORMALS), np.tile([0.0, value, 0.0], (2, 1))


def run_case(comm, correction):
    master = comm.Get_rank() == 0
    fvm = SimpleNamespace(step=10, time=0.5, parallel=SimpleNamespace(comm=comm, is_parallel=True))
    vpm = SimpleNamespace(strength=3.0) if master else None
    transfer = SimpleNamespace(step=4)
    c = SimpleNamespace(
        fvm_solver=fvm,
        vpm_solver=vpm,
        vorticity_transfer=transfer,
        _is_root=master,
        config=SimpleNamespace(
            interface_iterations=4,
            interface_normal_tolerance=1e-5,
            interface_gradient_tolerance=1e-5,
        ),
        # Only the master owns the global trace and its correction.
        interface_trace_correction=trace(correction) if master and correction else None,
    )
    old, raw = trace(0.3 if master else 0.0), trace(1.0 if master else 0.0)
    iteration._write_trace(c, old, old=True)
    iteration._write_trace(c, raw)
    iteration.capture_restart_state = lambda local: (local.step, local.time)

    def restore(local, state):
        local.step, local.time = state

    iteration.restore_restart_state = restore
    iteration._particle_state_snapshot = lambda owner, **kwargs: owner.strength
    iteration._restore_particle_state = lambda owner, state: setattr(owner, "strength", state)
    inputs, outputs = [], []

    def advance(owner, *args):
        # The FVM boundary update is collective; the master supplies the trace.
        inputs.append(comm.bcast(float(args[-1][0, 0]) if master else None, root=0))
        fvm.step, fvm.time = fvm.step + 1, fvm.time + 0.05
        return 0.0

    iteration.advance_fvm = advance

    def transfer_step(*args):
        if master:
            vpm.strength += 2.0
        transfer.step += 1
        return transfer.step, 0.0

    c._transfer_vorticity_to_vpm = transfer_step

    def update(owner, *args):
        if master:
            iteration._write_trace(owner, trace(0.1 * inputs[-1]), old=True)

    iteration.update_boundary_history_after_replacement = update
    fvm.write_step_output = lambda: outputs.append((fvm.step, fvm.time))
    iteration.advance_iterated_interface(c, (np.zeros((2, 3)), NORMALS, np.ones(2)), raw[0])
    assert len(outputs) == 1 and fvm.step == 11 and transfer.step == 5
    records = comm.allgather(c._last_interface_iteration_diagnostics["residuals"])
    assert records[0] == records[1]
    assert comm.allgather(inputs)[0] == comm.allgather(inputs)[1]
    return len(inputs)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    counts = [run_case(comm, correction) for correction in (-1.0, 1.0, None)]
    if comm.Get_rank() == 0:
        print(json.dumps({"status": "passed", "sweep_counts": counts}))
