"""Two-rank fixed-point trial with a root-owned trace and local FVM rollback."""

from __future__ import annotations

import json
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np

from source.coupler import interface_iteration as iteration


def run_case(comm, map_value):
    rank = comm.Get_rank()
    root = rank == 0
    setup = SimpleNamespace(
        coupling_patch="numericalBoundary",
        interface_iterations=3,
        interface_normal_tolerance=1e-5,
        interface_gradient_tolerance=1e-5,
        interface_acceleration="aitken",
    )
    fvm = SimpleNamespace(
        step=10,
        time=0.5,
        field=7.0 + rank,
        boundaries=[{"name": "numericalBoundary", "normal_velocity_field": np.array([0.0])}],
        parallel=SimpleNamespace(is_parallel=True, comm=comm),
    )
    outputs = []
    fvm.write_accepted_step_output = lambda: outputs.append((fvm.step, fvm.field))
    vpm = SimpleNamespace(strength=3.0) if root else None
    transfer = SimpleNamespace(step=4)
    coupler = SimpleNamespace(
        setup=setup,
        fvm_solver=fvm,
        vpm_solver=vpm,
        vorticity_transfer=transfer,
        _is_master=root,
    )
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    areas = np.ones(2)

    def trace(value):
        velocity = np.tile([value, 0.0, 0.0], (2, 1))
        normal = np.einsum("ij,ij->i", velocity, normals)
        gradient = np.tile([0.0, value, 0.0], (2, 1))
        return velocity, normal, gradient

    initial = trace(1.0 if root else 0.0)
    iteration._write_trace(coupler, initial, old=True)
    iteration._write_trace(coupler, initial)
    inputs = []

    def capture(local_fvm):
        return local_fvm.step, local_fvm.time, local_fvm.field

    def restore(local_fvm, snapshot):
        local_fvm.step, local_fvm.time, local_fvm.field = snapshot

    def advance(local_coupler, *args):
        value = float(args[-1][0, 0])
        inputs.append(value)
        fvm.step += 1
        fvm.time += 0.05
        fvm.field = 10.0 * rank + value
        fvm.boundaries[0]["normal_velocity_field"] = np.array([value])
        return 0.0

    def renewal(*_args):
        if root:
            vpm.strength += 1.0
            transfer.step += 1
            return transfer.step, 0.0
        return None, 0.0

    def update(local_coupler, *_args):
        if root:
            iteration._write_trace(local_coupler, trace(map_value(inputs[-1])), old=True)

    saved = (
        iteration.capture_restart_payload,
        iteration.publish_restart_payload,
        iteration._particle_state_snapshot,
        iteration._restore_particle_state,
        iteration.advance_fvm,
        iteration.update_boundary_history_after_replacement,
    )
    try:
        iteration.capture_restart_payload = capture
        iteration.publish_restart_payload = restore
        iteration._particle_state_snapshot = lambda solver: solver.strength
        iteration._restore_particle_state = lambda solver, value: setattr(solver, "strength", value)
        iteration.advance_fvm = advance
        iteration.update_boundary_history_after_replacement = update
        coupler._transfer_vorticity_to_vpm = renewal
        result, _, _ = iteration.advance_iterated_interface(
            coupler, (np.zeros((2, 3)), normals, areas), initial[0]
        )
    finally:
        (
            iteration.capture_restart_payload,
            iteration.publish_restart_payload,
            iteration._particle_state_snapshot,
            iteration._restore_particle_state,
            iteration.advance_fvm,
            iteration.update_boundary_history_after_replacement,
        ) = saved
    diagnostics = coupler._last_interface_iteration_diagnostics
    return {
        "rank": rank,
        "inputs": inputs,
        "step": fvm.step,
        "field": fvm.field,
        "patch": float(fvm.boundaries[0]["normal_velocity_field"][0]),
        "transfer_step": transfer.step,
        "vpm_strength": vpm.strength if root else None,
        "result": result,
        "outputs": outputs,
        "sweeps": diagnostics["sweeps"],
        "accepted_sweep": diagnostics["accepted_sweep"],
        "rejected": diagnostics["residuals"][-1]["acceleration_rejected"],
    }


def main():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        raise RuntimeError("This regression requires exactly two MPI ranks")

    def growth(value):
        if value > 0.75:
            return 0.5
        if value > 0.45:
            return 0.4
        return 1.0

    local = run_case(comm, growth)
    gathered = comm.gather(local, root=0)
    if comm.Get_rank() == 0:
        root, worker = gathered
        assert np.allclose(root["inputs"], [1.0, 0.5, 0.375]), root
        assert worker["inputs"] == [0.0, 0.0, 0.0], worker
        assert all(item["sweeps"] == 3 and item["accepted_sweep"] == 2 for item in gathered)
        assert all(item["rejected"] for item in gathered)
        assert all(item["step"] == 11 for item in gathered)
        assert root["field"] == root["patch"] == 0.5
        assert worker["field"] == 10.0 and worker["patch"] == 0.0
        assert root["vpm_strength"] == 4.0 and root["transfer_step"] == root["result"] == 5
        assert worker["vpm_strength"] is None and worker["transfer_step"] == 4
        assert root["outputs"] == [(11, 0.5)]
        assert worker["outputs"] == [(11, 10.0)]
        print("INTERFACE_MPI_ROLLBACK_QUALIFIED " + json.dumps(gathered), flush=True)


if __name__ == "__main__":
    main()
