"""Iteration must restore predictors and publish only the accepted endpoint."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import CouplerSetup
from source.coupler import interface_iteration as iteration


def test_fixed_predictor_map_does_not_accumulate_time_or_particles(monkeypatch):
    cfg = CouplerSetup(
        transfer_method="buffered_m4_renewal",
        boundary_condition_mode="vorticity_mixed",
        interface_iterations=12,
        interface_normal_tolerance=1e-4,
        interface_gradient_tolerance=1e-4,
    )
    written = []
    fvm = SimpleNamespace(step=10, time=0.5, parallel=SimpleNamespace(is_parallel=False))
    fvm.write_accepted_step_output = lambda: written.append((fvm.step, fvm.time))
    vpm = SimpleNamespace(strength=3.0, step=11, time=0.55)
    transfer = SimpleNamespace(step=4)
    c = SimpleNamespace(
        setup=cfg, fvm_solver=fvm, vpm_solver=vpm, vorticity_transfer=transfer, _is_master=True
    )
    old = (np.ones((2, 3)), np.ones(2), np.ones((2, 3)))
    iteration._write_trace(c, old, old=True)
    iteration._write_trace(c, old)
    starts = []

    monkeypatch.setattr(iteration, "capture_restart_payload", lambda f: (f.step, f.time))

    def restore(f, state):
        f.step, f.time = state

    monkeypatch.setattr(iteration, "publish_restart_payload", restore)
    monkeypatch.setattr(iteration, "_particle_state_snapshot", lambda v: v.strength)
    monkeypatch.setattr(
        iteration, "_restore_particle_state", lambda v, p: setattr(v, "strength", p)
    )

    def advance(coupler, *args):
        starts.append((fvm.step, fvm.time, vpm.strength, transfer.step))
        fvm.step += 5
        fvm.time += 0.05
        c.trace = iteration._read_trace(c, velocity=args[-1])
        return 0.01

    monkeypatch.setattr(iteration, "advance_fvm", advance)

    def renewal(*args):
        vpm.strength += 2.0
        transfer.step += 1
        return object(), 0.01

    c._transfer_vorticity_to_vpm = renewal

    def update(coupler, *args):
        iteration._write_trace(c, tuple(0.2 * value for value in c.trace), old=True)

    monkeypatch.setattr(iteration, "update_boundary_history_after_replacement", update)
    iteration.advance_iterated_interface(
        c, (np.zeros((2, 3)), np.zeros((2, 3)), np.ones(2)), old[0]
    )
    assert len(starts) > 2
    assert starts == [(10, 0.5, 3.0, 4)] * len(starts)
    assert (fvm.step, vpm.step, transfer.step, vpm.strength) == (15, 11, 5, 5.0)
    assert written == [(15, 0.55)]
    assert c._last_interface_iteration_diagnostics["converged"]


def test_iteration_rejects_output_inside_provisional_interval():
    from source.solvers.fvm import RunSchedule

    fvm = SimpleNamespace(
        _output_schedule=RunSchedule(every_n_steps=5),
        _backup_config=SimpleNamespace(schedule=None),
        _sampler_schedules={0: RunSchedule(every_n_steps=2)},
    )
    c = SimpleNamespace(fvm_solver=fvm, n_fvm_substeps=5, vpm_time_step_size=0.05)
    with pytest.raises(ValueError, match="aligned"):
        iteration.validate_output_schedules(c)
