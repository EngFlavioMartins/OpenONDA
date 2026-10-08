"""Picard sweeps restart from one accepted state and publish only the endpoint."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from openonda.coupler import CouplerSetup
from openonda.coupler import interface_iteration as iteration

NORMALS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


def trace(value):
    """Uniform interface trace whose three fields all scale with ``value``."""
    velocity = np.tile([value, 0.0, 0.0], (2, 1))
    return velocity, np.einsum("ij,ij->i", velocity, NORMALS), np.tile([0.0, value, 0.0], (2, 1))


@pytest.fixture
def rig(monkeypatch):
    """Scalar stand-ins: one sweep maps the imposed trace value v to 0.01 v."""
    fvm = SimpleNamespace(step=10, time=0.5, field=7.0, parallel=SimpleNamespace(is_parallel=False))
    vpm = SimpleNamespace(strength=3.0)
    transfer = SimpleNamespace(step=4)
    c = SimpleNamespace(
        setup=CouplerSetup(
            interface_iterations=6,
            interface_normal_tolerance=1e-5,
            interface_gradient_tolerance=1e-5,
        ),
        fvm_solver=fvm,
        vpm_solver=vpm,
        vorticity_transfer=transfer,
        _is_master=True,
        interface_trace_correction=None,
        map_value=lambda value: 0.01 * value,
    )
    monkeypatch.setattr(iteration, "capture_restart_state", lambda f: (f.step, f.time, f.field))

    def restore(f, state):
        f.step, f.time, f.field = state

    monkeypatch.setattr(iteration, "restore_restart_state", restore)
    monkeypatch.setattr(iteration, "_particle_state_snapshot", lambda v, **kwargs: v.strength)
    monkeypatch.setattr(
        iteration, "_restore_particle_state", lambda v, p: setattr(v, "strength", p)
    )
    old, raw = trace(0.3), trace(1.0)
    iteration._write_trace(c, old, old=True)
    iteration._write_trace(c, raw)
    inputs, starts, outputs = [], [], []
    fvm.write_accepted_step_output = lambda: outputs.append((fvm.step, fvm.time, fvm.field))

    def advance(owner, *args):
        value = float(args[-1][0, 0])
        assert np.array_equal(args[-2], old[0]), "the physical old endpoint changed"
        assert np.array_equal(owner._normal_velocity_boundary_condition, trace(value)[1])
        assert np.array_equal(owner._tangential_gradient_boundary_condition, trace(value)[2])
        starts.append((fvm.step, fvm.time, fvm.field, vpm.strength, transfer.step))
        inputs.append(value)
        fvm.step, fvm.time, fvm.field = fvm.step + 5, fvm.time + 0.05, value
        return 0.01

    monkeypatch.setattr(iteration, "advance_fvm", advance)

    def renewal(*args):
        vpm.strength += 2.0
        transfer.step += 1
        return transfer.step, 0.01

    c._transfer_vorticity_to_vpm = renewal

    def update(owner, *args):
        iteration._write_trace(owner, trace(owner.map_value(inputs[-1])), old=True)

    monkeypatch.setattr(iteration, "update_boundary_history_after_replacement", update)
    geometry = np.zeros((2, 3)), NORMALS, np.ones(2)
    return SimpleNamespace(
        c=c,
        fvm=fvm,
        vpm=vpm,
        transfer=transfer,
        raw=raw,
        inputs=inputs,
        starts=starts,
        outputs=outputs,
        run=lambda: iteration.advance_iterated_interface(c, geometry, raw[0]),
    )


def test_cold_start_iterates_from_the_vpm_predictor(rig):
    rig.run()
    assert rig.inputs == pytest.approx([1.0, 0.01, 1e-4, 1e-6])
    assert rig.starts == [(10, 0.5, 7.0, 3.0, 4)] * 4
    assert (rig.fvm.step, rig.transfer.step, rig.vpm.strength) == (15, 5, 5.0)
    assert rig.outputs == [(15, 0.55, pytest.approx(1e-6))]
    diagnostics = rig.c._last_interface_iteration_diagnostics
    assert diagnostics["converged"] and diagnostics["sweeps"] == 4
    assert diagnostics["initial_guess"] == "vpm_trace"
    # The stored correction maps this predictor onto the converged endpoint.
    correction = rig.c.interface_trace_correction
    assert np.allclose(rig.raw[0] + correction[0], trace(1e-8)[0])


def test_previous_correction_converges_in_one_sweep(rig):
    rig.c.interface_trace_correction = trace(-1.0)
    rig.run()
    assert rig.inputs == [0.0]
    assert rig.outputs == [(15, 0.55, 0.0)]
    diagnostics = rig.c._last_interface_iteration_diagnostics
    assert diagnostics["sweeps"] == 1 and diagnostics["initial_guess"] == "previous_correction"
    assert np.array_equal(rig.c.interface_trace_correction[0], trace(-1.0)[0])


def test_inexact_guess_continues_picard_from_its_own_sweep(rig):
    rig.c.interface_trace_correction = trace(1.0)
    rig.run()
    assert rig.inputs == pytest.approx([2.0, 0.02, 2e-4, 2e-6])
    assert len(set(rig.starts)) == 1
    assert len(rig.outputs) == 1


def test_nonconverged_iteration_keeps_no_correction(rig):
    rig.c.interface_trace_correction = trace(1.0)
    rig.c.map_value = lambda value: 0.5 * value
    rig.run()
    assert len(rig.inputs) == rig.c.setup.interface_iterations
    assert not rig.c._last_interface_iteration_diagnostics["converged"]
    assert rig.c.interface_trace_correction is None
    assert len(rig.outputs) == 1


def test_new_solve_and_backup_load_start_cold(monkeypatch):
    from openonda.coupler import solver as driver

    owner = driver.FVMVPMCoupler.__new__(driver.FVMVPMCoupler)
    owner._comm = None
    owner.interface_trace_correction = trace(1.0)

    def prepare():
        assert owner.interface_trace_correction is None
        raise RuntimeError("stop before any solver work")

    owner._prepare_run = prepare
    with pytest.raises(RuntimeError, match="before any solver work"):
        owner.solve()
    owner.interface_trace_correction = trace(1.0)

    def load(*args, **kwargs):
        assert owner.interface_trace_correction is None
        return 11

    monkeypatch.setattr(driver, "load_coupled_backup", load)
    assert owner.load_backup("unused-checkpoint") == 11


def test_iteration_rejects_output_inside_provisional_interval():
    from openonda.fvm import RunSchedule

    fvm = SimpleNamespace(
        _output_schedule=RunSchedule(every_n_steps=5),
        _backup_config=SimpleNamespace(schedule=None),
        _sampler_schedules={0: RunSchedule(every_n_steps=2)},
    )
    c = SimpleNamespace(fvm_solver=fvm, n_fvm_substeps=5, vpm_time_step_size=0.05)
    with pytest.raises(ValueError, match="aligned"):
        iteration.validate_output_schedules(c)


@pytest.mark.integration
def test_two_rank_sweeps_follow_the_master_decision():
    pytest.importorskip("mpi4py")
    launcher = Path(sys.executable).with_name("mpiexec")
    if not launcher.is_file():
        pytest.skip("MPI launcher is unavailable")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [
            str(launcher),
            "-n",
            "2",
            sys.executable,
            str(Path(__file__).with_name("_interface_iteration_mpi.py")),
        ],
        text=True,
        capture_output=True,
        timeout=90,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"sweep_counts": [1, 4, 4]' in result.stdout
