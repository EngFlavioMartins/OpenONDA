"""Coupling iterations restart from one accepted state and write only the accepted end of the interval."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from openonda.coupler import CouplerConfig
from openonda.coupler import coupling_iteration as iteration

NORMALS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


def boundary_data(value):
    """Uniform boundary data whose three fields all scale with ``value``."""
    velocity = np.tile([value, 0.0, 0.0], (2, 1))
    return velocity, np.einsum("ij,ij->i", velocity, NORMALS), np.tile([0.0, value, 0.0], (2, 1))


@pytest.fixture
def rig(monkeypatch):
    """Scalar stand-ins: one coupling iteration maps the imposed boundary-data value v to 0.01 v."""
    fvm = SimpleNamespace(step=10, time=0.5, field=7.0, parallel=SimpleNamespace(is_parallel=False))
    vpm = SimpleNamespace(strength=3.0)
    transfer = SimpleNamespace(step=4)
    c = SimpleNamespace(
        config=CouplerConfig(
            transfer_inset_spacings=0.0,
            blend_end_depth=0.0,
            max_coupling_iterations=6,
            normal_velocity_tolerance=1e-5,
            normal_gradient_tolerance=1e-5,
        ),
        fvm_solver=fvm,
        vpm_solver=vpm,
        vorticity_transfer=transfer,
        _is_root=True,
        boundary_data_correction=None,
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
    old, raw = boundary_data(0.3), boundary_data(1.0)
    iteration._write_boundary_data(c, old, old=True)
    iteration._write_boundary_data(c, raw)
    inputs, starts, outputs = [], [], []
    fvm.write_step_output = lambda: outputs.append((fvm.step, fvm.time, fvm.field))

    def advance(owner, *args):
        value = float(args[-1][0, 0])
        assert np.array_equal(args[-1], boundary_data(value)[0])
        assert np.array_equal(owner._normal_velocity_boundary_condition, boundary_data(value)[1])
        assert np.array_equal(owner._normal_gradient_boundary_condition, boundary_data(value)[2])
        starts.append((fvm.step, fvm.time, fvm.field, vpm.strength, transfer.step))
        inputs.append(value)
        fvm.step, fvm.time, fvm.field = fvm.step + 1, fvm.time + 0.05, value
        return 0.01

    monkeypatch.setattr(iteration, "advance_fvm", advance)

    def vorticity_transfer(*args):
        vpm.strength += 2.0
        transfer.step += 1
        return transfer.step, 0.01

    c._transfer_vorticity_to_vpm = vorticity_transfer

    def update(owner, *args):
        iteration._write_boundary_data(owner, boundary_data(owner.map_value(inputs[-1])), old=True)

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
        run=lambda: iteration.advance_coupling_iterations(c, geometry, raw[0]),
    )


def test_cold_start_iterates_from_the_vpm_predictor(rig):
    rig.run()
    assert rig.inputs == pytest.approx([1.0, 0.01, 1e-4, 1e-6])
    assert rig.starts == [(10, 0.5, 7.0, 3.0, 4)] * 4
    assert (rig.fvm.step, rig.transfer.step, rig.vpm.strength) == (11, 5, 5.0)
    assert rig.outputs == [(11, 0.55, pytest.approx(1e-6))]
    diagnostics = rig.c._last_coupling_iteration_diagnostics
    assert diagnostics["converged"] and diagnostics["iterations"] == 4
    assert diagnostics["first_estimate"] == "vpm_predictor"
    # The stored correction maps this predictor onto the accepted end-of-interval data.
    correction = rig.c.boundary_data_correction
    assert np.allclose(rig.raw[0] + correction[0], boundary_data(1e-8)[0])


def test_previous_correction_converges_in_one_coupling_iteration(rig):
    rig.c.boundary_data_correction = boundary_data(-1.0)
    rig.run()
    assert rig.inputs == [0.0]
    assert rig.outputs == [(11, 0.55, 0.0)]
    diagnostics = rig.c._last_coupling_iteration_diagnostics
    assert diagnostics["iterations"] == 1 and diagnostics["first_estimate"] == "previous_correction"
    assert np.array_equal(rig.c.boundary_data_correction[0], boundary_data(-1.0)[0])


def test_inexact_first_estimate_continues_from_its_own_iteration(rig):
    rig.c.boundary_data_correction = boundary_data(1.0)
    rig.run()
    assert rig.inputs == pytest.approx([2.0, 0.02, 2e-4, 2e-6])
    assert len(set(rig.starts)) == 1
    assert len(rig.outputs) == 1


def test_nonconverged_iteration_keeps_no_correction(rig):
    rig.c.boundary_data_correction = boundary_data(1.0)
    rig.c.map_value = lambda value: 0.5 * value
    rig.run()
    assert len(rig.inputs) == rig.c.config.max_coupling_iterations
    assert not rig.c._last_coupling_iteration_diagnostics["converged"]
    assert rig.c.boundary_data_correction is None
    assert len(rig.outputs) == 1


def test_new_start_and_backup_load_start_cold(monkeypatch):
    from openonda.coupler import solver as driver

    owner = driver.CoupledSolver.__new__(driver.CoupledSolver)
    owner._started = False
    owner.boundary_data_correction = boundary_data(1.0)

    def stop(time):
        assert owner.boundary_data_correction is None
        raise RuntimeError("stop before any solver work")

    owner.fvm_solver = SimpleNamespace(apply_velocity_boundaries=stop, time=0.0)
    with pytest.raises(RuntimeError, match="before any solver work"):
        owner._start()

    owner.boundary_data_correction = boundary_data(1.0)
    owner.vorticity_transfer = SimpleNamespace(step=0)

    def load(*args, **kwargs):
        assert owner.boundary_data_correction is None
        return 11

    monkeypatch.setattr(driver, "load_coupled_backup", load)
    owner.load_backup("unused-backup")
    assert owner.vorticity_transfer.step == 12


@pytest.mark.integration
def test_two_rank_coupling_iterations_follow_the_root_decision():
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
            str(Path(__file__).with_name("_coupling_iteration_mpi.py")),
        ],
        text=True,
        capture_output=True,
        timeout=90,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"iteration_counts": [1, 4, 4]' in result.stdout
