"""Iteration must restore predictors and publish only the accepted endpoint."""

from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import CouplerSetup
from source.coupler import interface_iteration as iteration


@pytest.mark.parametrize("planar", [False, True])
def test_fixed_predictor_map_does_not_accumulate_time_or_particles(monkeypatch, planar):
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
    if planar:
        vpm.induction = SimpleNamespace(planar_span=1.0)
        vpm.particles = SimpleNamespace(
            n_particles_total=2,
            position_cpu=lambda: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
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
    rows = c._last_interface_iteration_diagnostics["residuals"]
    if planar:
        assert all(row["particle_count"] == 2 for row in rows)
        assert len({row["particle_support_digest"] for row in rows}) == 1
        # For x[k+1] = .2 x[k], the two-sweep distance is six times
        # the one-sweep distance. This distinguishes contraction from cycling.
        for row in rows[1:]:
            assert row["two_sweep_normal_residual_rms"] == pytest.approx(
                6 * row["normal_residual_rms"]
            )
    else:
        assert all("particle_support_digest" not in row for row in rows)


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


def _run_accelerated_scalar_map(monkeypatch, map_value, *, gradient_map=None):
    """Exercise the complete sweep/rollback path with a flux-free trace."""
    cfg = SimpleNamespace(
        interface_iterations=3,
        interface_normal_tolerance=1e-5,
        interface_gradient_tolerance=1e-5,
        interface_acceleration="aitken",
        coupling_patch="numericalBoundary",
    )
    fvm = SimpleNamespace(
        step=10,
        time=0.5,
        field=7.0,
        _last_residuals={"marker": 0},
        last_diagnostics={"marker": 0},
        boundaries=[{"name": "numericalBoundary", "normal_velocity_field": np.array([0.0])}],
        parallel=SimpleNamespace(is_parallel=False),
    )
    outputs = []
    fvm.write_accepted_step_output = lambda: outputs.append((fvm.step, fvm.field))
    vpm = SimpleNamespace(
        strength=3.0,
        physics=SimpleNamespace(
            induction=SimpleNamespace(last_tail={"marker": 0}),
            last_solid_projection={"marker": 0},
        ),
    )
    transfer = SimpleNamespace(
        step=4,
        last_interface_flow={"marker": 0},
        last_vortex_line_closure={"marker": 0},
        last_spanwise_metrics={"marker": 0},
    )
    coupler = SimpleNamespace(
        setup=cfg,
        fvm_solver=fvm,
        vpm_solver=vpm,
        vorticity_transfer=transfer,
        _is_master=True,
        _last_vpm_boundary_condition_flux_diagnostics={"marker": 0},
        _last_fvm_boundary_trace_diagnostics={"marker": 0},
        _step_transfer_stats={"marker": 0},
    )
    face_normal = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    face_area = np.ones(2)

    def trace(value):
        velocity = np.tile([value, 0.0, 0.0], (2, 1))
        normal = np.einsum("ij,ij->i", velocity, face_normal)
        gradient = np.tile([0.0, value, 0.0], (2, 1))
        return velocity, normal, gradient

    iteration._write_trace(coupler, trace(1.0), old=True)
    iteration._write_trace(coupler, trace(1.0))
    monkeypatch.setattr(iteration, "capture_restart_payload", lambda f: (f.step, f.time, f.field))

    def restore(f, snapshot):
        f.step, f.time, f.field = snapshot

    monkeypatch.setattr(iteration, "publish_restart_payload", restore)
    monkeypatch.setattr(iteration, "_particle_state_snapshot", lambda v: v.strength)
    monkeypatch.setattr(
        iteration, "_restore_particle_state", lambda v, p: setattr(v, "strength", p)
    )
    inputs = []

    def advance(c, *args):
        value = float(args[-1][0, 0])
        inputs.append(value)
        assert np.allclose(args[-1] @ face_normal[0], value)
        assert np.allclose(c._normal_velocity_boundary_condition, [value, -value])
        fvm.step += 1
        fvm.time += 0.05
        fvm.field = value
        fvm._last_residuals = {"marker": fvm.step}
        fvm.last_diagnostics = {"marker": fvm.step}
        fvm.boundaries[0]["normal_velocity_field"] = np.array([value])
        c._last_fvm_boundary_trace_diagnostics = {"marker": fvm.step}
        return 0.01

    monkeypatch.setattr(iteration, "advance_fvm", advance)

    def renewal(*args):
        vpm.strength += 1.0
        transfer.step += 1
        transfer.last_interface_flow = {"marker": transfer.step}
        transfer.last_vortex_line_closure = {"marker": transfer.step}
        transfer.last_spanwise_metrics = {"marker": transfer.step}
        coupler._step_transfer_stats = {"marker": transfer.step}
        return transfer.step, 0.01

    coupler._transfer_vorticity_to_vpm = renewal

    def update(c, *args):
        post = list(trace(map_value(inputs[-1])))
        if gradient_map is not None:
            post[2] = np.tile([0.0, gradient_map(inputs[-1]), 0.0], (2, 1))
        iteration._write_trace(c, tuple(post), old=True)
        c._last_vpm_boundary_condition_flux_diagnostics = {"marker": transfer.step}
        vpm.physics.induction.last_tail = {"marker": transfer.step}
        vpm.physics.last_solid_projection = {"marker": transfer.step}

    monkeypatch.setattr(iteration, "update_boundary_history_after_replacement", update)
    result, _, _ = iteration.advance_iterated_interface(
        coupler,
        (np.zeros((2, 3)), face_normal, face_area),
        trace(1.0)[0],
    )
    return coupler, fvm, vpm, transfer, inputs, outputs, result


def test_aitken_contraction_improves_residual_and_preserves_flux(monkeypatch):
    c, fvm, vpm, transfer, inputs, outputs, result = _run_accelerated_scalar_map(
        monkeypatch, lambda value: 0.2 * value
    )
    assert inputs == pytest.approx([1.0, 0.2, 0.0], abs=1e-14)
    rows = c._last_interface_iteration_diagnostics["residuals"]
    assert rows[1]["next_acceleration_alpha"] == pytest.approx(1.25)
    assert rows[2]["normal_residual_rms"] < rows[1]["normal_residual_rms"]
    assert c._last_interface_iteration_diagnostics["converged"]
    assert (fvm.step, vpm.strength, transfer.step, result) == (11, 4.0, 5, 5)
    assert outputs == [(11, pytest.approx(0.0, abs=1e-14))]
    assert np.dot(c._velocity_boundary_condition_old[:, 0], [1.0, -1.0]) == pytest.approx(0.0)


def test_aitken_residual_growth_restores_prior_full_endpoint(monkeypatch):
    def nonlinear_map(value):
        if value > 0.75:
            return 0.5
        if value > 0.45:
            return 0.4
        return 1.0

    c, fvm, vpm, transfer, inputs, outputs, result = _run_accelerated_scalar_map(
        monkeypatch, nonlinear_map
    )
    assert inputs == pytest.approx([1.0, 0.5, 0.375])
    diagnostics = c._last_interface_iteration_diagnostics
    assert diagnostics["sweeps"] == 3
    assert diagnostics["accepted_sweep"] == 2
    assert diagnostics["residuals"][-1]["acceleration_rejected"]
    assert not diagnostics["residuals"][-1]["accepted"]
    assert (fvm.step, fvm.time, fvm.field) == pytest.approx((11, 0.55, 0.5))
    assert fvm._last_residuals == {"marker": 11}
    assert fvm.last_diagnostics == {"marker": 11}
    assert fvm.boundaries[0]["normal_velocity_field"] == pytest.approx([0.5])
    assert (vpm.strength, transfer.step, result) == (4.0, 5, 5)
    assert transfer.last_interface_flow == {"marker": 5}
    assert transfer.last_vortex_line_closure == {"marker": 5}
    assert transfer.last_spanwise_metrics == {"marker": 5}
    assert c._step_transfer_stats == {"marker": 5}
    assert c._last_vpm_boundary_condition_flux_diagnostics == {"marker": 5}
    assert c._last_fvm_boundary_trace_diagnostics == {"marker": 11}
    assert vpm.physics.induction.last_tail == {"marker": 5}
    assert vpm.physics.last_solid_projection == {"marker": 5}
    assert c._velocity_boundary_condition_old[0, 0] == pytest.approx(0.4)
    assert c._normal_velocity_boundary_condition_old == pytest.approx([0.4, -0.4])
    assert outputs == [(11, 0.5)]


def test_aitken_bounds_overrelaxation_and_keeps_trace_constraints():
    face_normal = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    face_area = np.array([2.0, 1.0])

    def trace(value):
        velocity = np.array([[value, 0.0, 0.0], [2.0 * value, 0.0, 0.0]])
        normal = np.einsum("ij,ij->i", velocity, face_normal)
        gradient = np.tile([0.0, value, 0.0], (2, 1))
        return velocity, normal, gradient

    candidate, alpha = iteration._aitken_candidate(
        trace(1.0),
        trace(0.8),
        trace(0.8),
        trace(0.64),
        1.0,
        face_normal,
        face_area,
        1e-5,
        1e-5,
    )
    assert alpha == pytest.approx(1.5)
    assert candidate[0][:, 0] == pytest.approx([0.56, 1.12])
    assert candidate[1] == pytest.approx(np.einsum("ij,ij->i", candidate[0], face_normal))
    assert np.dot(candidate[1], face_area) == pytest.approx(0.0, abs=1e-14)
    assert np.einsum("ij,ij->i", candidate[2], face_normal) == pytest.approx([0.0, 0.0])


def test_aitken_rejects_nonfinite_gradient_even_when_normal_improves(monkeypatch):
    def velocity_map(value):
        if value > 0.75:
            return 0.5
        if value > 0.45:
            return 0.4
        return 0.35

    def gradient_map(value):
        return np.nan if value < 0.45 else velocity_map(value)

    c, fvm, vpm, transfer, _inputs, outputs, result = _run_accelerated_scalar_map(
        monkeypatch, velocity_map, gradient_map=gradient_map
    )
    diagnostics = c._last_interface_iteration_diagnostics
    assert diagnostics["residuals"][-1]["acceleration_rejected"]
    assert diagnostics["accepted_sweep"] == 2
    assert (fvm.step, vpm.strength, transfer.step, result) == (11, 4.0, 5, 5)
    assert outputs == [(11, 0.5)]


@pytest.mark.integration
def test_aitken_rejection_restores_rank_local_fvm_state_under_mpi(tmp_path):
    if find_spec("mpi4py") is None:
        pytest.skip("mpi4py is required")
    bundled_mpiexec = Path(sys.executable).with_name("mpiexec")
    mpiexec = str(bundled_mpiexec) if bundled_mpiexec.is_file() else shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("mpiexec is required")
    script = Path(__file__).with_name("_interface_iteration_mpi.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(
        [mpiexec, "-n", "2", sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "INTERFACE_MPI_ROLLBACK_QUALIFIED" in completed.stdout
