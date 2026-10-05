"""Boundary query evidence and wall-time attribution without solver stepping."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import boundary, interface_iteration, interface_prediction, reporting, solver
from tests.coupler.test_interface_prediction import rig as rig


@pytest.mark.parametrize(
    "seed, sweeps, fallback", [(None, 3, False), (-1.0, 1, False), (1.0, 4, True)]
)
def test_interface_boundary_refresh_is_separate_from_transfer(
    monkeypatch, rig, seed, sweeps, fallback
):
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(interface_iteration, "perf_counter", lambda: clock.value)

    def timed(function, seconds, *, result=None):
        def call(*args, **kwargs):
            value = function(*args, **kwargs)
            clock.value += seconds
            return value if result is None else result

        return call

    for name, seconds in (
        ("capture_restart_state", 2.0),
        ("_particle_state_snapshot", 0.2),
        ("restore_restart_state", 0.4),
        ("_restore_particle_state", 0.6),
        ("update_boundary_history_after_replacement", 11.0),
    ):
        monkeypatch.setattr(
            interface_iteration, name, timed(getattr(interface_iteration, name), seconds)
        )
    monkeypatch.setattr(
        interface_iteration, "advance_fvm", timed(interface_iteration.advance_fvm, 3.0, result=3.0)
    )
    monkeypatch.setattr(
        interface_prediction,
        "_prediction_inputs",
        timed(interface_prediction._prediction_inputs, 0.1),
    )
    rig.c._transfer_vorticity_to_vpm = timed(rig.c._transfer_vorticity_to_vpm, 7.0)
    rig.fvm.write_accepted_step_output = timed(rig.fvm.write_accepted_step_output, 5.0)
    rig.c.map_value = lambda value: 0.001 * value
    if seed is not None:
        rig.seed_history(seed)
    clock.value = 0.0

    _, fvm_seconds, transfer_seconds = interface_iteration.advance_iterated_interface(
        rig.c, rig.geometry, rig.raw[0]
    )
    diagnostics = rig.c._last_interface_iteration_diagnostics
    phases = diagnostics["phase_seconds"]
    assert diagnostics["sweeps"] == sweeps
    assert diagnostics["converged"]
    assert diagnostics["prediction"]["fallback"] is fallback
    assert fvm_seconds == pytest.approx(3.0 * sweeps + 5.0)
    assert phases["boundary_refresh"] == pytest.approx(11.0 * sweeps)
    assert transfer_seconds == pytest.approx(
        7.0 * sweeps + phases["state_capture"] + phases["state_restore"]
    )
    assert fvm_seconds + transfer_seconds + phases["boundary_refresh"] == pytest.approx(clock.value)


def test_single_sweep_driver_charges_refresh_to_boundary(monkeypatch):
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(solver.time, "perf_counter", lambda: clock.value)

    def spend(seconds, value=None):
        def call(*args, **kwargs):
            clock.value += seconds
            return value

        return call

    geometry = np.zeros((2, 3)), np.ones((2, 3)), np.ones(2)
    trace = np.ones((2, 3))
    rows = []
    owner = SimpleNamespace(
        _comm=None,
        _is_master=True,
        _restart_loaded=True,
        interface_predictor=SimpleNamespace(reset=lambda: None, commit=lambda: None),
        _prepare_run=lambda: (geometry, 1),
        _validate_start_step=lambda *args: 0,
        _validate_step_limit=solver.FVMVPMCoupler._validate_step_limit,
        _advance_vpm=spend(4.0, 4.0),
        _update_freestream=lambda time: None,
        fvm_solver=SimpleNamespace(time=0.0),
        _transfer_vorticity_to_vpm=spend(7.0, (None, 7.0)),
        vpm_time_step_size=0.04,
        vpm_solver=SimpleNamespace(execute_scheduled_samplers=spend(5.0)),
        setup=SimpleNamespace(interface_iterations=1, backup_interval_steps=0),
    )
    monkeypatch.setattr(solver, "write_run_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(solver, "initialize_vpm_boundary_history", lambda *args: None)
    monkeypatch.setattr(solver, "evaluate_vpm_boundary", spend(6.0, (trace, trace, 6.0)))
    monkeypatch.setattr(solver, "advance_fvm", spend(3.0, 3.0))
    monkeypatch.setattr(solver, "update_boundary_history_after_replacement", spend(11.0))
    monkeypatch.setattr(solver, "flush_log", lambda *args: None)
    monkeypatch.setattr(
        solver, "record_step", lambda *args, **kwargs: rows.append((args[3], kwargs))
    )

    assert solver.FVMVPMCoupler.solve(owner) == 1
    times, options = rows[0]
    assert times == (4.0, 17.0, 3.0, 7.0)
    assert options["state_checks_and_sampling_seconds"] == 5.0
    assert sum(times) + options["state_checks_and_sampling_seconds"] == clock.value == 36.0


def test_boundary_evidence_survives_later_queries_and_omits_image_descriptors(monkeypatch):
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(boundary.time, "perf_counter", lambda: clock.value)
    induction = SimpleNamespace(last_tail=None)
    calls = []

    def query(points, normals, **kwargs):
        calls.append(len(points))
        initial = len(calls) == 1
        clock.value += 2.0 if initial else 3.0
        induction.last_tail = {
            "seconds": 1.9 if initial else 2.9,
            "mesh": {
                "field_prepare_seconds": 1.0 if initial else 0.0,
                "field_reused": not initial,
                "finite_images": 514,
                "allimage_specs": object(),
                "finite_evaluation": {
                    "execution_backend": "cpu" if initial else "cupy_cuda",
                    "memory_fallback": "scratch budget exhausted" if initial else None,
                    "query_seconds": 0.6 if initial else 0.4,
                    "source_count": 300000,
                    "correction": {"aabb_skipped_images": object()},
                },
            },
        }
        return np.tile([1.0, 0.0, 0.0], (len(points), 1)), np.zeros_like(points)

    owner = SimpleNamespace(
        _is_master=True,
        fvm_solver=SimpleNamespace(parallel=SimpleNamespace(comm=None)),
        vpm_solver=SimpleNamespace(
            physics=SimpleNamespace(induction=induction),
            particles=SimpleNamespace(n_particles_total=300000),
            compute_velocity_and_tangential_normal_gradient_at_points=query,
        ),
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        fvm_box=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        vpm_particle_spacing=0.1,
        _velocity_boundary_condition_old=None,
        _normal_velocity_boundary_condition_old=None,
        _tangential_gradient_boundary_condition_old=None,
        _last_transfer_result=None,
        vorticity_transfer=None,
        n_fvm_substeps=5,
    )
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = points.copy()
    area = np.ones(2)
    boundary.evaluate_vpm_boundary(owner, points, normals, area)
    boundary.update_boundary_history_after_replacement(owner, points, normals, area)
    boundary.update_boundary_history_after_replacement(owner, points, normals, area)
    induction.last_tail = {"seconds": 0.01, "sampler_targets": 65}

    record = reporting.compute_diagnostics(owner)
    evidence = record["boundary_induction"]
    assert calls == [2, 2, 2]
    assert evidence["initial"]["execution_backend"] == "cpu"
    assert evidence["initial"]["memory_fallback"] == "scratch budget exhausted"
    assert evidence["initial"]["wall_seconds"] == 2.0
    assert evidence["initial"]["field_prepare_seconds"] == 1.0
    assert evidence["refresh"]["calls"] == 2
    assert evidence["refresh"]["execution_backends"] == {"cupy_cuda": 2}
    assert evidence["refresh"]["wall_seconds"] == 6.0
    assert evidence["refresh"]["query_seconds"] == pytest.approx(0.8)
    assert evidence["refresh"]["last"]["target_count"] == 2
    assert record["last_induction_image_call"] == {"seconds": 0.01, "sampler_targets": 65}
    assert len(json.dumps(evidence)) < 2500
    assert "allimage_specs" not in json.dumps(evidence)
    assert "aabb_skipped_images" not in json.dumps(evidence)
