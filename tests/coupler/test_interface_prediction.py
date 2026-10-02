"""A speculative initial guess cannot consume or corrupt the baseline solve."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import interface_iteration as iteration
from source.coupler import interface_prediction as prediction


@pytest.fixture
def rig(monkeypatch):
    cfg = SimpleNamespace(
        coupling_patch="outer",
        interface_iterations=3,
        interface_normal_tolerance=1e-5,
        interface_gradient_tolerance=1e-5,
    )
    fvm = SimpleNamespace(
        step=10,
        time=0.5,
        field=7.0,
        _last_residuals={"marker": 0},
        last_diagnostics={"marker": 0},
        parallel=SimpleNamespace(is_parallel=False),
        boundaries=[{"name": "outer", "normal_velocity_field": np.zeros(2)}],
    )
    vpm = SimpleNamespace(
        strength=3.0,
        step=11,
        time=0.55,
        physics=SimpleNamespace(
            induction=SimpleNamespace(last_tail={"marker": 0}), last_solid_projection={"marker": 0}
        ),
    )
    transfer = SimpleNamespace(step=4, last_interface_flow={"marker": 0})
    c = SimpleNamespace(
        setup=cfg,
        fvm_solver=fvm,
        vpm_solver=vpm,
        vorticity_transfer=transfer,
        _is_master=True,
        interface_predictor=prediction.SafeguardedInterfacePredictor(),
        _step_transfer_stats={"marker": 0},
        dt=0.05,
        substeps=1,
        operator=object(),
    )
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    geometry = np.zeros((2, 3)), normals, np.ones(2)

    def trace(value):
        velocity = np.tile([value, 0.0, 0.0], (2, 1))
        return (
            velocity,
            np.einsum("ij,ij->i", velocity, normals),
            np.tile([0.0, value, 0.0], (2, 1)),
        )

    def identity(owner, geo):
        digest = hashlib.sha256()
        for array in geo:
            digest.update(array.tobytes())
        digest.update(repr((owner.dt, owner.substeps, vars(owner.setup))).encode())
        return digest.hexdigest(), (owner.operator,)

    monkeypatch.setattr(prediction, "_identity", identity)
    monkeypatch.setattr(iteration, "capture_restart_payload", lambda f: (f.step, f.time, f.field))

    def restore(f, snapshot):
        f.step, f.time, f.field = snapshot

    monkeypatch.setattr(iteration, "publish_restart_payload", restore)
    monkeypatch.setattr(iteration, "_particle_state_snapshot", lambda v, **kwargs: v.strength)
    monkeypatch.setattr(
        iteration, "_restore_particle_state", lambda v, p: setattr(v, "strength", p)
    )
    inputs, starts, outputs = [], [], []
    fvm.write_accepted_step_output = lambda: outputs.append((fvm.step, fvm.time, fvm.field))

    def advance(owner, *args):
        value = float(args[-1][0, 0])
        assert np.array_equal(args[-2], old[0]), "The physical old endpoint changed"
        assert np.array_equal(owner._normal_velocity_boundary_condition, trace(value)[1])
        assert np.array_equal(owner._tangential_gradient_boundary_condition, trace(value)[2])
        starts.append(
            (
                fvm.step,
                fvm.time,
                vpm.strength,
                transfer.step,
                dict(fvm._last_residuals),
                dict(transfer.last_interface_flow),
            )
        )
        inputs.append(value)
        fvm.step += 1
        fvm.time += 0.05
        fvm.field = value
        fvm._last_residuals = {"marker": len(inputs)}
        fvm.last_diagnostics = {"marker": len(inputs)}
        fvm.boundaries[0]["normal_velocity_field"] = trace(value)[1]
        return 0.01

    monkeypatch.setattr(iteration, "advance_fvm", advance)

    def renewal(*args):
        vpm.strength += 2.0
        transfer.step += 1
        transfer.last_interface_flow = {"marker": len(inputs)}
        c._step_transfer_stats = {"marker": len(inputs)}
        return transfer.step, 0.01

    c._transfer_vorticity_to_vpm = renewal
    c.map_value = lambda value: 0.01 * value

    def update(owner, *args):
        iteration._write_trace(c, trace(c.map_value(inputs[-1])), old=True)
        vpm.physics.last_solid_projection = {"marker": len(inputs)}
        vpm.physics.induction.last_tail = {"marker": len(inputs)}

    monkeypatch.setattr(iteration, "update_boundary_history_after_replacement", update)
    old = trace(0.3)
    raw = trace(1.0)
    iteration._write_trace(c, old, old=True)
    iteration._write_trace(c, raw)

    def seed_history(correction):
        c.interface_predictor._history = {
            "identity": identity(c, geometry),
            "clock": (fvm.step, fvm.time),
            "endpoint": prediction._copy_trace(old),
            "correction": trace(correction),
        }

    return SimpleNamespace(
        c=c,
        fvm=fvm,
        vpm=vpm,
        transfer=transfer,
        geometry=geometry,
        old=old,
        raw=raw,
        trace=trace,
        inputs=inputs,
        starts=starts,
        outputs=outputs,
        seed_history=seed_history,
    )


def run(rig):
    return iteration.advance_iterated_interface(rig.c, rig.geometry, rig.raw[0])


def test_seed_installs_first_full_trace_and_preserves_physical_old_endpoint(rig):
    rig.seed_history(-1.0)
    run(rig)
    assert rig.inputs == [0.0]
    assert rig.outputs == [(11, 0.55, 0.0)]
    info = rig.c._last_interface_iteration_diagnostics["prediction"]
    assert {key: value for key, value in info.items() if not key.endswith("_seconds")} == {
        "enabled": True,
        "attempted": True,
        "reason": "previous_accepted_correction",
        "accepted": True,
        "fallback": False,
    }
    assert rig.c._last_interface_iteration_diagnostics["picard_sweeps"] == 0
    # Staging is not publication: health/output must complete first.
    assert rig.c.interface_predictor._history is None
    rig.c.interface_predictor.commit()
    history = rig.c.interface_predictor._history
    assert history["clock"] == (11, 0.55)
    # Correction must be against RAW, never against the seeded zero input.
    assert np.array_equal(history["correction"][0], rig.trace(-1.0)[0])
    assert all(
        info[key] >= 0 for key in ("identity_seconds", "snapshot_seconds", "history_stage_seconds")
    )


def test_rejected_seed_restores_full_state_and_full_original_allowance(rig):
    rig.seed_history(1.0)
    run(rig)
    assert rig.inputs == pytest.approx([2.0, 1.0, 0.01, 0.0001])
    assert rig.starts[0] == rig.starts[1]
    assert all(start[:4] == (10, 0.5, 3.0, 4) for start in rig.starts)
    diagnostics = rig.c._last_interface_iteration_diagnostics
    assert diagnostics["sweeps"] == 4 and diagnostics["picard_sweeps"] == 3
    assert diagnostics["residuals"][0]["prediction_rejected"]
    assert not diagnostics["residuals"][0]["accepted"]
    assert [row["picard_sweep"] for row in diagnostics["residuals"]] == [0, 1, 2, 3]
    assert len(rig.outputs) == 1
    assert rig.vpm.strength == 5.0 and rig.transfer.step == 5


def test_rejected_seed_restores_actual_physics_diagnostic_storage(rig, monkeypatch):
    from source.solvers.vpm.physics.engine import PhysicsEngine

    physics = object.__new__(PhysicsEngine)
    physics._init_grid_diffusion()
    physics.last_solid_projection = {"marker": 0}
    physics.induction = SimpleNamespace(last_tail={"marker": 0})
    physics._last_gbd_wall_transfer = {"nested": {"marker": 12}}
    physics._last_gbd_moment_recovery = {"nested": {"marker": 13}}
    rig.vpm.physics = physics
    observations = []
    advance = iteration.advance_fvm
    update = iteration.update_boundary_history_after_replacement

    def observe(owner, *args):
        observations.append(
            (
                physics.last_gbd_wall_transfer["nested"]["marker"],
                physics.last_gbd_moment_recovery["nested"]["marker"],
            )
        )
        return advance(owner, *args)

    def mutate(owner, *args):
        update(owner, *args)
        # In-place nested mutation proves the captured diagnostic values are
        # detached, not merely writable references to a public property copy.
        physics._last_gbd_wall_transfer["nested"]["marker"] = -len(rig.inputs)
        physics._last_gbd_moment_recovery["nested"]["marker"] = -len(rig.inputs)

    monkeypatch.setattr(iteration, "advance_fvm", observe)
    monkeypatch.setattr(iteration, "update_boundary_history_after_replacement", mutate)
    rig.seed_history(1.0)
    run(rig)
    assert observations[:2] == [(12, 13), (12, 13)]
    assert rig.inputs == pytest.approx([2.0, 1.0, 0.01, 0.0001])
    assert rig.c._last_interface_iteration_diagnostics["prediction"]["fallback"]
    assert rig.c._last_interface_iteration_diagnostics["picard_sweeps"] == 3
    assert physics.last_gbd_wall_transfer == {"nested": {"marker": -4}}
    assert physics.last_gbd_moment_recovery == {"nested": {"marker": -4}}
    assert len(rig.outputs) == 1


def test_nonconverged_full_baseline_is_not_cached(rig):
    rig.seed_history(1.0)
    rig.c.map_value = lambda value: 0.5 * value
    run(rig)
    assert rig.inputs == pytest.approx([2.0, 1.0, 0.5, 0.25])
    assert not rig.c._last_interface_iteration_diagnostics["converged"]
    rig.c.interface_predictor.commit()
    assert rig.c.interface_predictor._history is None


@pytest.mark.parametrize(
    "change", ["geometry", "order", "dt", "substeps", "operator", "endpoint", "clock", "tolerance"]
)
def test_changed_identity_or_endpoint_cold_starts(rig, change):
    rig.seed_history(-1.0)
    if change == "geometry":
        rig.geometry[0][0, 0] += 0.1
    elif change == "order":
        rig.geometry[1][:] = rig.geometry[1][::-1]
    elif change == "dt":
        rig.c.dt /= 2
    elif change == "substeps":
        rig.c.substeps += 1
    elif change == "operator":
        rig.c.operator = object()
    elif change == "endpoint":
        rig.c._velocity_boundary_condition_old[0, 1] += 0.01
    elif change == "clock":
        rig.fvm.step += 1
    else:
        rig.c.setup.interface_gradient_tolerance /= 2
    seed, info = rig.c.interface_predictor.prepare(
        rig.c, rig.geometry, iteration._read_trace(rig.c, old=True), rig.raw
    )
    assert seed is None and not info["attempted"]
    assert rig.c.interface_predictor._history is None


def test_reset_and_uncommitted_history_are_cold(rig):
    rig.seed_history(-1.0)
    rig.c.interface_predictor.reset()
    assert rig.c.interface_predictor._history is None
    run(rig)
    assert rig.inputs == pytest.approx([1.0, 0.01, 0.0001])
    assert not rig.c._last_interface_iteration_diagnostics["prediction"]["attempted"]
    rig.c.interface_predictor.commit()
    assert rig.c.interface_predictor._history is None
    rig.seed_history(-1.0)
    rig.c.interface_predictor.reset()
    assert rig.c.interface_predictor._history is None


def test_nonfinite_seed_is_not_probed(rig):
    rig.seed_history(float("nan"))
    run(rig)
    assert rig.inputs == pytest.approx([1.0, 0.01, 0.0001])
    assert not rig.c._last_interface_iteration_diagnostics["prediction"]["attempted"]


def test_mutation_during_trial_prevents_history_publication(rig):
    rig.seed_history(-1.0)

    def changed(value):
        rig.c.operator = object()
        return 0.0

    rig.c.map_value = changed
    run(rig)
    rig.c.interface_predictor.commit()
    assert rig.c.interface_predictor._history is None


@pytest.mark.parametrize(
    "failure_at",
    [
        "initial_capture",
        "seed_algebra",
        "seed_capture",
        "seed_install",
        "seed_restore",
        "trial",
        "stage",
    ],
)
def test_errors_discard_all_history_without_retry_or_error_masking(rig, monkeypatch, failure_at):
    rig.seed_history(1.0 if failure_at == "seed_restore" else -1.0)
    failure = MemoryError("injected local failure")

    def fail(*args, **kwargs):
        raise failure

    if failure_at == "initial_capture":
        monkeypatch.setattr(iteration, "capture_restart_payload", fail)
    elif failure_at == "seed_algebra":
        monkeypatch.setattr(prediction, "_same_arrays", fail)
    elif failure_at == "seed_capture":
        monkeypatch.setattr(iteration, "_capture_trial_fallback", fail)
    elif failure_at == "seed_install":
        monkeypatch.setattr(iteration, "_write_trace", fail)
    elif failure_at == "seed_restore":
        monkeypatch.setattr(iteration, "_restore_trial_fallback", fail)
    elif failure_at == "trial":
        rig.c._transfer_vorticity_to_vpm = fail
    else:
        monkeypatch.setattr(prediction, "_copy_trace", fail)
    with pytest.raises(MemoryError) as raised:
        run(rig)
    assert raised.value is failure
    history = rig.c.interface_predictor
    assert history._history is history._pending is history._active_identity is None
    assert len(rig.inputs) == (1 if failure_at in ("seed_restore", "trial", "stage") else 0)
    history.commit()
    assert history._history is None


@pytest.mark.parametrize("failure_at", ["health", "reporting"])
def test_driver_health_or_output_failure_clears_staged_history(rig, monkeypatch, failure_at):
    from source.coupler import solver as driver

    owner = rig.c
    owner._comm = None
    owner._prepare_run = lambda: (rig.geometry, 11)
    owner._validate_start_step = lambda step, count: step
    owner._validate_step_limit = lambda limit: limit
    owner._advance_vpm = lambda step, time: 0.0
    owner._restart_loaded = True
    owner.map_value = lambda value: 0.001 * value
    owner.vpm_time_step_size = 0.05
    owner.setup.backup_interval_steps = 0
    owner.solution_dir = Path("unused-no-io")
    failure = RuntimeError("injected accepted-output failure")

    def fail(*args, **kwargs):
        assert owner.interface_predictor._pending is not None
        raise failure

    owner.vpm_solver.execute_scheduled_samplers = fail if failure_at == "health" else lambda: None
    monkeypatch.setattr(driver, "write_run_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "initialize_vpm_boundary_history", lambda *args: None)
    monkeypatch.setattr(
        driver, "evaluate_vpm_boundary", lambda *args: (rig.old[0], rig.raw[0], 0.0)
    )
    monkeypatch.setattr(
        driver, "record_step", fail if failure_at == "reporting" else lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError) as raised:
        driver.FVMVPMCoupler.solve(owner, start_step=10)
    assert raised.value is failure
    history = owner.interface_predictor
    assert history._history is history._pending is history._active_identity is None
    history.commit()
    assert history._history is None


def test_new_solve_and_native_load_clear_history_before_any_state_change(rig, monkeypatch):
    from source.coupler import solver as driver

    owner = driver.FVMVPMCoupler.__new__(driver.FVMVPMCoupler)
    owner.interface_predictor = rig.c.interface_predictor
    owner._comm = None
    rig.seed_history(-1.0)

    def prepare():
        assert owner.interface_predictor._history is None
        raise RuntimeError("stop before any solver work")

    owner._prepare_run = prepare
    with pytest.raises(RuntimeError, match="before any solver work"):
        owner.solve()
    rig.seed_history(-1.0)

    def load(*args, **kwargs):
        assert owner.interface_predictor._history is None
        return 11

    monkeypatch.setattr(driver, "load_coupled_backup", load)
    assert owner.load_backup("unused-checkpoint") == 11
    assert owner._restart_loaded


def test_native_identity_hashes_live_values_not_mutable_operator_identity():
    from source.coupler import CouplerSetup
    from source.solvers.fvm.config import FVMSetup
    from source.solvers.vpm.config.case import Numerics
    from source.solvers.vpm.physics.induction.direct import DirectInduction

    setup = Numerics()
    runtime = DirectInduction()  # distinct runtime clone, not setup.induction
    vpm = SimpleNamespace(
        setup=setup,
        induction=runtime,
        physics=SimpleNamespace(),
        stage_rhs=SimpleNamespace(induction=runtime, providers=(), strength_enabled=True),
        time_step_size=0.05,
    )
    mesh = {
        "boundary": [],
        "vertex_position": np.zeros((1, 3)),
        "faces": np.empty((0, 3), dtype=int),
        "owners": np.empty(0, dtype=int),
        "neighbours": np.empty(0, dtype=int),
        "n_cells": 0,
        "n_faces": 0,
        "n_interior_faces": 0,
    }
    fvm = SimpleNamespace(setup=FVMSetup(case_name="identity"), mesh_data=mesh, time_step_size=0.01)
    owner = SimpleNamespace(
        setup=CouplerSetup(),
        fvm_solver=fvm,
        vpm_solver=vpm,
        vorticity_transfer=SimpleNamespace(),
        _is_master=True,
        fvm_time_step_size=0.01,
        vpm_time_step_size=0.05,
        n_fvm_substeps=5,
        vpm_particle_spacing=0.04,
        vpm_diffusion_grid_spacing=0.04,
        vpm_core_radius_ratio=2.5,
        freestream_velocity=np.ones(3),
    )
    geometry = np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), np.ones(1)
    original = prediction._identity(owner, geometry)
    assert original is not None
    assert prediction._same_identity(original, prediction._identity(owner, geometry))
    runtime.stretching_scheme = "DIRECT"
    assert not prediction._same_identity(original, prediction._identity(owner, geometry))
    runtime.stretching_scheme = "TRANSPOSED"
    assert prediction._same_identity(original, prediction._identity(owner, geometry))
    mesh["vertex_position"][0, 0] = 1.0
    assert not prediction._same_identity(original, prediction._identity(owner, geometry))
    mesh["vertex_position"][0, 0] = 0.0
    fvm.time_step_size = 0.005
    assert not prediction._same_identity(original, prediction._identity(owner, geometry))
    fvm.time_step_size = 0.01
    vpm.physics.velocity_override = lambda *_args: None
    assert prediction._identity(owner, geometry) is None
    vpm.physics.velocity_override = None
    vpm.stage_rhs.providers = (object(),)
    assert prediction._identity(owner, geometry) is None


@pytest.mark.integration
def test_two_rank_seed_fallback_and_cold_history_are_collective():
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
            str(Path(__file__).with_name("_interface_prediction_mpi.py")),
        ],
        text=True,
        capture_output=True,
        timeout=90,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"trial_counts": [1, 4, 3]' in result.stdout
    assert '"collective_failure_cases": 13' in result.stdout
