"""Output contracts: numerical and lifecycle contracts."""

from __future__ import annotations

import contextlib
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import openonda.vpm as vpm
from source.solvers.vpm.boundary_elements.vlm.solver.diagnostics import VLMDiagnostics
from source.solvers.vpm.config.artifacts import Backup, Samplers
from source.solvers.vpm.io.sampler import OutputEvent, OutputManager
from source.solvers.vpm.io.sampling import EverySteps, EveryTime, FinalOnly
from source.solvers.vpm.io.solver_io import SolverIO


class _Sample:
    def sample(self, _solver):
        return None


def test_backup_defaults_and_custom_directories():
    assert Backup() == Backup(0, "solution", "solution")
    configured = Backup(25, "results/state", "results/logs")
    assert configured.interval_steps == 25
    assert configured.directory == "results/state"
    assert configured.log_directory == "results/logs"


@pytest.mark.parametrize("interval", [-1, True, 1.5])
def test_backup_rejects_invalid_intervals(interval):
    error = ValueError if interval == -1 else TypeError
    with pytest.raises(error):
        Backup(interval_steps=interval)


def test_samplers_accept_a_tuple_and_own_the_directory():
    first = _Sample()
    second = _Sample()
    configured = Samplers((first, second), "ring/run_a")

    assert configured.samples == (first, second)
    assert configured.directory == "ring/run_a"


@pytest.mark.parametrize("directory", ["", "/absolute", ".", "../escape"])
def test_sampler_directory_must_stay_below_samples(directory):
    with pytest.raises(ValueError):
        Samplers(samples=(_Sample(),), directory=directory)


def test_public_case_owns_backup_and_sampler_construction_objects():
    case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        backup=Backup(interval_steps=10),
        samplers=Samplers(samples=(_Sample(),)),
    )

    assert case.backup.interval_steps == 10
    assert len(case.samplers.samples) == 1
    assert {"backup", "samplers"} <= set(inspect.signature(vpm.VPMCase).parameters)


def _coupled_vlm(**kwargs):
    return vpm.VLMSetup(
        surfaces=(vpm.VLMSurfaceSetup(object()),),
        **kwargs,
    )


def test_coupled_vlm_rejects_an_independent_output_cadence():
    with pytest.raises(ValueError, match="cannot configure logging_interval_steps"):
        vpm.VPMCase(
            numerics=vpm.Numerics(
                vlm=_coupled_vlm(logging_interval_steps=2),
            )
        )


def test_coupled_vlm_rejects_scientific_output_opt_out():
    with pytest.raises(ValueError, match="sample_surface_forces=True"):
        vpm.VPMCase(
            numerics=vpm.Numerics(
                vlm=_coupled_vlm(sample_surface_forces=False),
            )
        )


def test_coupled_vlm_rejects_surface_level_output_opt_out():
    vlm = vpm.VLMSetup(
        surfaces=(vpm.VLMSurfaceSetup(object(), sample_forces=False),),
    )
    with pytest.raises(ValueError, match="cannot opt out"):
        vpm.VPMCase(numerics=vpm.Numerics(vlm=vlm))


def test_coupled_vlm_rejects_vlm_geometry_sampler():
    with pytest.raises(ValueError, match="VLMSampler is standalone-only"):
        vpm.VPMCase(
            numerics=vpm.Numerics(vlm=_coupled_vlm()),
            samplers=Samplers(samples=(vpm.VLMSampler(schedule=EverySteps(1)),)),
        )


def test_coupled_vlm_diagnostics_are_owner_clocked_even_if_standalone_frequency_differs(
    tmp_path, monkeypatch
):
    class _Field:
        def to_numpy(self):
            return np.empty(0)

    calls = []
    monkeypatch.setattr(
        VLMDiagnostics,
        "export_forces_csv",
        staticmethod(lambda *args: calls.append(args)),
    )
    vlm = SimpleNamespace(
        _last_forces={"lift_coefficient": 0.0, "drag_coefficient": 0.0},
        logging_interval_steps=99,
        lattice=SimpleNamespace(n_panels=0, leading_edge_suction_parameter=_Field()),
        compute_total_bound_vortex_strength=lambda: np.zeros(3),
    )
    history = {
        name: []
        for name in (
            "vlm_lift_coefficient",
            "vlm_drag_coefficient",
            "vlm_bound_vortex_strength_y",
            "vlm_wake_vortex_strength_y",
            "vlm_max_leading_edge_suction_parameter",
            "vlm_n_particles_total",
        )
    }

    VLMDiagnostics.record_vlm_diagnostics(
        vlm,
        SimpleNamespace(n_particles_total=0),
        np.empty((0, 3)),
        history,
        step=1,
        time=0.1,
        case_dir=str(tmp_path),
    )

    assert len(calls) == 1
    assert calls[0][7] == 1


@pytest.mark.parametrize("strength_factor", [2.0, float("inf"), np.float32("inf")])
def test_vpm_solver_creates_solution_and_samples_directories_on_startup(tmp_path, strength_factor):
    with contextlib.redirect_stdout(io.StringIO()):
        solver = vpm.VPMSolver(
            vpm.VPMCase(
                name="startup_case",
                directory=tmp_path,
                backup=vpm.Backup(interval_steps=0),
                numerics=vpm.Numerics(
                    time_step_size=0.01,
                    compute_device="CPU",
                    precision="f32",
                    max_n_particles=16,
                    max_evaluation_points=16,
                    induction=vpm.DirectInduction(),
                    viscous=vpm.ViscousConfig.inviscid(),
                    stabilization=vpm.StabilizationConfig(
                        filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                            interval_steps=1,
                            max_vortex_strength_factor=strength_factor,
                            max_absolute_vortex_strength=1.0,
                        ),
                    ),
                    verbose=False,
                ),
            )
        )
    try:
        assert (tmp_path / "solution").is_dir()
        assert (tmp_path / "samples").is_dir()
        assert (tmp_path / "solution" / "vpm.log").is_file()
        metadata_path = tmp_path / "solution" / "vpm_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["schema_version"] == 1
        assert metadata["solver"] == "VPM"
        assert metadata["case_name"] == "startup_case"
        assert metadata["lifecycle"] == {"status": "created"}
        assert metadata["state"]["step"] == 0
        refinement = metadata["configuration"]["numerics"]["stabilization"]["filament_refinement"]
        assert float(refinement["max_vortex_strength_factor"]) == strength_factor
        json.dumps(metadata, allow_nan=False)
        serialized = json.dumps(metadata).lower()
        assert "reason" not in serialized
        assert "failure" not in serialized
    finally:
        solver.close()


@pytest.mark.parametrize("requested_steps,status", [(1, "completed"), (3, "partial")])
def test_close_records_state_after_direct_advancement(tmp_path, requested_steps, status):
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            name="manual_advance",
            directory=tmp_path,
            numerics=vpm.Numerics(
                compute_device="CPU",
                time_step_size=0.01,
                max_n_particles=16,
                max_evaluation_points=16,
                induction=vpm.DirectInduction(),
                viscous=vpm.ViscousConfig.inviscid(),
            ),
            run=vpm.RunPlan(steps=requested_steps),
        )
    )
    try:
        solver.advance()
    finally:
        solver.close()
    path = tmp_path / "solution/vpm_metadata.json"
    metadata = json.loads(path.read_text())
    assert metadata["state"]["step"] == 1
    assert metadata["state"]["time"] == pytest.approx(0.01)
    assert metadata["lifecycle"]["status"] == status
    written_at = path.stat().st_mtime_ns
    solver.close()
    assert path.stat().st_mtime_ns == written_at


@pytest.mark.parametrize("managed", [True, False])
def test_native_metadata_tracks_checkpoint_progress_before_completion(tmp_path, managed):
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            backup=Backup(interval_steps=2),
            numerics=vpm.Numerics(
                compute_device="CPU",
                time_step_size=0.01,
                max_n_particles=16,
                max_evaluation_points=16,
                induction=vpm.DirectInduction(),
                viscous=vpm.ViscousConfig.inviscid(),
            ),
            run=vpm.RunPlan(steps=3, initial_samples=False, final_backup=False),
        )
    )
    path = tmp_path / "solution/vpm_metadata.json"
    observed = []
    advance = solver.advance

    def advance_and_observe():
        before = json.loads(path.read_text())
        advance()
        after = json.loads(path.read_text())
        observed.append((before, after))

    solver.advance = advance_and_observe
    try:
        if managed:
            solver.run()
        else:
            for _ in range(3):
                solver.advance()
            solver.save_backup()
            saved = json.loads(path.read_text())
            assert saved["state"]["step"] == 3
            assert saved["lifecycle"]["status"] == "partial"
    finally:
        solver.close()

    initial_status = "running" if managed else "created"
    assert observed[0][0]["lifecycle"]["status"] == initial_status
    assert observed[0][1]["state"]["step"] == 0
    assert (tmp_path / "solution/vpm/vpm_000002.h5").is_file()
    for _, after in observed[1:]:
        assert after["state"]["step"] == 2
        assert after["state"]["time"] == pytest.approx(0.02)
        assert after["lifecycle"]["status"] == ("running" if managed else "partial")
    terminal = json.loads(path.read_text())
    assert terminal["state"]["step"] == 3
    assert terminal["lifecycle"]["status"] == "completed"


class _TableSampler:
    file_name = "table"
    schedule = EverySteps(1)
    include_derivatives = False

    def sample(self, _solver):
        return {
            name: np.asarray([1.0])
            for name in (
                "position_x",
                "position_y",
                "position_z",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "vorticity_x",
                "vorticity_y",
                "vorticity_z",
            )
        }


class _VtkSampler:
    file_name = "surface"
    schedule = EverySteps(1)

    def save_vtp(self, _solver, path: Path, time: float | None = None) -> None:
        del time
        path.write_text("vts", encoding="utf-8")


def _solver(tmp_path, samplers: Samplers):
    return SimpleNamespace(
        case_dir=tmp_path,
        case=SimpleNamespace(backup=Backup(), samplers=samplers),
        step=1,
        time=0.1,
        time_step_size=0.1,
        backups=0,
        _restart_loaded=False,
    )


def test_every_time_dispatches_after_crossing_a_physical_cadence():
    schedule = EveryTime(0.25)

    assert not schedule.is_due(2, 0.2, 0.1)
    assert schedule.is_due(3, 0.3, 0.1)


def test_final_only_is_not_an_accepted_step_schedule():
    schedule = FinalOnly()

    assert schedule.is_final_only
    assert not schedule.is_due(10, 1.0, 0.1)


def test_csv_series_is_atomic_and_appended_within_one_run(tmp_path):
    solver = _solver(tmp_path, Samplers(samples=(_TableSampler(),)))
    manager = OutputManager(solver)

    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step, solver.time = 2, 0.2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    assert len((tmp_path / "samples" / "table.csv").read_text().splitlines()) == 3


def test_pvd_series_is_atomic_and_appended_within_one_run(tmp_path):
    solver = _solver(tmp_path, Samplers(samples=(_VtkSampler(),)))
    manager = OutputManager(solver)

    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step, solver.time = 2, 0.2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    assert (tmp_path / "samples" / "surface.pvd").read_text().count("<DataSet") == 2


def test_resume_rejects_nonmonotonic_csv_event(tmp_path):
    solver = _solver(tmp_path, Samplers(samples=(_TableSampler(),)))
    manager = OutputManager(solver)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    with pytest.raises(RuntimeError, match="duplicate or nonmonotonic"):
        manager.dispatch(OutputEvent.ACCEPTED_STEP)


def test_restart_rewinds_post_checkpoint_sampler_history(tmp_path):
    samplers = Samplers(samples=(_TableSampler(), _VtkSampler()))
    solver = _solver(tmp_path, samplers)
    manager = OutputManager(solver)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step, solver.time = 2, 0.2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    manager.rewind_histories(0.1)

    csv_path = tmp_path / "samples/table.csv"
    assert csv_path.read_text().splitlines()[-1].startswith("0.1,1,")
    pvd_path = tmp_path / "samples/surface.pvd"
    assert pvd_path.read_text().count("<DataSet") == 1
    assert list((tmp_path / "samples/restart-branches").glob("before-*/surface.pvd"))

    solver.step, solver.time = 2, 0.2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    assert csv_path.read_text().splitlines()[-1].startswith("0.2,2,")
    assert pvd_path.read_text().count("<DataSet") == 2


def test_fresh_run_replaces_stale_csv_while_restart_appends(tmp_path):
    sampler = _TableSampler()
    original = _solver(tmp_path, Samplers(samples=(sampler,)))
    OutputManager(original).dispatch(OutputEvent.ACCEPTED_STEP)

    fresh = _solver(tmp_path, Samplers(samples=(sampler,)))
    fresh.step, fresh.time = 2, 0.2
    OutputManager(fresh).dispatch(OutputEvent.ACCEPTED_STEP)
    path = tmp_path / "samples" / "table.csv"
    fresh_rows = path.read_text().splitlines()
    assert len(fresh_rows) == 2
    assert fresh_rows[-1].startswith("0.2,2,")

    resumed = _solver(tmp_path, Samplers(samples=(sampler,)))
    resumed._restart_loaded = True
    resumed.step, resumed.time = 3, 0.3
    OutputManager(resumed).dispatch(OutputEvent.ACCEPTED_STEP)
    resumed_rows = path.read_text().splitlines()
    assert len(resumed_rows) == 3
    assert resumed_rows[-1].startswith("0.3,3,")


def test_output_manager_is_the_only_backup_cadence_owner(tmp_path):
    solver = _solver(tmp_path, Samplers())
    solver.case.backup = Backup(interval_steps=2)
    solver._write_backup = lambda: setattr(solver, "backups", solver.backups + 1)
    manager = OutputManager(solver)

    solver.step = 1
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step = 2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    assert solver.backups == 1


def test_budget_terminal_sampling_keeps_due_rows_and_adds_missing_current_rows(tmp_path):
    sampler = _TableSampler()
    solver = _solver(tmp_path, Samplers(samples=(sampler,)))
    manager = OutputManager(solver)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    manager.write_all(OutputEvent.FINAL, skip_current=True)
    path = tmp_path / "samples/table.csv"
    assert len(path.read_text().splitlines()) == 2
    solver.step, solver.time = 2, 0.2
    manager.write_all(OutputEvent.FINAL, skip_current=True)
    assert len(path.read_text().splitlines()) == 3


@pytest.mark.parametrize("terminal_step", [1540, 1547])
def test_ring_periodic_and_final_schedules_write_one_terminal_event(tmp_path, terminal_step):
    periodic = vpm.RingDiagnosticsSampler(schedule=EverySteps(10))
    final = vpm.RingDiagnosticsSampler(schedule=FinalOnly())
    solver = _solver(tmp_path, Samplers(samples=(periodic, final)))
    solver.particle_position = np.array(
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]
    )
    solver.particle_vortex_strength = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] * 2)
    solver.particle_group_id = np.array([0, 0, 1, 1])
    solver.time_step_size = 0.00375
    solver.step, solver.time = 1540, 1540 * solver.time_step_size
    manager = OutputManager(solver)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step, solver.time = terminal_step, terminal_step * solver.time_step_size
    # Both ordinary completion and an off-cadence health/wall-time stop.
    if terminal_step == 1540:
        manager.dispatch(OutputEvent.FINAL)
    else:
        manager.write_all(OutputEvent.FINAL, skip_current=True)
    path = tmp_path / "samples/ring_diagnostics.csv"
    contents = path.read_bytes()
    assert len(contents.splitlines()) == (3 if terminal_step == 1540 else 5)
    manager.write_all(OutputEvent.FINAL, skip_current=True)
    assert path.read_bytes() == contents
    # Replaying the case without loading a checkpoint starts a fresh output
    # stream and replaces stale rows from the prior run.
    OutputManager(solver).dispatch(OutputEvent.FINAL)
    assert len(path.read_text().splitlines()) == 3


class _ExecutionSample:
    file_name = "probe"

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def save_csv(self, _solver_17, path, *, time: float) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error
        path.write_text(f"time={time}\n", encoding="utf-8")


def _solver_17(tmp_path, samples: tuple[object, ...]):
    samplers = Samplers(samples=samples)
    return SimpleNamespace(
        case_dir=tmp_path,
        case=SimpleNamespace(
            samplers=samplers,
            backup=SimpleNamespace(interval_steps=0),
        ),
        setup=SimpleNamespace(samplers=samplers),
        step=4,
        time=0.2,
        time_step_size=0.05,
        particles=SimpleNamespace(n_particles_total=0),
        particle_vortex_strength=np.empty((0, 3)),
    )


def test_empty_particle_state_still_executes_configured_sampler(tmp_path):
    sample = _ExecutionSample()

    OutputManager(_solver_17(tmp_path, (sample,))).write_all()

    assert sample.calls == 1


def test_sampler_failure_is_fatal_by_default(tmp_path):
    sample = _ExecutionSample(error=OSError("disk full"))

    with pytest.raises(RuntimeError, match="Sampler 'probe' failed at step 4") as error:
        OutputManager(_solver_17(tmp_path, (sample,))).write_all()

    assert isinstance(error.value.__cause__, OSError)


def test_sampler_failure_is_reported(tmp_path):
    sample = _ExecutionSample(error=OSError("disk full"))
    with pytest.raises(RuntimeError, match="disk full"):
        OutputManager(_solver_17(tmp_path, (sample,))).write_all()
    assert sample.calls == 1


def test_individual_sampler_can_declare_an_unmet_prerequisite(tmp_path):
    class _PrerequisiteSample(_ExecutionSample):
        def is_applicable(self, _solver_17) -> bool:
            return False

    sample = _PrerequisiteSample()

    OutputManager(_solver_17(tmp_path, (sample,))).write_all()

    assert sample.calls == 0


def test_scheduled_backup_does_not_dispatch_scientific_output(tmp_path, monkeypatch):
    calls: list[str] = []
    solver = SimpleNamespace(
        _backup_path=tmp_path / "solution",
        setup=SimpleNamespace(backup=SimpleNamespace(interval_steps=2)),
        step=2,
        time=0.1,
    )
    io = SolverIO(solver)

    monkeypatch.setattr(
        "source.solvers.vpm.io.solver_io._BackupIO.save",
        lambda *_args, **_kwargs: calls.append("backup"),
    )
    monkeypatch.setattr(io, "export_state", lambda *_args, **_kwargs: calls.append("state"))
    monkeypatch.setattr(io, "_export_panel_loads", lambda *_args: calls.append("panel"))

    io.write_backup()

    assert calls == ["backup"]


TIME_STEP_SIZE = 0.006


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval": 0},
        {"interval": 20, "first_step": 0},
        {"interval": 20, "start_time": -1.0},
    ],
)
def test_invalid_schedules_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EverySteps(**kwargs)


def test_step_schedule_obeys_cadence_and_both_offsets():
    # Independent predicate over a range of cadences and aligned/unaligned
    # offsets covers the combinations formerly split across five examples.
    dt = 0.006
    for interval in (1, 7, 20):
        for first in (1, 9, 20):
            for start in (0.0, 0.5, 1.5):
                schedule = EverySteps(interval=interval, first_step=first, start_time=start)
                for step in range(501):
                    expected = step >= first and step % interval == 0 and step * dt >= start
                    assert schedule.is_due(step, step * dt, dt) == expected


def test_native_integrals_sample_both_bound_and_wake_moments(tmp_path):
    from _flat_plate_geometry import create_flat_plate
    import pandas as pd

    geometry = create_flat_plate(
        chord=1.0,
        span=4.0,
        angle_of_attack_degrees=5.0,
        n_chordwise_panels=2,
        n_spanwise_panels=2,
    )
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            numerics=vpm.Numerics(
                compute_device="CPU",
                precision="f64",
                time_step_size=0.01,
                max_n_particles=128,
                max_evaluation_points=128,
                induction=vpm.DirectInduction(),
                viscous=vpm.ViscousConfig.inviscid(),
                freestream_velocity=[10.0, 0.0, 0.0],
                vlm=vpm.VLMSetup(
                    surfaces=(vpm.VLMSurfaceSetup(geometry),),
                    density=1.7,
                    kinematic_viscosity=0.0,
                    dtype="f64",
                ),
            ),
            samplers=Samplers((vpm.FlowIntegralsSampler(schedule=EverySteps(1)),), "plate"),
            run=vpm.RunPlan(steps=2, final_backup=False),
        )
    )
    try:
        solver.advance()
        solver.advance()
        row = pd.read_csv(tmp_path / "samples/plate/flow_integrals.csv").iloc[-1]
        wake = row[[f"linear_impulse_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
        bound = row[[f"bound_linear_impulse_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
        total = row[[f"coupled_linear_impulse_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
        np.testing.assert_allclose(wake, solver.total_linear_impulse, atol=1e-12)
        np.testing.assert_allclose(
            bound, solver.vlm_solver.compute_bound_linear_impulse(), atol=1e-12
        )
        np.testing.assert_allclose(total, wake + bound, atol=1e-12)
        # The sampler's moments stay per density, even for a non-unit fluid density.
        assert np.linalg.norm(bound) > 0.01
        for axis in "xyz":
            assert row[f"coupled_vortex_strength_{axis}"] == pytest.approx(
                row[f"net_vortex_strength_{axis}"] + row[f"bound_vortex_strength_{axis}"], abs=1e-12
            )
        assert not list((tmp_path / "solution").glob("*.h5"))
    finally:
        solver.close()


@pytest.mark.parametrize(
    "schedule", [EverySteps(3, first_step=8, start_time=1.1), EveryTime(0.4, start_time=0.7)]
)
def test_native_metadata_retains_the_complete_sample_schedule(schedule):
    from dataclasses import fields

    from source.solvers.vpm.io.manifest import _sampler_identity

    record = json.loads(
        json.dumps(_sampler_identity(SimpleNamespace(schedule=schedule)), allow_nan=False)
    )["schedule"]
    rebuilt = type(schedule)(**{field.name: record[field.name] for field in fields(schedule)})
    expected = [step for step in range(1, 41) if schedule.is_due(step, step * 0.1, 0.1)]
    restored = [step for step in range(1, 41) if rebuilt.is_due(step, step * 0.1, 0.1)]
    assert expected and restored == expected


@pytest.mark.parametrize(
    "schedule", [EverySteps(1, start_time=3 * 0.1), EveryTime(0.1, start_time=3 * 0.1)]
)
def test_sampling_start_matches_the_accepted_clock_with_roundoff(schedule):
    # The solver rounds accepted clocks; authored arithmetic can differ by one ULP.
    assert schedule.is_due(3, 0.3, 0.1)
    assert not schedule.is_due(2, 0.2, 0.1)
    assert not schedule.is_due(3, 0.3 - 1e-8, 0.1)
