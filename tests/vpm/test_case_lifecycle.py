"""Public VPM case-construction and framework-run lifecycle tests."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openonda import vpm
from source.solvers.vpm.config.fingerprint import numerical_configuration
from source.solvers.vpm.config.health import HealthError
from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.io.manifest import build_manifest
from source.solvers.vpm.io.sampler import OutputEvent


def test_terminal_backup_does_not_repeat_a_scheduled_backup(tmp_path, monkeypatch):
    case = vpm.VPMCase(
        directory=tmp_path,
        numerics=vpm.Numerics(compute_device="CPU", max_n_particles=8, verbose=False),
        run=vpm.RunPlan(steps=2, initial_samples=False, final_backup=True),
        backup=vpm.Backup(interval_steps=1),
    )
    solver = vpm.VPMSolver(case)
    writes = []
    original = solver.io.write_backup

    def record():
        writes.append(solver.step)
        original()

    monkeypatch.setattr(solver.io, "write_backup", record)
    solver.run()
    assert writes == [1, 2]
    assert len(list((tmp_path / "solution" / "vpm").glob("vpm_*.h5"))) == 2


def test_unstable_run_retains_only_earlier_accepted_backup(tmp_path, monkeypatch):
    case = vpm.VPMCase(
        directory=tmp_path,
        numerics=vpm.Numerics(compute_device="CPU", max_n_particles=8, verbose=False),
        run=vpm.RunPlan(
            steps=2, initial_samples=False, final_backup=True, health_limit_action="STOP"
        ),
        backup=vpm.Backup(interval_steps=1),
    )
    solver = vpm.VPMSolver(case)
    original_health = solver._refresh_accepted_step_health

    def reject_second_step() -> None:
        if solver.step == 2:
            raise HealthError("non-finite accepted state", restartable=False)
        original_health()

    monkeypatch.setattr(solver, "_refresh_accepted_step_health", reject_second_step)
    solver.run()

    assert solver.run_status == "unstable"
    assert solver.run_failure is not None
    assert sorted(path.name for path in (tmp_path / "solution" / "vpm").glob("vpm_*.h5")) == [
        "vpm_000001.h5"
    ]
    metadata = json.loads((tmp_path / "solution" / "vpm_metadata.json").read_text())
    assert metadata["lifecycle"]["status"] == "unstable"
    assert metadata["state"]["step"] == 2


def test_induction_configuration_builds_independent_runtime_evaluators() -> None:
    for configured in (
        vpm.DirectInduction(),
        vpm.TreecodeInduction(),
        vpm.FMMInduction(),
    ):
        runtime = configured.build()
        assert runtime is not configured
        assert runtime.physics is None
        assert runtime.method == configured.method


def test_fmm_advertises_only_qualified_device_backends() -> None:
    assert vpm.FMMInduction.supported_devices == frozenset({"AUTO", "CPU", "VULKAN", "METAL"})
    numerics = vpm.Numerics(
        induction=vpm.FMMInduction(),
        compute_device="METAL",
        verbose=False,
    )
    assert numerics.compute_device == "METAL"
    for device in ("CUDA",):
        with pytest.raises(ValueError, match="does not support compute_device"):
            vpm.Numerics(
                induction=vpm.FMMInduction(),
                compute_device=device,
                verbose=False,
            )


def test_runtime_compute_device_override_preserves_restart_identity(tmp_path) -> None:
    """A diagnostic backend override must not rewrite numerical configuration."""
    case = vpm.VPMCase(
        directory=tmp_path,
        numerics=vpm.Numerics(compute_device="AUTO", max_n_particles=8, verbose=False),
        run=vpm.RunPlan(
            steps=0,
            initial_samples=False,
            final_backup=False,
            runtime_compute_device="CPU",
        ),
    )
    solver = vpm.VPMSolver(case)
    try:
        assert solver.compute_device == "CPU"
        assert solver.setup.compute_device == "AUTO"
        assert numerical_configuration(solver.setup)["compute_device"] == "AUTO"
        solver.advance(defer_output=True)
        solver.save_backup()
        checkpoint = tmp_path / "solution" / "vpm" / "vpm_000001.h5"
        assert checkpoint.exists()
        manifest = build_manifest(solver)
        assert manifest["configuration"]["run"]["runtime_compute_device"] == "CPU"
        assert manifest["runtime"] == {
            "configured_compute_device": "AUTO",
            "requested_compute_device": "CPU",
            "effective_compute_device": "CPU",
            "numerical_identity_unchanged": True,
        }
    finally:
        solver.close()
    resumed = vpm.VPMSolver(replace(case, directory=tmp_path / "resumed"))
    try:
        resumed.load_backup(checkpoint)
        assert (resumed.step, resumed.time) == (1, pytest.approx(case.numerics.time_step_size))
        assert resumed.compute_device == "CPU"
        assert resumed.setup.compute_device == "AUTO"
    finally:
        resumed.close()


def test_runtime_manifest_distinguishes_auto_request_from_resolved_backend(tmp_path) -> None:
    """AUTO is a request; provenance must report the initialized backend separately."""
    case = vpm.VPMCase(
        directory=tmp_path,
        numerics=vpm.Numerics(compute_device="AUTO", max_n_particles=8, verbose=False),
        run=vpm.RunPlan(steps=0, runtime_compute_device="AUTO"),
    )
    solver = SimpleNamespace(
        case=case,
        setup=case.numerics,
        compute_device="CPU",
        _backend_name="CPU",
        _runtime_compute_device_override="AUTO",
        step=0,
        time=0.0,
        _run_initial_step=0,
        _run_initial_time=0.0,
        _initial_n_particles_total=0,
    )
    manifest = build_manifest(solver)
    assert manifest["runtime"] == {
        "configured_compute_device": "AUTO",
        "requested_compute_device": "AUTO",
        "effective_compute_device": "CPU",
        "numerical_identity_unchanged": True,
    }


def test_default_runtime_path_has_no_backend_override_provenance(tmp_path) -> None:
    case = vpm.VPMCase(
        directory=tmp_path,
        numerics=vpm.Numerics(compute_device="CPU", max_n_particles=8, verbose=False),
        run=vpm.RunPlan(steps=0),
    )
    solver = SimpleNamespace(
        case=case,
        setup=case.numerics,
        compute_device="CPU",
        _backend_name="CPU",
        _runtime_compute_device_override=None,
        step=0,
        time=0.0,
        _run_initial_step=0,
        _run_initial_time=0.0,
        _initial_n_particles_total=0,
    )
    assert "runtime" not in build_manifest(solver)


def test_numerics_rejects_treecode_double_precision_before_solver_allocation() -> None:
    with pytest.raises(ValueError, match="does not support precision='f64'"):
        vpm.Numerics(induction=vpm.TreecodeInduction(), precision="f64", verbose=False)


def test_requested_vlm_initialization_failure_is_fatal(monkeypatch) -> None:
    from source.solvers.vpm.boundary_elements.vlm.solver import vlm_solver as vlm_module

    class _VLM:
        def __init__(self, _config):
            self._solved = False

    def fail_setup(self):
        raise ValueError("invalid VLM mesh")

    monkeypatch.setattr(vlm_module, "VLMSolver", _VLM)
    solver = object.__new__(VPMSolver)
    solver._stage_providers = []
    solver._setup_vlm_solver = fail_setup.__get__(solver, VPMSolver)
    setup = SimpleNamespace(
        vlm=SimpleNamespace(kinematic_viscosity=0.0),
        viscous=SimpleNamespace(scheme="NONE"),
    )

    with pytest.raises(RuntimeError, match="Failed to initialize VLM solver"):
        solver._init_optional_solvers(setup)


def test_run_owns_the_complete_event_lifecycle() -> None:
    """Run dispatches construction, initial, accepted, final, and cleanup once."""
    events: list[object] = []

    class Manager:
        def dispatch(self, event: OutputEvent) -> None:
            events.append(event)

    solver = object.__new__(VPMSolver)
    solver.case = vpm.VPMCase(numerics=vpm.Numerics(), run=vpm.RunPlan(steps=2))
    solver.output_manager = Manager()
    solver._run_started = False
    solver._run_finished = False
    solver.restart_state = vpm.RestartState()
    solver.time = 0.0
    solver.step = 0
    solver.particles = SimpleNamespace(n_particles_total=0)
    solver._build_initial_conditions = lambda: events.append("build")
    solver._refresh_diagnostics_for_output = lambda: events.append("diagnostics")
    solver.advance = lambda: events.append("advance")
    solver.save_backup = lambda: events.append("backup")
    solver._write_run_manifest = lambda status, failure: events.append((status, failure))
    solver.close = lambda: events.append("close")

    VPMSolver.run(solver)

    assert events == [
        "build",
        ("running", None),
        "diagnostics",
        OutputEvent.INITIAL,
        "advance",
        "advance",
        "diagnostics",
        "backup",
        OutputEvent.FINAL,
        ("completed", None),
        "close",
    ]


def test_run_plan_can_persist_and_return_from_a_resolution_limit(capsys) -> None:
    events: list[object] = []

    class Manager:
        def dispatch(self, event: OutputEvent) -> None:
            events.append(("dispatch", event))

        def write_all(self, event: OutputEvent, *, skip_current=False) -> None:
            events.append(("write_all", event))

    solver = object.__new__(VPMSolver)
    solver.case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        run=vpm.RunPlan(steps=4, health_limit_action="stop"),
    )
    solver.output_manager = Manager()
    solver._run_started = False
    solver._run_finished = False
    solver.restart_state = vpm.RestartState()
    solver.time = 0.0
    solver.step = 0
    solver.particles = SimpleNamespace(n_particles_total=0)
    solver._build_initial_conditions = lambda: events.append("build")
    solver._refresh_diagnostics_for_output = lambda: events.append("diagnostics")

    def advance() -> None:
        events.append("advance")
        solver.step += 1
        solver.time += 0.1
        if solver.step == 2:
            raise HealthError("declared resolution limit")

    solver.advance = advance
    solver.save_backup = lambda: events.append("backup")
    solver._write_run_manifest = lambda status, failure: events.append((status, failure))
    solver.close = lambda: events.append("close")

    VPMSolver.run(solver)

    assert solver.run_status == "resolution_lost"
    assert isinstance(solver.run_failure, HealthError)
    assert solver.step == 2
    assert events == [
        "build",
        ("running", None),
        "diagnostics",
        ("dispatch", OutputEvent.INITIAL),
        "advance",
        "advance",
        "diagnostics",
        "backup",
        ("write_all", OutputEvent.FINAL),
        ("resolution_lost", solver.run_failure),
        "close",
    ]
    terminal_output = capsys.readouterr().out
    assert "Stopped" in terminal_output
    assert "declared resolution limit" in terminal_output


def test_run_plan_can_persist_and_return_from_a_resource_limit() -> None:
    events: list[object] = []

    class Manager:
        def dispatch(self, event: OutputEvent) -> None:
            events.append(("dispatch", event))

        def write_all(self, event: OutputEvent, *, skip_current=False) -> None:
            events.append(("write_all", event))

    solver = object.__new__(VPMSolver)
    solver.case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        run=vpm.RunPlan(
            steps=4,
            health_limit_action="stop",
            resource_limits=vpm.ResourceLimits(max_particles=10),
        ),
    )
    solver.output_manager = Manager()
    solver._run_started = False
    solver._run_finished = False
    solver.restart_state = vpm.RestartState()
    solver.time = 0.0
    solver.step = 0
    solver.particles = SimpleNamespace(n_particles_total=0)
    solver._build_initial_conditions = lambda: events.append("build")
    solver._refresh_diagnostics_for_output = lambda: events.append("diagnostics")

    def advance() -> None:
        events.append("advance")
        solver.step += 1
        solver.time += 0.1
        if solver.step == 2:
            raise vpm.ResourceLimitError("resource limit: test boundary")

    solver.advance = advance
    solver.save_backup = lambda: events.append("backup")
    solver._write_run_manifest = lambda status, failure: events.append((status, failure))
    solver.close = lambda: events.append("close")

    VPMSolver.run(solver)

    assert solver.run_status == "resource_limit"
    assert isinstance(solver.run_failure, vpm.ResourceLimitError)
    assert events[-4:] == [
        "backup",
        ("write_all", OutputEvent.FINAL),
        ("resource_limit", solver.run_failure),
        "close",
    ]


def test_terminal_sampler_failure_retains_backup_and_underlying_health_reason() -> None:
    events = []
    output_error = RuntimeError("terminal sampler failed")

    class Manager:
        def dispatch(self, event):
            events.append(event)

        def write_all(self, event, *, skip_current=False):
            assert "backup" in events
            raise output_error

    solver = object.__new__(VPMSolver)
    solver.case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        run=vpm.RunPlan(steps=2, initial_samples=False, health_limit_action="STOP"),
    )
    solver.output_manager = Manager()
    solver._run_started = False
    solver._run_finished = False
    solver.restart_state = vpm.RestartState()
    solver.time, solver.step = 0.0, 0
    solver.particles = SimpleNamespace(n_particles_total=0)
    solver._build_initial_conditions = lambda: None
    solver._refresh_diagnostics_for_output = lambda: None

    def advance():
        solver.step, solver.time = 1, 0.1
        raise HealthError("vorticity divergence exceeds declared maximum")

    solver.advance = advance
    solver.save_backup = lambda: events.append("backup")
    solver._write_run_manifest = lambda status, failure: events.append(status)
    solver.close = lambda: events.append("close")
    with pytest.raises(RuntimeError, match="terminal sampler failed") as error:
        solver.run()
    assert error.value is output_error
    assert "vorticity divergence" in " ".join(output_error.__notes__)
    assert solver.run_status == "failed"
    assert events.count("backup") == 1
    assert events[-1] == "close"


@pytest.mark.parametrize(
    ("health_limit_action", "expected_status"),
    [("STOP", "unstable"), ("RAISE", "failed")],
)
def test_run_plan_does_not_persist_an_invalid_state_as_a_resolution_limit(
    health_limit_action: str, expected_status: str
) -> None:
    events: list[object] = []

    class Manager:
        def dispatch(self, event: OutputEvent) -> None:
            events.append(("dispatch", event))

        def write_all(self, event: OutputEvent, *, skip_current=False) -> None:
            events.append(("write_all", event))

    solver = object.__new__(VPMSolver)
    solver.case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        run=vpm.RunPlan(steps=1, health_limit_action=health_limit_action),
    )
    solver.output_manager = Manager()
    solver._run_started = False
    solver._run_finished = False
    solver.restart_state = vpm.RestartState()
    solver.time = 0.1
    solver.step = 1
    solver._build_initial_conditions = lambda: events.append("build")
    solver._refresh_diagnostics_for_output = lambda: events.append("diagnostics")
    solver.advance = lambda: (_ for _ in ()).throw(
        HealthError("non-finite accepted state", restartable=False)
    )
    solver.save_backup = lambda: events.append("backup")
    solver._write_run_manifest = lambda status, failure: events.append((status, failure))
    solver.close = lambda: events.append("close")

    if health_limit_action == "RAISE":
        with pytest.raises(HealthError, match="non-finite accepted state"):
            VPMSolver.run(solver)
    else:
        VPMSolver.run(solver)

    assert solver.run_status == expected_status
    assert isinstance(solver.run_failure, HealthError)
    assert (("dispatch", OutputEvent.FAILED) in events) == (health_limit_action == "RAISE")
    assert not any(event == "backup" for event in events)
    assert not any(isinstance(event, tuple) and event[0] == "write_all" for event in events)


def test_run_plan_rejects_an_unknown_health_limit_action() -> None:
    with pytest.raises(ValueError, match="health_limit_action"):
        vpm.RunPlan(steps=1, health_limit_action="continue")


def test_initial_conditions_are_globally_pruned_once_after_assembly() -> None:
    events: list[object] = []

    class InitialCondition:
        def __init__(self, group_id: int) -> None:
            self.group_id = group_id

        def build(self):
            events.append(("build", self.group_id))
            return SimpleNamespace(
                position=np.zeros((1, 3)),
                velocity=np.zeros((1, 3)),
                vortex_strength=np.array([[float(self.group_id + 1), 0.0, 0.0]]),
                core_radius=np.ones(1),
                particle_volume=np.ones(1),
                kinematic_viscosity=np.zeros(1),
                group_id=np.array([self.group_id], dtype=np.int32),
                zone_id=None,
            )

    solver = object.__new__(VPMSolver)
    solver.case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        initial_conditions=(InitialCondition(0), InitialCondition(1)),
        initial_weak_particle_percent=5.0,
    )
    solver._initial_conditions_built = False
    solver.add_vortex_particles = lambda **values: events.append(
        ("add", int(values["group_id"][0]))
    )
    solver.remove_weak_particles = lambda percent: events.append(("prune", percent))

    VPMSolver._build_initial_conditions(solver)

    assert events == [
        ("build", 0),
        ("add", 0),
        ("build", 1),
        ("add", 1),
        ("prune", 5.0),
    ]
    assert solver._initial_conditions_built


@pytest.mark.parametrize("percent", [-1.0, 100.1, np.inf, np.nan])
def test_initial_weak_particle_percent_is_bounded(percent: float) -> None:
    with pytest.raises(ValueError, match="initial_weak_particle_percent"):
        vpm.VPMCase(numerics=vpm.Numerics(), initial_weak_particle_percent=percent)


@pytest.mark.parametrize("percent", [True, "5", None])
def test_initial_weak_particle_percent_must_be_numeric(percent: object) -> None:
    with pytest.raises(TypeError, match="initial_weak_particle_percent"):
        vpm.VPMCase(numerics=vpm.Numerics(), initial_weak_particle_percent=percent)


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), True])
def test_run_plan_rejects_invalid_wall_time_limit(limit):
    with pytest.raises((TypeError, ValueError), match="wall_time_limit"):
        vpm.RunPlan(steps=10, wall_time_limit_seconds=limit)


def test_wall_time_stop_persists_accepted_state_without_reporting_physics_failure(monkeypatch):
    import source.solvers.vpm.core.solver as solver_module

    events = []
    clock = [0.0]
    monkeypatch.setattr(solver_module, "perf_counter", lambda: clock[0])

    class Manager:
        def dispatch(self, event):
            events.append(event)

        def write_all(self, event, *, skip_current=False):
            assert skip_current
            events.append("terminal_samples")

    solver = object.__new__(VPMSolver)
    solver.case = vpm.VPMCase(
        numerics=vpm.Numerics(), run=vpm.RunPlan(steps=10, wall_time_limit_seconds=1.0)
    )
    solver.output_manager = Manager()
    solver._run_started = False
    solver._run_finished = False
    solver.restart_state = vpm.RestartState()
    solver.time, solver.step = 0.0, 0
    solver._build_initial_conditions = lambda: None
    solver._refresh_diagnostics_for_output = lambda: None

    def advance():
        solver.step += 1
        solver.time += 0.1
        clock[0] += 2.0

    solver.advance = advance
    solver.save_backup = lambda: events.append("backup")
    solver._write_run_manifest = lambda status, failure: events.append((status, failure))
    solver.close = lambda: events.append("close")
    VPMSolver.run(solver)
    assert solver.step == 1
    assert solver.run_status == "wall_time_limit"
    assert solver.run_failure is None
    assert events == [
        ("running", None),
        OutputEvent.INITIAL,
        "backup",
        "terminal_samples",
        ("wall_time_limit", None),
        "close",
    ]


def test_run_elapsed_time_includes_output_and_backups(monkeypatch, capsys):
    import source.solvers.vpm.core.solver as solver_module

    clock = [100.0]
    monkeypatch.setattr(solver_module, "perf_counter", lambda: clock[0])

    class Manager:
        def dispatch(self, event):
            clock[0] += 5.0 if event == OutputEvent.ACCEPTED_STEP else 3.0

    solver = object.__new__(VPMSolver)
    solver.case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        run=vpm.RunPlan(steps=2, initial_samples=False),
    )
    solver.output_manager = Manager()
    solver._run_started = False
    solver.restart_state = vpm.RestartState()
    solver.time, solver.step, solver.wall_time = 0.0, 0, 0.0
    solver._build_initial_conditions = lambda: None
    solver._refresh_diagnostics_for_output = lambda: None

    def advance():
        solver.step += 1
        solver.time += 0.1
        solver.wall_time += 2.0
        clock[0] += 2.0
        solver.output_manager.dispatch(OutputEvent.ACCEPTED_STEP)

    def backup():
        clock[0] += 11.0

    solver.advance = advance
    solver.save_backup = backup
    solver._write_run_manifest = lambda status, failure: None
    solver.close = lambda: None
    assert solver.elapsed_wall_time == 0.0
    solver.run()
    assert solver.wall_time == 4.0
    assert solver.elapsed_wall_time == 28.0
    output = capsys.readouterr().out
    assert "Elapsed" in output and "00:00:28.0" in output
    clock[0] += 100.0
    assert solver.elapsed_wall_time == 28.0
