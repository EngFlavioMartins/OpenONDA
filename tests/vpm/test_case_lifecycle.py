"""Public VPM case-construction and framework-run lifecycle tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openonda import vpm
from source.solvers.vpm.config.health import HealthError
from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.io.sampler import OutputEvent


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
    assert vpm.FMMInduction.supported_devices == frozenset({"AUTO", "CPU", "VULKAN"})
    for device in ("CUDA", "METAL"):
        with pytest.raises(ValueError, match="does not support compute_device"):
            vpm.Numerics(
                induction=vpm.FMMInduction(),
                compute_device=device,
                verbose=False,
            )


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
        panel_solver=None,
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
    solver._build_initial_conditions = lambda: events.append("build")
    solver._refresh_diagnostics_for_output = lambda: events.append("diagnostics")
    solver.advance = lambda: events.append("advance")
    solver.save_backup = lambda: events.append("backup")
    solver._write_run_manifest = lambda status, failure: events.append((status, failure))
    solver.close = lambda: events.append("close")

    VPMSolver.run(solver)

    assert events == [
        "build",
        "diagnostics",
        OutputEvent.INITIAL,
        "advance",
        "advance",
        "diagnostics",
        OutputEvent.FINAL,
        "backup",
        ("completed", None),
        "close",
    ]


def test_run_plan_can_persist_and_return_from_a_resolution_limit(capsys) -> None:
    events: list[object] = []

    class Manager:
        def dispatch(self, event: OutputEvent) -> None:
            events.append(("dispatch", event))

        def write_all(self, event: OutputEvent) -> None:
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
        "diagnostics",
        ("dispatch", OutputEvent.INITIAL),
        "advance",
        "advance",
        "diagnostics",
        ("write_all", OutputEvent.FINAL),
        "backup",
        ("resolution_lost", solver.run_failure),
        "close",
    ]
    terminal_output = capsys.readouterr().out
    assert "Stopped" in terminal_output
    assert "declared resolution limit" not in terminal_output


def test_run_plan_does_not_persist_an_invalid_state_as_a_resolution_limit() -> None:
    events: list[object] = []

    class Manager:
        def dispatch(self, event: OutputEvent) -> None:
            events.append(("dispatch", event))

        def write_all(self, event: OutputEvent) -> None:
            events.append(("write_all", event))

    solver = object.__new__(VPMSolver)
    solver.case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        run=vpm.RunPlan(steps=1, health_limit_action="STOP"),
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

    with pytest.raises(HealthError, match="non-finite accepted state"):
        VPMSolver.run(solver)

    assert solver.run_status == "failed"
    assert isinstance(solver.run_failure, HealthError)
    assert ("dispatch", OutputEvent.FAILED) in events
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
    clock = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(solver_module, "perf_counter", lambda: next(clock))

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

    solver.advance = advance
    solver.save_backup = lambda: events.append("backup")
    solver._write_run_manifest = lambda status, failure: events.append((status, failure))
    solver.close = lambda: events.append("close")
    VPMSolver.run(solver)
    assert solver.step == 1
    assert solver.run_status == "wall_time_limit"
    assert solver.run_failure is None
    assert events == [
        OutputEvent.INITIAL,
        "terminal_samples",
        "backup",
        ("wall_time_limit", None),
        "close",
    ]
