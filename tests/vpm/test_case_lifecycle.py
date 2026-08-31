"""Public VPM case-construction and framework-run lifecycle tests."""

from __future__ import annotations

import inspect

from openonda import vpm
from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.io.sampler import OutputEvent


def test_public_solver_requires_one_case_construction_object() -> None:
    """The public solver accepts only the immutable case construction path."""
    signature = inspect.signature(vpm.VPMSolver)
    assert tuple(signature.parameters) == ("case",)
    assert not hasattr(vpm, "VPMSetup")
    assert not hasattr(vpm, "create_vpm_solver")
    assert "time" not in inspect.signature(vpm.Numerics).parameters
    assert "step" not in inspect.signature(vpm.Numerics).parameters


def test_public_namespace_hides_internal_runtime_services() -> None:
    """Only construction/value objects are exposed from ``openonda.vpm``."""
    assert set(vpm.__all__).isdisjoint(
        {
            "BodyPose",
            "OutputManager",
            "SamplerExecutor",
            "SamplingSchedule",
            "SolverIO",
            "VPMSetup",
            "VLMLoadingDistribution",
        }
    )
    for name in ("SamplingSchedule", "VPMSetup"):
        assert not hasattr(vpm, name)


def test_public_solver_methods_do_not_accept_untyped_keyword_forwarding() -> None:
    """Construction and output APIs stay explicit and inspectable."""
    for method_name in ("export_state", "set_velocity_override"):
        signature = inspect.signature(vpm.VPMSolver.__dict__[method_name])
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for name, parameter in signature.parameters.items()
            if name != "self"
        )
        assert signature.return_annotation is not inspect.Signature.empty


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
