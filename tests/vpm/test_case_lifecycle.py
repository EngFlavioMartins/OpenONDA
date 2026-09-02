"""Public VPM case-construction and framework-run lifecycle tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

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
    assert "treecode_theta" not in inspect.signature(
        vpm.VPMSolver.compute_pressure_gradient_at_points
    ).parameters


def test_induction_configuration_builds_independent_runtime_evaluators() -> None:
    for configured in (
        vpm.DirectInduction(),
        vpm.TreecodeInduction(theta=0.4),
        vpm.FMMInduction(tolerance=1.0e-3),
    ):
        runtime = configured.build()
        assert runtime is not configured
        assert runtime.physics is None
        assert runtime.method == configured.method


def test_numerics_rejects_an_unsupported_treecode_kernel() -> None:
    with pytest.raises(ValueError, match="does not support particle_kernel=SUPER_GAUSSIAN"):
        vpm.Numerics(
            induction=vpm.TreecodeInduction(),
            particle_kernel="SUPER_GAUSSIAN",
        )


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
