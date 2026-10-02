"""Installed campaign support preserves local physical factories and lifecycle."""

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from openonda import cylinder_campaign
import openonda.coupler as coupling
from openonda.cylinder_case import resolve_cylinder_variant
from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def test_native_configuration_matches_selected_reference_physics_and_schedules():
    """The selected ordinary run must match the reference, not the old campaign."""
    setup = load_case_module(CASE)
    reference = load_case_module(CASE / "reference_flow")
    flow, particles, policy, mesh = setup.build_case()
    control, control_mesh = reference.build_case("phase_h004", 0.04)
    for field in ("schemes", "pimple", "linear", "transport", "turbulence", "time"):
        assert asdict(getattr(flow, field)) == asdict(getattr(control, field))
    assert len(mesh.levels) == len(control_mesh.levels) == 25
    assert flow.time.time_step_size == 0.008
    assert particles.numerics.time_step_size == 0.04
    assert particles.numerics.viscous.particle_spacing == 0.04
    assert particles.numerics.compute_device == "AUTO"
    assert policy.backup_interval_steps == 25
    assert flow.samplers[0].schedule.every_n_steps == 5
    assert particles.samplers.samples[0].schedule.interval == 1


def test_resolved_campaign_inputs_are_immutable():
    inputs = resolve_cylinder_variant(
        {"interface_iterations": 4},
        hxy=0.08,
        span=0.96,
        exchange_dt=0.04,
        cores=4,
        particle_limit=400000,
        end_time=0.8,
        fvm_time_step=0.01,
    )
    with pytest.raises(FrozenInstanceError):
        inputs.hxy = 0.1
    with pytest.raises(TypeError):
        inputs.coupler_overrides["interface_iterations"] = 2


@pytest.mark.parametrize(
    "start_from,explicit", [("latest", False), ("initial", False), ("latest", True)]
)
def test_execution_helper_uses_local_factory_and_native_selection(
    tmp_path, monkeypatch, start_from, explicit
):
    calls = []
    flow, policy, mesh = object(), object(), object()
    particles = SimpleNamespace(
        numerics=SimpleNamespace(induction=SimpleNamespace(z_min=-0.48, z_max=0.48))
    )
    restart = tmp_path / "native-backup" if explicit else None
    overrides = {"hxy": 0.08}

    def local_factory(**kwargs):
        calls.append(("build", kwargs))
        return flow, particles, policy, mesh

    class Solver:
        fvm_solver = object()

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, *args):
            calls.append(("exit",))

        def initialize(self):
            calls.append(("initialize",))

        def run(self, **kwargs):
            calls.append(("run", kwargs))
            return 2

    solver = Solver()

    def create(*args, **kwargs):
        assert args == (flow, particles, policy)
        assert kwargs == {"mesh": mesh, "case_dir": tmp_path}
        return solver

    monkeypatch.setattr(coupling, "create_coupler", create)
    monkeypatch.setattr(
        cylinder_campaign,
        "initialize_cylinder_perturbation",
        lambda fvm, span: calls.append(("perturb", fvm, span)),
    )
    assert (
        cylinder_campaign.run_coupled_cylinder(
            local_factory,
            start_from=start_from,
            output_root=tmp_path,
            end_time=0.8,
            restart_from=restart,
            max_coupling_steps=2,
            overrides=overrides,
        )
        == 2
    )
    assert calls[0] == ("build", {"end_time": 0.8, "overrides": overrides})
    assert calls[-1] == ("exit",)
    run = next(item[1] for item in calls if item[0] == "run")
    assert run == {
        "restart_from": restart,
        "start_from": None if explicit else start_from,
        "max_coupling_steps": 2,
        "backup_at_stop": True,
    }
    assert any(item[0] == "initialize" for item in calls) is not explicit
    assert any(item[0] == "perturb" for item in calls) is not explicit
    if not explicit:
        assert next(item for item in calls if item[0] == "perturb")[1:] == (solver.fvm_solver, 0.96)


def test_public_execution_wrapper_preserves_local_factory_and_arguments(tmp_path, monkeypatch):
    setup = load_case_module(CASE)
    captured = {}

    def execute(factory, **kwargs):
        captured.update(factory=factory, **kwargs)
        return 3

    monkeypatch.setattr(setup, "run_coupled_cylinder", execute)
    restart = tmp_path / "backup"
    overrides = {"hxy": 0.08}
    assert (
        setup.create_solver(
            output_root=tmp_path,
            end_time=0.8,
            restart_from=restart,
            max_coupling_steps=3,
            overrides=overrides,
        )
        == 3
    )
    factory = captured.pop("factory")
    assert captured == {
        "start_from": setup.START_FROM,
        "output_root": tmp_path,
        "end_time": 0.8,
        "restart_from": restart,
        "max_coupling_steps": 3,
        "overrides": overrides,
    }
    resolved = tuple(object() for _ in range(4))
    monkeypatch.setattr(setup, "build_case", lambda **kwargs: resolved)
    assert factory(end_time=0.8, overrides=overrides) == resolved
