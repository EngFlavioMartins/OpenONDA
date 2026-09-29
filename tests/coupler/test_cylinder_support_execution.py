"""Installed campaign support preserves local physical factories and lifecycle."""

from dataclasses import FrozenInstanceError, asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openonda import cylinder_campaign
import openonda.coupler as coupling
from openonda.cylinder_case import resolve_cylinder_variant
from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
FIXTURE = Path(__file__).with_name("fixtures") / "cylinder_factory_configuration.json"


def test_native_configuration_matches_pre_refactor_physics_and_schedules():
    """The fixture is a configuration snapshot, not numerical solution evidence."""
    setup = load_case_module(CASE)
    spec = importlib.util.spec_from_file_location(
        "support_campaign", CASE / "assets/run_campaign.py"
    )
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    for expected in json.loads(FIXTURE.read_text()):
        flow, particles, policy, _ = setup.build_case(end_time=0.8, overrides=expected["overrides"])
        actual = {
            "overrides": expected["overrides"],
            "resolved": campaign._resolved_coupled_config(setup, 0.8, expected["overrides"]),
            "fvm_schedules": [asdict(sample.schedule) for sample in flow.samplers],
            "vpm_schedules": [asdict(sample.schedule) for sample in particles.samplers.samples],
            "output_schedule": asdict(flow.time.output_schedule),
            "backup_interval_steps": policy.backup_interval_steps,
        }
        # Provenance changes with a refactor; physical configuration must not.
        actual["resolved"].pop("source_hash")
        actual["resolved"].pop("software_fingerprint")
        assert json.loads(json.dumps(actual)) == expected


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
    assert captured == {
        "factory": setup.build_case,
        "start_from": setup.START_FROM,
        "output_root": tmp_path,
        "end_time": 0.8,
        "restart_from": restart,
        "max_coupling_steps": 3,
        "overrides": overrides,
    }
