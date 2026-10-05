"""Tutorial-owned configuration preserves its physical factory and lifecycle."""

from dataclasses import asdict
from pathlib import Path

from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def test_native_configuration_matches_selected_reference_physics_and_schedules():
    """The selected ordinary run and reference share declared physical inputs."""
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
        "startup_duration": setup.STARTUP_DURATION,
        "startup_transition_duration": setup.STARTUP_TRANSITION_DURATION,
        "steady_freestream_velocity": setup.FREESTREAM_VELOCITY,
        "perturbation": setup.INITIAL_PERTURBATION,
    }
    resolved = tuple(object() for _ in range(4))
    monkeypatch.setattr(setup, "build_case", lambda **kwargs: resolved)
    assert factory(end_time=0.8, overrides=overrides) == resolved
