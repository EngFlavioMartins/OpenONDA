"""Tutorial-owned configuration preserves its physical factory and run_stages."""

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def test_native_configuration_matches_selected_reference_physics_and_schedules():
    """The selected ordinary run and reference share declared physical inputs."""
    setup = load_case_module(CASE)
    reference = load_case_module(CASE / "reference_flow")
    flow, particles, settings, mesh = setup.build_case()
    control, control_mesh = reference.build_case("phase_h004", 0.04)
    for field in ("schemes", "pimple", "linear", "transport", "turbulence", "time"):
        assert asdict(getattr(flow, field)) == asdict(getattr(control, field))
    assert mesh.levels == control_mesh.levels == (-0.5, 0.5)
    assert particles.numerics.induction.z_max - particles.numerics.induction.z_min == 1.0
    assert settings.interface_iterations == 6
    assert flow.time.time_step_size == 0.008
    assert particles.numerics.time_step_size == 0.04
    assert particles.numerics.viscous.particle_spacing == 0.04
    assert particles.numerics.compute_device == "AUTO"
    assert settings.backup_interval_steps == 25
    assert flow.samplers[0].schedule.every_n_steps == 5
    assert particles.samplers.samples[0].schedule.interval == 1


def test_public_execution_uses_native_run_stages_with_physical_initial_field(tmp_path, monkeypatch):
    setup = load_case_module(CASE)
    captured = {}

    def execute(**kwargs):
        captured.update(kwargs)
        return 3

    def factory(flow, particles, settings, **kwargs):
        captured.update(flow=flow, particles=particles, settings=settings, **kwargs)
        return nullcontext(SimpleNamespace(run=execute))

    monkeypatch.setattr(setup.coupling, "create_coupler", factory)
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
    assert captured["start_from"] == restart
    assert captured["case_dir"] == tmp_path
    assert captured["max_coupling_steps"] == 3
    assert captured["backup_at_stop"]
    assert captured["flow"].time.end_time == 0.8
    assert captured["settings"].freestream.end_time == setup.STARTUP_DURATION
    np.testing.assert_allclose(
        captured["initial_velocity"](np.array([[3.0, 0.0, 0.0]])),
        [setup.STARTUP_FREESTREAM_VELOCITY],
    )
