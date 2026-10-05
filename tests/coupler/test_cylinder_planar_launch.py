"""Ordinary planar cylinder CLI configuration without a solver or external archive."""

from pathlib import Path

import pytest

from openonda import vpm
from openonda.tutorial_runner import load_case_module
from source.solvers.vpm.config.configuration_values import numerical_configuration

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
observations = load_case_module(CASE)


def test_ordinary_builder_selects_native_settings():
    setup = load_case_module(CASE)
    flow, particles, _, mesh = setup.build_case()
    numerics = particles.numerics
    assert type(numerics.induction) is vpm.PlanarInduction
    assert numerics.induction.planar_span == 1.0
    assert numerics.induction.plane_z == 0.0
    assert numerics.compute_device == "AUTO"
    assert numerics.precision == "f32" and numerics.particle_kernel == "GAUSSIAN"
    assert flow.time.end_time == 100.0
    assert particles.run.steps == 2500
    assert mesh.levels == (-0.5, 0.5)
    periodic = {patch.name: patch for patch in flow.boundaries if patch.velocity_type == "cyclic"}
    assert set(periodic) == {"zmin", "zmax"}
    assert periodic["zmin"].neighbour_patch == "zmax"
    assert periodic["zmax"].neighbour_patch == "zmin"


def test_cpu_override_keeps_the_same_planar_operator():
    setup = load_case_module(CASE)
    _, portable, _, _ = setup.build_case(overrides={"compute_device": "CPU"})
    assert type(portable.numerics.induction) is vpm.PlanarInduction
    assert portable.numerics.induction.planar_span == 1.0
    assert portable.numerics.induction.plane_z == 0.0
    assert portable.numerics.compute_device == "CPU"


def test_variant_observations_use_resolved_span_and_both_clocks():
    clocks = {"exchange_dt": 0.08, "fvm_time_step": 0.01}
    flow = observations.fvm_samplers(False, span=0.48, freestream_speed=2.0, diameter=3.0, **clocks)
    particles = observations.vpm_samplers(**clocks)
    sampling = observations.sampling_plan(observations.PROFILES, **clocks)
    assert flow[0].reference_area == pytest.approx(1.44)
    assert flow[0].reference_velocity == 2.0
    assert flow[0].reference_length == 3.0
    span_samples = [sample for sample in flow if sample.name.startswith("span_")]
    assert [sample.name for sample in span_samples] == ["span_middle"]
    assert [sample.start[2] for sample in span_samples] == [0.0]
    assert flow[0].schedule.every_n_steps == 8
    assert particles[0].schedule.interval == 1
    profile = next(sample for sample in particles if sample.file_name == "vpm_transverse_x2")
    assert profile.schedule.interval == 2
    for sampler in flow:
        assert sampler.schedule.every_n_steps % 8 == 0
    assert sampling.fvm_output_steps % 8 == 0


@pytest.mark.parametrize("argv,expected", [([], None), (["--max-coupling-steps", "3"], 3)])
def test_normal_cli_only_limits_execution_not_physics(monkeypatch, argv, expected):
    setup = load_case_module(CASE)
    before = numerical_configuration(setup.build_case()[1].numerics)
    calls = []
    monkeypatch.setattr(setup, "create_solver", lambda **kwargs: calls.append(kwargs) or 281)
    assert setup.main(argv) == 0
    assert calls == [{"max_coupling_steps": expected}]
    assert numerical_configuration(setup.build_case()[1].numerics) == before
    assert setup.build_case()[0].time.end_time == 100.0


@pytest.mark.parametrize(
    "argv",
    [
        ["--max-coupling-steps", "1.5"],
        ["--max-coupling-steps"],
        ["--unknown"],
        ["--compute-device", "CPU"],
    ],
)
def test_invalid_or_unknown_cli_never_runs(monkeypatch, argv):
    setup = load_case_module(CASE)
    calls = []
    monkeypatch.setattr(setup, "create_solver", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit) as error:
        setup.main(argv)
    assert error.value.code == 2
    assert not calls
