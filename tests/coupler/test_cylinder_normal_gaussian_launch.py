"""Ordinary cylinder CLI/backend identity without a solver or external archive."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from openonda import vpm
from openonda.tutorial_runner import load_case_module
from source.coupler.backup import config_mapping_digest
from source.solvers.vpm.config.fingerprint import numerical_configuration
from source.solvers.vpm.config.restart import canonical_restart_configuration

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"

# Identity recorded by the native Gaussian continuation at step 281 / 11.24 s.
# Keep this small regression independent of generated/untracked backup files.
QUALIFIED_VPM_SHA256 = "ef1ce0df5bfae7a2a69102cc54b44a65be5eb1f8049f81b8e0a109ab4d71727d"
QUALIFIED_POLICY = {
    "mesh": {"broadening_ratio": 3.0, "spacing_over_tau": 0.25,
             "correction_radius_over_tau": 5.0, "order": 10},
    "tail_contract": "gaussian_interval_remainder_v1", "backend": "cupy_cuda",
    "max_sources": 1_000_000, "max_query_points": 1_000_000,
    "max_scratch_bytes": 2147483648, "max_correction_bytes": 268435456,
    "max_plan_bytes": 134217728, "max_total_bytes": 2415919104,
}


def test_both_authored_constructors_select_exact_native_policy():
    setup = load_case_module(CASE)
    flow, particles, _, mesh = setup.build_case()
    for numerics in (setup.VPM_CASE.numerics, particles.numerics):
        assert type(numerics.induction.gaussian_mesh_policy) is vpm.GaussianSlabPolicy
        assert asdict(numerics.induction.gaussian_mesh_policy) == {**QUALIFIED_POLICY, "backend": "auto"}
        assert numerics.compute_device == "AUTO"
        assert numerics.precision == "f32" and numerics.particle_kernel == "GAUSSIAN"
    current = numerical_configuration(particles.numerics)
    original = deepcopy(current)
    original["induction"]["gaussian_mesh_policy"]["backend"] = "cupy_cuda"
    assert config_mapping_digest(original) == QUALIFIED_VPM_SHA256
    assert canonical_restart_configuration(current) == canonical_restart_configuration(original)
    assert flow.time.end_time == 100.0
    assert particles.run.steps == 2500
    assert len(mesh.levels) == 25


def test_cpu_override_keeps_the_same_gaussian_operator():
    setup = load_case_module(CASE)
    _, portable, _, _ = setup.build_case(overrides={"compute_device": "CPU"})
    assert portable.numerics.induction.gaussian_mesh_policy == vpm.GaussianSlabPolicy()
    assert portable.numerics.compute_device == "CPU"
    _, particles, _, _ = setup.build_case(
        overrides={"compute_device": "CPU"}, gaussian_mesh_policy=None)
    assert particles.numerics.induction.gaussian_mesh_policy is None
    assert particles.numerics.compute_device == "CPU"


def test_variant_observations_use_resolved_span_and_both_clocks():
    setup = load_case_module(CASE)
    observations = load_case_module(CASE, "assets.sampling")
    configured = replace(setup.FVM_SETUP,
                         time=replace(setup.FVM_SETUP.time, time_step_size=.01))
    flow = observations.configure_fvm(configured, False, .8, span=.48, exchange_dt=.08)
    particles = observations.vpm_samplers(end=.8, exchange_dt=.08, fvm_time_step=.01)
    assert flow.time.time_step_size == .01
    assert flow.time.adjustment is None
    assert flow.samplers[0].reference_area == .48
    span_samples = [sample for sample in flow.samplers if sample.name.startswith("span_")]
    assert [sample.start[2] for sample in span_samples] == [-.12, 0., .12]
    assert flow.samplers[0].schedule.every_n_steps == 8
    assert particles[0].schedule.interval == 1
    profile = next(sample for sample in particles if sample.file_name == "vpm_transverse_x2")
    assert profile.schedule.interval == 2
    for sampler in flow.samplers:
        assert sampler.schedule.every_n_steps % 8 == 0
    assert flow.time.output_schedule.every_n_steps % 8 == 0
    with pytest.raises(ValueError, match="physical accepted time"):
        observations.configure_fvm(configured, False, .8, exchange_dt=.075)


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


@pytest.mark.parametrize("argv", [
    ["--max-coupling-steps", "0"], ["--max-coupling-steps", "-1"],
    ["--max-coupling-steps", "1.5"], ["--max-coupling-steps"],
    ["--unknown"], ["--compute-device", "CPU"],
])
def test_invalid_or_unknown_cli_never_runs(monkeypatch, argv):
    setup = load_case_module(CASE)
    calls = []
    monkeypatch.setattr(setup, "create_solver", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit) as error:
        setup.main(argv)
    assert error.value.code == 2
    assert not calls
