"""Focused checks for the vortex-interaction coverage diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module

CASE_DIRECTORY = Path(__file__).resolve().parents[2] / "tutorials" / "vpm" / "03_vortex_interactions_PENDING"
MATCHED_SMAGORINSKY = 0.24080542149013215


def test_weighted_quantiles_follow_dominant_vorticity_weight():
    coverage = load_case_module(CASE_DIRECTORY, "assets.assess_particle_coverage")
    values = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 8.0, 1.0])
    result = coverage._weighted_quantiles(values, weights, (0.25, 0.5, 0.75))
    assert result[1] == pytest.approx(1.0)
    assert 0.0 < result[0] < result[1] < result[2] < 2.0


def test_zero_strength_sources_leave_exact_host_induction_unchanged():
    check = load_case_module(CASE_DIRECTORY, "assets.check_initial_coverage_field")
    source = {
        "position": np.array([[0.0, 0.0, 0.0], [0.5, 0.2, -0.1]]),
        "vortex_strength": np.array([[0.0, 1.0, 0.0], [0.2, 0.0, -0.1]]),
        "core_radius": np.array([0.1, 0.12]),
    }
    target = np.array([[0.3, 0.4, 0.2], [0.8, -0.1, 0.5]])
    reference_velocity, reference_gradient = check._evaluate_field(target, source)
    augmented = {
        "position": np.vstack((source["position"], [[0.4, 0.0, 0.0]])),
        "vortex_strength": np.vstack((source["vortex_strength"], [[0.0, 0.0, 0.0]])),
        "core_radius": np.append(source["core_radius"], 0.2),
    }
    velocity, gradient = check._evaluate_field(target, augmented)
    np.testing.assert_array_equal(velocity, reference_velocity)
    np.testing.assert_array_equal(gradient, reference_gradient)


def test_fixed_sigma_coverage_contrast_preserves_ring_circulation_and_impulse():
    check = load_case_module(CASE_DIRECTORY, "assets.check_initial_coverage_field")
    coarse_specification = next(
        item for item in check.REPRESENTATIONS if item.name == "h060_sigma060_tail1e4"
    )
    dense_specification = next(
        item for item in check.REPRESENTATIONS if item.name == "h050_sigma060_tail1e4"
    )
    coarse = check._build_pair(coarse_specification)
    dense = check._build_pair(dense_specification)
    for pair in (coarse, dense):
        assert check._scalar_circulation(pair[0], -0.5) == pytest.approx(np.pi)
        assert check._scalar_circulation(pair[1], 0.5) == pytest.approx(np.pi)
    coarse_cloud = check._concatenate(coarse)
    dense_cloud = check._concatenate(dense)
    np.testing.assert_allclose(
        check._linear_impulse(dense_cloud["position"], dense_cloud["vortex_strength"]),
        check._linear_impulse(coarse_cloud["position"], coarse_cloud["vortex_strength"]),
        rtol=2.0e-6,
        atol=1.0e-12,
    )


def test_coverage_qualification_holds_core_physics_and_filter_scale_controls():
    setup = load_case_module(CASE_DIRECTORY, "assets.legacy_les")
    case = setup.build_case(
        "baseline",
        scenario="seeded_breakdown",
        compute_device="METAL",
        steps=400,
        wall_minutes=15.0,
        qualification=True,
        particle_spacing=0.05,
        particle_core_radius=0.06,
        smagorinsky_coefficient=MATCHED_SMAGORINSKY,
        case_name="cs_breakdown_coverage_h05_fixed_sigma",
    )

    assert case.name == "cs_breakdown_coverage_h05_fixed_sigma"
    assert case.run.steps == 400
    assert case.run.wall_time_limit_seconds == pytest.approx(900.0)
    assert case.numerics.compute_device == "METAL"
    assert case.numerics.viscous.particle_spacing == pytest.approx(0.05)
    assert case.numerics.viscous.core_radius_ratio == pytest.approx(1.2)
    assert case.numerics.turbulence.smagorinsky_coefficient == pytest.approx(MATCHED_SMAGORINSKY)

    clouds = [ring.build() for ring in case.initial_conditions]
    assert sum(len(cloud.position) for cloud in clouds) == 27_200
    for ring, cloud in zip(case.initial_conditions, clouds, strict=True):
        assert ring.vortex_core_radius == pytest.approx(0.1)
        assert ring.distribution.spacing == pytest.approx(0.05)
        assert ring.distribution.core_radius_ratio == pytest.approx(1.2)
        np.testing.assert_allclose(cloud.core_radius, 0.06)


def test_filter_pair_preflight_changes_only_smagorinsky_and_matches_primary_arrays():
    from source.solvers.vpm.config.fingerprint import numerical_configuration

    preflight = load_case_module(CASE_DIRECTORY, "assets.check_filter_pair_preflight")
    control = preflight._build_case(preflight.CONTROL_NAME, 0.20)
    molecular = preflight._build_case(preflight.MOLECULAR_NAME, 0.0)
    control_config = numerical_configuration(control.numerics)
    molecular_config = numerical_configuration(molecular.numerics)
    differences = preflight._flatten_differences(control_config, molecular_config)
    assert differences == [
        {
            "path": "turbulence.smagorinsky_coefficient",
            "control": 0.2,
            "molecular": 0.0,
        }
    ]

    control_cloud = preflight._cloud(control)
    molecular_cloud = preflight._cloud(molecular)
    assert set(control_cloud) == set(preflight.PRIMARY_FIELDS)
    for field in preflight.PRIMARY_FIELDS:
        np.testing.assert_array_equal(control_cloud[field], molecular_cloud[field])
    assert len(control_cloud["position"]) == 16_104
    assert control.run.steps == molecular.run.steps == 80
    assert control.run.wall_time_limit_seconds == molecular.run.wall_time_limit_seconds == 300.0


def test_filter_pair_step300_continuations_keep_strict_physics_and_bounded_plans():
    runner = load_case_module(CASE_DIRECTORY, "assets.run_filter_pair_continuation")
    control = runner.build_continuation("control_cs020")
    molecular = runner.build_continuation("molecular_cs000")

    for case in (control, molecular):
        assert case.numerics.compute_device == "CPU"
        assert case.numerics.time_step_size == pytest.approx(0.0075)
        assert case.numerics.viscous.particle_spacing == pytest.approx(0.06)
        assert case.numerics.viscous.core_radius_ratio == pytest.approx(1.0)
        assert case.run.steps == 220
        assert not case.run.initial_samples
        assert case.run.final_backup
    assert control.numerics.turbulence.smagorinsky_coefficient == pytest.approx(0.20)
    assert molecular.numerics.turbulence.smagorinsky_coefficient == pytest.approx(0.0)
    assert control.run.wall_time_limit_seconds == pytest.approx(600.0)
    assert molecular.run.wall_time_limit_seconds == pytest.approx(300.0)


def test_thread_qualification_is_an_unchanged_model_restart_without_initial_samples():
    runner = load_case_module(CASE_DIRECTORY, "assets.run_coverage_thread_qualification")
    case = runner.build_continuation("thread_qualification", steps=10, wall_minutes=5.0)

    assert case.name == "thread_qualification"
    assert case.numerics.compute_device == "CPU"
    assert case.numerics.time_step_size == pytest.approx(0.0075)
    assert case.numerics.viscous.particle_spacing == pytest.approx(0.05)
    assert case.numerics.viscous.core_radius_ratio == pytest.approx(1.2)
    assert case.numerics.turbulence.smagorinsky_coefficient == pytest.approx(MATCHED_SMAGORINSKY)
    assert case.run.steps == 10
    assert case.run.wall_time_limit_seconds == pytest.approx(300.0)
    assert not case.run.initial_samples
    assert case.run.final_backup


def test_continuation_checkpoint_state_uses_declared_step_and_fixed_time_step(tmp_path):
    import h5py

    runner = load_case_module(CASE_DIRECTORY, "assets.run_coverage_thread_qualification")
    checkpoint = tmp_path / "vpm_000050.h5"
    with h5py.File(checkpoint, "w") as handle:
        solver = handle.create_group("solver")
        solver.attrs["step"] = 50
        solver.attrs["time"] = 0.375
        solver.attrs["n_particles_total"] = 27_200

    assert runner._checkpoint_state(checkpoint) == (50, 0.375, 27_200)


def test_thread_parity_comparator_uses_declared_rms_gate_and_exact_samples(tmp_path):
    import h5py

    comparator = load_case_module(CASE_DIRECTORY, "assets.compare_thread_qualification")
    left_solution = tmp_path / "left.h5"
    right_solution = tmp_path / "right.h5"
    for path in (left_solution, right_solution):
        with h5py.File(path, "w") as handle:
            particles = handle.create_group("particles")
            particles["position"] = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            particles["group_id"] = np.array([0], dtype=np.int32)

    left_samples = tmp_path / "left"
    right_samples = tmp_path / "right"
    left_samples.mkdir()
    right_samples.mkdir()
    for directory in (left_samples, right_samples):
        (directory / "flow_integrals.csv").write_text("step,value\n50,1\n")
        (directory / "ring_diagnostics.csv").write_text("step,group\n50,0\n")
        (directory / "core_section_000050.vts").write_bytes(b"core")
        (directory / "cross_section_000050.vts").write_bytes(b"cross")

    result = comparator.compare(left_solution, right_solution, left_samples, right_samples)
    assert result["native_state"]["relative_rms_limit"] == pytest.approx(5.0e-5)
    assert result["native_state"]["fields"]["position"]["declared_scale"] == pytest.approx(
        1.0 / np.sqrt(3.0)
    )
    assert result["passed"]
