"""The inexpensive phase benchmark must have matched clocks and complete data."""

from pathlib import Path

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module

ASSETS = Path(__file__).resolve().parents[2] / "tests/support/cylinder"


def load(name):
    return load_case_module(ASSETS, name)


case = load("phase_benchmark")


def test_phase_driver_defaults_to_supported_backend(tmp_path):
    driver = load("run_phase_benchmark")
    args = driver.parse_args(["coupled", "--root", str(tmp_path), "--pilot"])
    assert args.device == "CPU"
    _, particles, _, _ = case.coupled_case(device=args.device)
    assert args.device in particles.numerics.induction.supported_devices


def test_phase_driver_accepts_qualified_cuda_without_starting_solver(tmp_path):
    driver = load("run_phase_benchmark")
    args = driver.parse_args(["coupled", "--root", str(tmp_path), "--device", "CUDA"])
    assert args.device == "CUDA"
    _, particles, _, _ = case.coupled_case(device=args.device)
    assert args.device in particles.numerics.induction.supported_devices
    assert list(tmp_path.iterdir()) == []


def test_reference_and_coupled_phase_conditions():
    data = case.comparison_settings()
    assert data["h"] == 0.04 and data["span"] == 1.0
    assert data["span_layers"] == data["particle_span_layers"] == 1
    assert data["spanwise_boundary"] == "periodic"
    assert data["interface"]["coupler"]["interface_iterations"] == 6
    assert data["fvm_dt"] == 0.008 and data["exchange_dt"] == 0.04
    assert data["schemes"]["time_scheme"] == "backward"
    assert data["force_phase_interval"] == 0.04
    assert data["startup"] == {
        "duration": 2.0,
        "transition_duration": 1.0,
        "freestream_velocity": [1.0, 0.1, 0.0],
        "steady_freestream_velocity": [1.0, 0.0, 0.0],
    }


def test_corresponding_probe_points_and_cadences():
    reference, _ = case.reference_case()
    _, particles, _, _ = case.coupled_case()
    by_name = {s.name: s for s in reference.samplers}
    for sampler in particles.samplers.samples:
        if "phase_" in sampler.file_name or "transverse_" in sampler.file_name:
            expected = by_name[sampler.file_name.removeprefix("vpm_")]
            np.testing.assert_allclose(sampler.line_points, expected.points, atol=5e-7, rtol=0)
            assert sampler.schedule.interval * case.EXCHANGE == pytest.approx(
                expected.schedule.every_n_steps * case.DT
            )


def test_sampling_and_backups_align_with_both_clocks():
    for dt in (case.DT, case.EXCHANGE):
        for interval in (case.FORCES, case.PROFILES, case.SLICES, case.BACKUPS, case.VOLUMES):
            assert case.steps(interval, dt) * dt == pytest.approx(interval)
    with pytest.raises(ValueError):
        case.steps(0.05, case.EXCHANGE)
