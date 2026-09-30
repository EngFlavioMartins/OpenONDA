"""The inexpensive phase benchmark must have matched clocks and complete data."""
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ASSETS = Path(__file__).resolve().parents[2]/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ASSETS/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


case = load("phase_benchmark")
audit = load("check_phase_samples")


def test_phase_driver_defaults_to_supported_backend(tmp_path):
    driver = load("run_phase_benchmark")
    args = driver.parse_args(["coupled", "--root", str(tmp_path), "--pilot"])
    assert args.device == "CPU"
    _, particles, _, _ = case.coupled_case(device=args.device)
    assert args.device in particles.numerics.induction.supported_devices


def test_phase_driver_rejects_unsupported_cuda_before_startup(tmp_path):
    driver = load("run_phase_benchmark")
    with pytest.raises(SystemExit) as exc:
        driver.parse_args(["coupled", "--root", str(tmp_path), "--device", "CUDA"])
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_queue_explicit_backend_agrees_with_driver(tmp_path):
    queue = load("queue_phase_benchmark")
    driver = load("run_phase_benchmark")
    command = queue.benchmark_command(ASSETS, "coupled", tmp_path, "--pilot")
    args = driver.parse_args(command[2:])
    assert args.device == "CPU" and args.pilot


def test_reference_and_coupled_phase_contract():
    data = case.contract()
    assert data["h"] == .04 and data["span_layers"] == 24
    assert data["fvm_dt"] == .008 and data["exchange_dt"] == .04
    assert data["schemes"]["time_scheme"] == "backward"
    assert data["force_phase_interval"] == .04


def test_corresponding_probe_points_and_cadences():
    reference, _ = case.reference_case()
    _, particles, _, _ = case.coupled_case()
    by_name = {s.name: s for s in reference.samplers}
    for sampler in particles.samplers.samples:
        if "phase_" in sampler.file_name or "transverse_" in sampler.file_name:
            expected = by_name[sampler.file_name.removeprefix("vpm_")]
            np.testing.assert_allclose(sampler.line_points, expected.points, atol=5e-7, rtol=0)
            assert sampler.schedule.interval*case.EXCHANGE == pytest.approx(
                expected.schedule.every_n_steps*case.DT)


def test_sampling_and_backups_align_with_both_clocks():
    for dt in (case.DT, case.EXCHANGE):
        for interval in (case.FORCES, case.PROFILES, case.SLICES, case.BACKUPS, case.VOLUMES):
            assert case.steps(interval, dt)*dt == pytest.approx(interval)
    with pytest.raises(ValueError):
        case.steps(.05, case.EXCHANGE)


def write_probes(path, rows):
    with path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "position_x", "velocity_x"])
        writer.writerows(rows)


def test_sample_audit_accepts_complete_data(tmp_path):
    p = tmp_path/"probe.csv"
    write_probes(p, [[.04, 1., 1.], [.08, 1., 2.]])
    row = audit.inspect_csv(p, 1, .04, .08, ["time", "position_x", "velocity_x"])
    assert row["rows"] == 2 and row["events"] == 2


@pytest.mark.parametrize("rows", [
    [[.04, 1., 1.]],
    [[.04, 1., 1.], [.04, 1., 1.], [.08, 1., 2.]],
    [[.04, 1., 1.], [.08, 2., 2.]],
    [[.04, 1., float("nan")], [.08, 1., 2.]],
    [[.08, 1., 2.], [.04, 1., 1.]],
])
def test_sample_audit_rejects_partial_duplicate_moving_or_invalid_records(tmp_path, rows):
    p = tmp_path/"probe.csv"
    write_probes(p, rows)
    with pytest.raises(AssertionError):
        audit.inspect_csv(p, 1, .04, .08, ["time", "position_x", "velocity_x"])


def test_force_frequency_and_normalization(tmp_path):
    p = tmp_path/"forces_history.csv"
    columns = ["time", "drag_coefficient", "lift_coefficient"]
    columns += [f"{q}_force_{c}" for q in ("total", "pressure", "viscous") for c in "xyz"]
    with p.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for t in np.arange(0, 100.001, .04):
            cl = .4*np.sin(2*np.pi*.18*t)
            row = dict.fromkeys(columns, 0.)
            row.update(time=t, drag_coefficient=1.3, lift_coefficient=cl,
                total_force_x=1.3*.48, pressure_force_x=1.3*.48,
                total_force_y=cl*.48, pressure_force_y=cl*.48)
            writer.writerow(row)
    result = audit.force_signal(p)
    assert result["usable_periodic_phase_signal"]
    assert result["strouhal"] == pytest.approx(.18, rel=1e-5)
    assert result["grid_independent"] is False
