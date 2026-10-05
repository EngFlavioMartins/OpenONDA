"""Behavioral tutorial tests for sampled data and post-processing schemas."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from tests._tutorial_helpers import load_tutorial_module

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = REPOSITORY_ROOT / "tutorials"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_repository_tutorial(name: str):
    """Import a tutorial module outside pytest's ``tests/tutorials`` namespace."""
    parts = name.split(".")
    if parts[:1] != ["tutorials"] or len(parts) < 3:
        raise ValueError(f"unsupported repository tutorial path: {name}")
    catalog_name = f"{parts[1]}/{parts[2]}"
    module = ".".join(parts[3:]) or "setup"
    return load_tutorial_module(catalog_name, module)


def _write_vpm_metadata(
    root: Path,
    case_name: str,
    *,
    stretching_scheme: str,
    status: str = "completed",
    step: int = 3000,
    time: float = 60.0,
    requested_steps: int = 3000,
    turbulence_model: str = "DNS",
) -> dict:
    """Write the universal solver record needed by tutorial-reader tests."""
    record = {
        "schema_version": 2,
        "solver": "VPM",
        "case_name": case_name,
        "configuration": {
            "numerics": {
                "integrator": {"type": "RKTableau", "name": "SSPRK3"},
                "induction": {
                    "type": "TreecodeInduction",
                    "method": "TREECODE",
                    "stretching_scheme": stretching_scheme,
                },
                "viscous": {"type": "ViscousConfig", "scheme": "CS"},
                "turbulence": {
                    "type": "TurbulenceConfig",
                    "model": turbulence_model,
                },
            },
            "run": {
                "steps": requested_steps,
                "initial_samples": True,
                "final_backup": True,
                "state_limit_action": "STOP",
                "wall_time_limit_seconds": None,
            },
            "backup": {
                "interval_steps": 25,
                "directory": f"solution/{case_name}",
                "log_directory": f"solution/{case_name}",
            },
            "samplers": {"directory": case_name, "items": []},
            "initial_conditions": [],
            "initial_weak_particle_percent": 0.0,
        },
        "state": {
            "initial_step": 0,
            "initial_time": 0.0,
            "step": step,
            "time": time,
            "requested_steps": requested_steps,
            "initial_n_particles_total": 8772,
            "n_particles_total": 8772,
        },
        "run_status": {"status": status},
    }
    destination = root / "solution" / case_name / "vpm_metadata.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record


def test_lamb_oseen_surface_reader_round_trips_the_sampler_schema(tmp_path: Path):
    import pyvista as pv

    diagnostics = _load_module(
        TUTORIALS / "vpm/01_lamb_oseen_vortex/assets/postprocess.py",
        "lamb_oseen_surface_schema_test",
    )
    x, y = np.meshgrid([-0.5, 0.5], [-0.25, 0.25], indexing="ij")
    points = np.column_stack((x.reshape(-1, order="F"), y.reshape(-1, order="F"), np.zeros(4)))
    velocity = np.column_stack((points[:, 0] + 1.0, points[:, 1] - 2.0, np.zeros(4)))
    vorticity = np.column_stack((np.zeros(4), np.zeros(4), points[:, 0] - points[:, 1]))
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (2, 2, 1)
    grid.point_data["velocity"] = velocity
    grid.point_data["vorticity"] = vorticity
    path = tmp_path / "surface.vts"
    grid.save(path)

    field = diagnostics.read_surface_field(path)

    assert {"x", "y", "velocity_x", "velocity_y", "vorticity_z"} <= set(field)
    np.testing.assert_allclose(field["velocity_x"], field["x"] + 1.0)
    np.testing.assert_allclose(field["velocity_y"], field["y"] - 2.0)
    np.testing.assert_allclose(field["vorticity_z"], field["x"] - field["y"])


def test_lamb_oseen_energy_reader_preserves_recorded_induction_method(tmp_path: Path):
    diagnostics = _load_module(
        TUTORIALS / "vpm/01_lamb_oseen_vortex/assets/postprocess.py",
        "lamb_oseen_energy_schema_test",
    )
    path = tmp_path / "flow_integrals.csv"
    path.write_text(
        "time,kinetic_energy_rate,viscous_kinetic_energy_rate,kinetic_energy_rate_source\n"
        "0.1,-1.0,-0.9,direct_energy_backward_difference\n"
        "0.2,-2.0,-1.8,undefined_dynamic_fourier_box\n",
        encoding="utf-8",
    )

    data = diagnostics.read_flow_integrals(path)

    assert data is not None
    np.testing.assert_allclose(data["time"], [0.1, 0.2])
    assert data["kinetic_energy_rate"][0] == pytest.approx(-1.0)
    assert np.isnan(data["kinetic_energy_rate"][1])
    np.testing.assert_allclose(data["viscous_kinetic_energy_rate"], [-0.9, -1.8])


def test_lamb_oseen_energy_reader_keeps_persistent_fourier_rate(tmp_path: Path):
    diagnostics = _load_module(
        TUTORIALS / "vpm/01_lamb_oseen_vortex/assets/postprocess.py",
        "lamb_oseen_persistent_fourier_energy_schema_test",
    )
    path = tmp_path / "flow_integrals.csv"
    path.write_text(
        "time,kinetic_energy_rate,viscous_kinetic_energy_rate,kinetic_energy_rate_source\n"
        "0.1,-1.0,-0.9,direct_energy_backward_difference\n"
        "0.2,-1.8,-1.7,fourier_energy_backward_difference\n",
        encoding="utf-8",
    )

    data = diagnostics.read_flow_integrals(path)

    assert data is not None
    np.testing.assert_allclose(data["kinetic_energy_rate"], [-1.0, -1.8])


def test_lamb_oseen_reads_native_solver_metadata(tmp_path: Path):
    diagnostics = _load_module(
        TUTORIALS / "vpm/01_lamb_oseen_vortex/assets/postprocess.py",
        "lamb_oseen_solver_metadata_test",
    )
    record = _write_vpm_metadata(
        tmp_path,
        "vortex_cs",
        stretching_scheme="TRANSPOSED",
        step=9,
        time=0.9,
        requested_steps=9,
    )
    record["configuration"]["numerics"].update(
        {
            "time_step_size": 0.1,
            "random_seed": 17,
            "compute_device": "CPU",
            "precision": "f32",
            "write_precision": "f32",
            "particle_kernel": "GAUSSIAN",
        }
    )
    record["configuration"]["initial_conditions"] = [
        {
            "type": "VortexFilament",
            "centre": [0.0, 0.0, 0.0],
            "circulation": 1.0,
            "vortex_core_radius": 0.125,
            "kinematic_viscosity": 0.002,
            "distribution": {
                "type": "TriangularPrismDistribution",
                "bounds": [[-1.0, 1.0], [-1.0, 1.0], [-2.5, 2.5]],
                "spacing": 0.075,
                "core_radius_ratio": 1.2,
            },
        }
    ]
    destination = tmp_path / "solution/vortex_cs/vpm_metadata.json"
    destination.write_text(json.dumps(record) + "\n", encoding="utf-8")
    samples = tmp_path / "samples/vortex_cs"
    samples.mkdir(parents=True)

    metadata = diagnostics._metadata(samples)

    assert metadata["status"] == "completed"
    assert metadata["number_of_steps"] == 9
    assert metadata["final_time"] == pytest.approx(0.9)
    assert metadata["random_seed"] == 17
    assert metadata["core_radius"] == pytest.approx(0.125)
    assert metadata["particle_spacing"] == pytest.approx(0.075)
    assert metadata["particle_core_radius"] == pytest.approx(0.09)
    assert metadata["column_half_length"] == pytest.approx(2.5)


def test_vortex_ring_backup_schedule_follows_the_completed_horizon():
    postprocess = __import__("tests.support.vpm.vortex_ring.postprocess", fromlist=["*"])

    assert postprocess._expected_backup_steps(45, 25) == {25}
    assert postprocess._expected_backup_steps(100, 25) == {25, 50, 75, 100}
    assert postprocess._expected_backup_steps(0, 25) == set()
    with pytest.raises(ValueError, match="positive"):
        postprocess._expected_backup_steps(100, 0)


def test_vortex_ring_empty_summary_has_no_ranked_case(tmp_path: Path):
    postprocess = __import__("tests.support.vpm.vortex_ring.postprocess", fromlist=["*"])

    summary = postprocess.build_summary(tmp_path / "samples", tmp_path / "figures")

    assert summary["runs"] == {}
    assert summary["stability_ranking"] == []
    assert summary["longest_sustained_variant"] is None
    assert summary["longest_sustained_variants"] == []


def test_vortex_ring_flow_integrals_allow_initially_undefined_particle_metrics_only(tmp_path: Path):
    postprocess = __import__("tests.support.vpm.vortex_ring.postprocess", fromlist=["*"])
    path = tmp_path / "flow_integrals.csv"
    path.write_text(
        "time,step,total_kinetic_energy,strain_increment_infinity\n0.0,0,1.0,\n0.1,5,0.9,0.2\n",
        encoding="utf-8",
    )

    assert postprocess._readable_finite_csv(path)

    path.write_text(
        "time,step,total_kinetic_energy,strain_increment_infinity\n0.0,0,1.0,\n0.1,5,0.9,\n",
        encoding="utf-8",
    )
    assert not postprocess._readable_finite_csv(path)


def test_vortex_ring_available_plot_validation_accepts_partial_parameter_study(
    tmp_path: Path, monkeypatch
):
    postprocess = __import__("tests.support.vpm.vortex_ring.postprocess", fromlist=["*"])
    samples = tmp_path / "samples"
    figures = tmp_path / "figures"
    dns = samples / "dns_direct"
    dns.mkdir(parents=True)
    _write_vpm_metadata(
        tmp_path,
        "dns_direct",
        stretching_scheme="DIRECT",
        status="running",
        step=5,
        time=0.1,
    )
    (dns / "ring_diagnostics.csv").write_text(
        "time,step,vortex_centroid_x,tube_circulation\n0.0,0,0.0,3.14\n0.1,5,0.2,3.13\n",
        encoding="utf-8",
    )
    (dns / "flow_integrals.csv").write_text(
        "time,step,total_kinetic_energy,kinetic_energy_rate,viscous_kinetic_energy_rate,"
        "n_particles_total\n"
        "0.0,0,1.0,0.0,-0.1,10\n0.1,5,0.9,-0.1,-0.1,10\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(postprocess, "SAMPLES_DIR", samples)
    monkeypatch.setattr(postprocess, "FIGURES_DIR", figures)

    assert postprocess.validate_available(pre_plot=True) == 0
    summary = postprocess.build_summary(samples, figures)
    assert summary["runs"]["dns_direct"]["status"] == "running"
    assert summary["runs"]["dns_direct"]["completed_steps"] == 5
    assert summary["runs"]["dns_direct"]["completed_time"] == pytest.approx(0.1)
    assert summary["runs"]["dns_direct"]["n_particles_total"] == 8772
    json.dumps(summary)


def test_vortex_ring_saffman_comparison_is_limited_to_thin_cores():
    metrics = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.ring_metrics")

    limit = metrics.saffman_valid_time_limit()
    core_ratio = (
        np.sqrt(metrics.CORE_RADIUS**2 + 4.0 * metrics.KINEMATIC_VISCOSITY * limit)
        / metrics.RING_RADIUS
    )

    assert core_ratio == pytest.approx(metrics.SAFFMAN_MAX_CORE_RATIO)
    assert limit / metrics.REFERENCE_TIME == pytest.approx(60.0)
    assert metrics.saffman_speed(np.array([0.0]))[0] == pytest.approx(metrics.REFERENCE_VELOCITY)


def test_ring_histories_preserve_large_finite_native_samples_and_their_clock(tmp_path):
    import h5py

    metrics = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.ring_metrics")
    paths = []
    clock = np.array([0.0, 0.4])
    for step, time, centre, radius, strength in (
        (0, clock[0], 0.0, 1.0, 0.25),
        (40, clock[1], 2500.0, 1200.0, 1000.0),
    ):
        path = tmp_path / f"vpm_{step:06d}.h5"
        with h5py.File(path, "w") as archive:
            solver = archive.create_group("solver")
            solver.attrs.update(step=step, time=time, n_particles_total=4)
            particles = archive.create_group("particles")
            particles["position"] = np.array(
                [
                    [centre, radius, 0.0],
                    [centre, 0.0, radius],
                    [centre, -radius, 0.0],
                    [centre, 0.0, -radius],
                ]
            )
            particles["vortex_strength"] = strength * np.array(
                [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
            )
            particles["core_radius"] = np.full(4, 0.1)
            particles["group_id"] = np.zeros(4, dtype=int)
        paths.append(path)

    raw = metrics.load_ring_data(paths)
    normalized = metrics.normalise_ring_data(raw)[0]
    np.testing.assert_array_equal(normalized["t_norm"], clock / metrics.REFERENCE_TIME)
    np.testing.assert_allclose(normalized["x_norm"], [0.0, 2500.0])
    np.testing.assert_allclose(normalized["R_norm"], [1.0, 1200.0])
    strength_time, strength = metrics.load_length_integrated_strength(paths)
    np.testing.assert_array_equal(strength_time, clock / metrics.REFERENCE_TIME)
    np.testing.assert_array_equal(strength, [1.0, 4000.0])
    assert raw[0][-1]["max_vortex_strength_magnitude"] == 1000.0


def test_ring_history_reader_uses_native_clock_checks_without_rewriting_rows(tmp_path):
    metrics = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.ring_metrics")
    path = tmp_path / "ring_diagnostics.csv"
    original = "time,step,vortex_centroid_x\n0.0,0,0.0\n0.4,40,2500.0\n"
    path.write_text(original, encoding="utf-8")
    frame = metrics.load_sampled_ring_data(path)
    np.testing.assert_array_equal(frame.time, [0.0, 0.4])
    np.testing.assert_array_equal(frame.vortex_centroid_x, [0.0, 2500.0])
    assert path.read_text(encoding="utf-8") == original
    path.write_text(original + "0.4,40,3000.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate or nonmonotonic"):
        metrics.load_sampled_ring_data(path)


def test_vortex_ring_records_a_resolution_limit_as_a_terminal_result(tmp_path: Path, monkeypatch):
    setup = _import_repository_tutorial("tutorials.vpm.vortex_ring.setup")
    assert not hasattr(setup, "write_run_metadata")
    assert not hasattr(setup, "result_exists")
    metadata = _write_vpm_metadata(
        tmp_path,
        "dns_direct",
        stretching_scheme="DIRECT",
        status="resolution_lost",
        step=242,
        time=4.84,
    )
    assert metadata["run_status"] == {"status": "resolution_lost"}
    assert metadata["state"]["step"] == 242
    assert metadata["state"]["time"] == pytest.approx(4.84)
    serialized = json.dumps(metadata).lower()
    assert "reason" not in serialized
    assert "failure" not in serialized

    sample_directory = tmp_path / "samples/dns_direct"
    sample_directory.mkdir(parents=True)
    for csv_name in ("flow_integrals.csv", "ring_diagnostics.csv", "ring_modes.csv"):
        (sample_directory / csv_name).write_text("time,step\n4.84,242\n", encoding="utf-8")

    postprocess = __import__("tests.support.vpm.vortex_ring.postprocess", fromlist=["*"])
    monkeypatch.setattr(postprocess, "SAMPLES_DIR", tmp_path / "samples")
    monkeypatch.setattr(postprocess, "SOLUTION_DIR", tmp_path / "solution")
    checked_metadata, expected_steps, failures = postprocess._run_validation("dns_direct")
    assert checked_metadata["status"] == "resolution_lost"
    assert expected_steps == set(range(25, 243, 25))
    assert failures == []


def test_vortex_ring_ranks_the_last_instability_from_terminal_times(tmp_path: Path):
    postprocess = __import__("tests.support.vpm.vortex_ring.postprocess", fromlist=["*"])
    metrics = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.ring_metrics")
    schemes = {
        "dns_direct": ("DIRECT", 4.84),
        "dns_transposed": ("TRANSPOSED", 12.0),
        "dns_mixed": ("MIXED", 8.0),
        "les_transposed": ("TRANSPOSED", 15.0),
    }
    for variant, (scheme, terminal_time) in schemes.items():
        directory = tmp_path / "samples" / variant
        directory.mkdir(parents=True)
        _write_vpm_metadata(
            tmp_path,
            variant,
            stretching_scheme=scheme,
            status="resolution_lost",
            step=round(terminal_time / 0.02),
            time=terminal_time,
            turbulence_model="LES_SMAGORINSKY" if variant.startswith("les_") else "DNS",
        )

    summary = postprocess.build_summary(tmp_path / "samples", tmp_path / "figures")
    assert summary["stability_ranking"] == [
        "les_transposed",
        "dns_transposed",
        "dns_mixed",
        "dns_direct",
    ]
    assert summary["longest_sustained_variant"] == "les_transposed"

    results = metrics.load_stability_results(tmp_path / "samples")
    assert {result["variant"]: result["time"] for result in results} == {
        variant: terminal_time for variant, (_, terminal_time) in schemes.items()
    }
