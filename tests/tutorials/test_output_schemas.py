"""Behavioral tutorial tests for sampled data and post-processing schemas."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.config.health import HealthError

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
    existing = sys.modules.get("tutorials")
    if existing is not None and getattr(existing, "__file__", None) is None:
        del sys.modules["tutorials"]
    root = str(REPOSITORY_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def test_cube_plot_metadata_accepts_only_supported_coupling_schemas():
    plot_util = _load_module(
        TUTORIALS / "coupled_fvm_vpm/cube_flow/assets/_plotutil.py",
        "cube_flow_plot_metadata_test",
    )

    plot_util._validate_metadata_provenance(
        {"schema_version": 2, "coupling_method": "absolute_common_m4_lattice_blend"}
    )
    plot_util._validate_metadata_provenance(
        {"schema_version": 3, "coupling_method": "buffered_m4_renewal"}
    )
    with pytest.raises(ValueError, match="supported cube-flow coupling metadata"):
        plot_util._validate_metadata_provenance(
            {"schema_version": 3, "coupling_method": "absolute_common_m4_lattice_blend"}
        )


def test_lamb_oseen_surface_reader_round_trips_the_sampler_schema(tmp_path: Path):
    import pyvista as pv

    diagnostics = _load_module(
        TUTORIALS / "vpm/lamb_oseen_vortex/assets/postprocess.py",
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

    assert set(field) == {"x", "y", "velocity_x", "velocity_y", "vorticity_z"}
    np.testing.assert_allclose(field["velocity_x"], field["x"] + 1.0)
    np.testing.assert_allclose(field["velocity_y"], field["y"] - 2.0)
    np.testing.assert_allclose(field["vorticity_z"], field["x"] - field["y"])


def test_lamb_oseen_energy_reader_preserves_backend_provenance(tmp_path: Path):
    diagnostics = _load_module(
        TUTORIALS / "vpm/lamb_oseen_vortex/assets/postprocess.py",
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
        TUTORIALS / "vpm/lamb_oseen_vortex/assets/postprocess.py",
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


def test_vortex_ring_backup_schedule_follows_the_completed_horizon():
    postprocess = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.postprocess")

    assert postprocess._expected_backup_steps(45, 25) == {25}
    assert postprocess._expected_backup_steps(100, 25) == {25, 50, 75, 100}
    assert postprocess._expected_backup_steps(0, 25) == set()
    with pytest.raises(ValueError, match="positive"):
        postprocess._expected_backup_steps(100, 0)


def test_vortex_ring_flow_integrals_allow_initially_undefined_health_only(tmp_path: Path):
    postprocess = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.postprocess")
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


def test_vortex_ring_available_plot_validation_accepts_partial_campaign(
    tmp_path: Path, monkeypatch
):
    postprocess = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.postprocess")
    samples = tmp_path / "samples"
    figures = tmp_path / "figures"
    dns = samples / "dns_direct"
    dns.mkdir(parents=True)
    (dns / "run_metadata.json").write_text(
        '{"schema_version": 3, "status": "running", "variant": "dns_direct", '
        '"stretching_scheme": "DIRECT"}\n',
        encoding="utf-8",
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
    manifest = postprocess.build_manifest(samples, figures)
    assert manifest["runs"]["dns_direct"]["status"] == "running"
    assert manifest["runs"]["dns_direct"]["completed_steps"] == 5
    assert manifest["runs"]["dns_direct"]["completed_time"] == pytest.approx(0.1)
    assert manifest["runs"]["dns_direct"]["n_particles_total"] == 10
    postprocess.json.dumps(manifest)


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


def test_vortex_ring_plot_selection_keeps_compatible_results_during_new_campaign(tmp_path: Path):
    metrics = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.ring_metrics")

    metadata = {
        "dns_direct": (3, "DIRECT"),
        "dns_treecode": (2, None),
        "les_treecode": (2, None),
    }
    for variant, (schema, scheme) in metadata.items():
        directory = tmp_path / variant
        directory.mkdir()
        payload = {"schema_version": schema, "variant": variant}
        if scheme is not None:
            payload["stretching_scheme"] = scheme
        (directory / "run_metadata.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert metrics.plot_variants(tmp_path) == (
        "dns_direct",
        "dns_treecode",
        "les_treecode",
    )

    transposed = tmp_path / "dns_transposed"
    transposed.mkdir()
    (transposed / "run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "variant": "dns_transposed",
                "stretching_scheme": "TRANSPOSED",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert metrics.plot_variants(tmp_path) == (
        "dns_direct",
        "dns_transposed",
        "les_treecode",
    )


def test_vortex_ring_records_a_resolution_limit_as_a_terminal_result(tmp_path: Path, monkeypatch):
    setup = _import_repository_tutorial("tutorials.vpm.vortex_ring.setup")
    monkeypatch.setattr(setup, "TUTORIAL_DIR", tmp_path)
    failure = HealthError("declared Lagrangian CFL limit")
    solver = SimpleNamespace(
        run_status="resolution_lost",
        run_failure=failure,
        step=242,
        time=4.84,
        integrator_tableau=SimpleNamespace(name="SSPRK3", order=3),
        induction=SimpleNamespace(
            method="TREECODE",
            strength_rate_mode="HIERARCHICAL_GRADIENT",
            stretching_scheme="DIRECT",
        ),
        viscous_scheme="CS",
        compute_device="CPU",
        particles=SimpleNamespace(n_particles_total=8772),
    )

    setup.write_run_metadata(
        variant="dns_direct",
        n_steps=3000,
        particle_core_radius=0.07,
        initial_n_particles_total=8772,
        solver=solver,
    )

    metadata = json.loads(
        (tmp_path / "samples/dns_direct/run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == 4
    assert metadata["status"] == "instability_detected"
    assert metadata["outcome"] == "instability_detected"
    assert not metadata["completed"]
    assert metadata["requested_steps"] == 3000
    assert metadata["completed_steps"] == 242
    assert metadata["instability_step"] == 242
    assert metadata["instability_time"] == pytest.approx(4.84)
    assert metadata["instability_reason"].startswith("HealthError:")
    assert metadata["maximum_lagrangian_cfl"] == pytest.approx(1.0)
    assert metadata["maximum_vorticity_divergence_error"] == pytest.approx(0.12)
    assert metadata["maximum_vortex_misalignment_degrees"] == pytest.approx(25.0)

    sample_directory = tmp_path / "samples/dns_direct"
    for csv_name in ("flow_integrals.csv", "ring_diagnostics.csv", "ring_modes.csv"):
        (sample_directory / csv_name).write_text("time,step\n4.84,242\n", encoding="utf-8")
    assert setup.result_exists("dns_direct", 3000)
    assert not setup.result_exists("dns_direct", 2999)

    postprocess = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.postprocess")
    monkeypatch.setattr(postprocess, "SAMPLES_DIR", tmp_path / "samples")
    monkeypatch.setattr(postprocess, "SOLUTION_DIR", tmp_path / "solution")
    checked_metadata, expected_steps, failures = postprocess._run_validation("dns_direct")
    assert checked_metadata["status"] == "instability_detected"
    assert expected_steps == set(range(25, 243, 25))
    assert failures == []


def test_vortex_ring_ranks_the_last_instability_from_terminal_times(tmp_path: Path):
    postprocess = _import_repository_tutorial("tutorials.vpm.vortex_ring.assets.postprocess")
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
        (directory / "run_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "variant": variant,
                    "stretching_scheme": scheme,
                    "status": "instability_detected",
                    "completed_steps": round(terminal_time / 0.02),
                    "requested_steps": 3000,
                    "final_time": terminal_time,
                    "instability_step": round(terminal_time / 0.02),
                    "instability_time": terminal_time,
                    "instability_reason": "HealthError: common CFL limit",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = postprocess.build_manifest(tmp_path / "samples", tmp_path / "figures")
    assert manifest["stability_ranking"] == [
        "les_transposed",
        "dns_transposed",
        "dns_mixed",
        "dns_direct",
    ]
    assert manifest["longest_sustained_variant"] == "les_transposed"

    results = metrics.load_stability_results(tmp_path / "samples")
    assert {result["variant"]: result["time"] for result in results} == {
        variant: terminal_time for variant, (_, terminal_time) in schemes.items()
    }
