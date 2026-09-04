"""Behavioral tutorial tests for sampled data and post-processing schemas."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = REPOSITORY_ROOT / "tutorials"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    postprocess = _load_module(
        TUTORIALS / "vpm/vortex_ring/assets/postprocess.py",
        "vortex_ring_backup_schedule_test",
    )

    assert postprocess._expected_backup_steps(45, 25) == {0, 25, 45}
    assert postprocess._expected_backup_steps(100, 25) == {0, 25, 50, 75, 100}
    with pytest.raises(ValueError, match="positive"):
        postprocess._expected_backup_steps(100, 0)
