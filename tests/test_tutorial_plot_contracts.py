"""Regression contracts for tutorial post-processing after nomenclature changes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = REPOSITORY_ROOT / "tutorials"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coupled_line_loader_sorts_along_the_varying_coordinate(tmp_path):
    plot_util = _load_module(
        TUTORIALS / "coupled_fvm_vpm/cylinder_shedding_flow/assets/_plotutil.py",
        "cylinder_shedding_plot_util",
    )
    plot_util.SOURCES["fvm"]["dir"] = tmp_path
    (tmp_path / "fvm_spanwise_line.csv").write_text(
        "time,step,position_x,position_y,position_z,velocity_y\n"
        "1.0,1,1.5,0.0,1.0,10.0\n"
        "1.0,1,1.5,0.0,-1.0,-10.0\n"
        "1.0,1,1.5,0.0,0.0,0.0\n",
        encoding="utf-8",
    )

    frame = plot_util.load_line("fvm", "spanwise_line", 1.0)

    assert frame is not None
    np.testing.assert_array_equal(frame["position_z"], [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(frame["velocity_y"], [-10.0, 0.0, 10.0])


def test_cube_plot_metadata_accepts_legacy_and_current_coupling_contracts():
    plot_util = _load_module(
        TUTORIALS / "coupled_fvm_vpm/cube_flow/assets/_plotutil.py",
        "cube_flow_plot_metadata_contract",
    )

    plot_util._validate_metadata_provenance(
        {
            "schema_version": 2,
            "coupling_method": "absolute_common_m4_lattice_blend",
        }
    )
    plot_util._validate_metadata_provenance(
        {
            "schema_version": 3,
            "coupling_method": "buffered_m4_renewal",
        }
    )

    with pytest.raises(ValueError, match="supported cube-flow coupling metadata"):
        plot_util._validate_metadata_provenance(
            {
                "schema_version": 3,
                "coupling_method": "absolute_common_m4_lattice_blend",
            }
        )


def test_plotters_use_canonical_export_fields_and_valid_math_commands():
    taylor_green = (TUTORIALS / "fvm/taylor_green/assets/plot_decay.py").read_text()
    cylinder = (
        TUTORIALS / "coupled_fvm_vpm/cylinder_shedding_flow/assets/plot_von_karman.py"
    ).read_text()
    assert 'data["analytic_total_kinetic_energy"]' in taylor_green
    assert 'data["analytic_energy"]' not in taylor_green
    assert 'frame["position_z"]' in cylinder
    assert 'table["z"]' not in cylinder

    invalid_math_command = re.compile(
        r"\\(?:angular_frequency|angular_velocity|nondimensional_time|mathbs)\b"
    )
    offenders = []
    for path in TUTORIALS.rglob("*.py"):
        if "cube_flow" in path.parts:
            continue
        if invalid_math_command.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(REPOSITORY_ROOT))
    assert offenders == []


def test_vortex_interaction_postprocessor_uses_canonical_cadence_keys():
    source = (TUTORIALS / "vpm/vortex_interactions/assets/postprocess.py").read_text(
        encoding="utf-8"
    )

    assert re.search(r'manifest\.get\("checkpoint_interval_steps"(?:,\s*0)?\)', source)
    assert 'manifest.get("snapshot_frequency")' not in source
    assert re.search(
        r'range\(ci,\s*manifest\["completed_steps"\]\s*\+\s*1,\s*ci\)',
        source,
    )


def test_lamb_oseen_surface_plot_derives_speed_from_the_canonical_velocity_vector():
    source = (TUTORIALS / "vpm/lamb_oseen_vortex/assets/plot_vortex_surface_fields.py").read_text(
        encoding="utf-8"
    )

    assert 'GetArray("velocity")' in source
    assert "np.linalg.norm(velocity, axis=2)" in source
    assert 'GetArray("velocity_magnitude")' not in source


def test_lamb_oseen_case_names_are_separate_from_energy_plot_metadata():
    diagnostics = _load_module(
        TUTORIALS / "vpm/lamb_oseen_vortex/assets/vortex_diagnostics.py",
        "lamb_oseen_vortex_diagnostics_contract",
    )

    assert diagnostics.CASES == ("vortex", "dipole", "merging")
    assert tuple(case_name for case_name, _, _ in diagnostics.ENERGY_CASES) == diagnostics.CASES
    assert all(isinstance(case_name, str) for case_name in diagnostics.CASES)


def test_vpm_tutorials_keep_their_compact_sampling_budgets():
    lamb_oseen = (TUTORIALS / "vpm/lamb_oseen_vortex/lamb_oseen_setup.py").read_text(
        encoding="utf-8"
    )
    vortex_ring = (TUTORIALS / "vpm/vortex_ring/ring_setup.py").read_text(encoding="utf-8")

    assert "FIELD_SPACING = 0.16 * CORE_RADIUS" in lamb_oseen
    assert "MERGING_SAMPLE_INTERVAL_TIME = SAMPLE_INTERVAL_TIME" in lamb_oseen
    assert 'field_padding = 0.0 if physics == "vortex" else 3.0 * final_core_radius' in lamb_oseen
    assert "bounds=field_bounds" in lamb_oseen
    assert "checkpoint_store_velocity_gradient=False" in lamb_oseen
    assert "SAMPLE_INTERVAL_TIME = 0.1" in vortex_ring
    assert "CHECKPOINT_INTERVAL_TIME = 0.5" in vortex_ring
    assert "checkpoint_store_velocity_gradient=False" in vortex_ring


def test_vpm_tutorials_request_compact_restartable_checkpoints():
    setup_files = (
        "vpm/delta_wing/delta_wing_setup.py",
        "vpm/flat_plate/setup_plate.py",
        "vpm/lamb_oseen_vortex/lamb_oseen_setup.py",
        "vpm/quadcopter/quadcopter_setup.py",
        "vpm/rotor_flow/rotor_setup.py",
        "vpm/vortex_interactions/rings_setup.py",
        "vpm/vortex_ring/ring_setup.py",
    )

    for relative_path in setup_files:
        source = (TUTORIALS / relative_path).read_text(encoding="utf-8")
        assert 'write_precision="f32"' in source
        assert "checkpoint_store_velocity_gradient=False" in source
