"""Regression contracts for tutorial post-processing after nomenclature changes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = REPOSITORY_ROOT / "tutorials"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_coupled_cylinder_plot_reads_canonical_force_coefficients():
    source = (
        TUTORIALS / "coupled_fvm_vpm/cylinder_shedding_flow/assets/plot_cylinder.py"
    ).read_text(encoding="utf-8")

    assert 'row["drag_coefficient"]' in source
    assert 'row["lift_coefficient"]' in source
    assert "forces_history.csv" in source


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
        TUTORIALS / "coupled_fvm_vpm/cylinder_shedding_flow/assets/plot_cylinder.py"
    ).read_text()
    assert 'data["analytic_total_kinetic_energy"]' in taylor_green
    assert 'data["analytic_energy"]' not in taylor_green
    assert 'row["drag_coefficient"]' in cylinder
    assert 'row["lift_coefficient"]' in cylinder

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


def test_lamb_oseen_postprocess_derives_speed_from_canonical_velocity_components():
    source = (TUTORIALS / "vpm/lamb_oseen_vortex/assets/postprocess.py").read_text(
        encoding="utf-8"
    )

    assert 'field["velocity_x"]' in source
    assert 'field["velocity_y"]' in source
    assert "np.hypot" in source


def test_lamb_oseen_has_separate_clean_plot_and_run_entry_points():
    tutorial = TUTORIALS / "vpm/lamb_oseen_vortex"
    shell_scripts = tuple(sorted(path.name for path in tutorial.glob("*.sh")))
    allrun = (tutorial / "allrun.sh").read_text(encoding="utf-8")
    allplot = (tutorial / "allplot.sh").read_text(encoding="utf-8")
    allclean = (tutorial / "allclean.sh").read_text(encoding="utf-8")

    assert shell_scripts == ("allclean.sh", "allplot.sh", "allrun.sh")
    assert "for " not in allrun
    assert "while " not in allrun
    assert "allplot.sh" not in allrun
    assert "allrun_rwm_ensemble.sh" not in allrun
    assert allrun.count("python -u lamb_oseen_setup.py") == 9
    assert allrun.count("python -u lamb_oseen_rwm_setup.py") == 3
    assert allrun.count("python assets/plot_") == 5
    assert "lamb_oseen_setup.py" not in allplot
    assert "lamb_oseen_rwm_setup.py" not in allplot
    assert allplot.count("python assets/plot_") == 5
    assert "python" not in allclean


def test_lamb_oseen_case_names_are_separate_from_energy_plot_metadata():
    diagnostics = _load_module(
        TUTORIALS / "vpm/lamb_oseen_vortex/assets/postprocess.py",
        "lamb_oseen_vortex_diagnostics_contract",
    )

    assert diagnostics.CASES == ("vortex", "dipole", "merging")
    assert tuple(case_name for case_name, _, _ in diagnostics.ENERGY_CASES) == diagnostics.CASES
    assert all(isinstance(case_name, str) for case_name in diagnostics.CASES)
    assert "is_pair_unresolved" in diagnostics.FIELD_CSV_COLUMNS
    assert "is_merged" not in diagnostics.FIELD_CSV_COLUMNS


def test_lamb_oseen_cs_is_always_the_top_comparison_layer():
    diagnostics = _load_module(
        TUTORIALS / "vpm/lamb_oseen_vortex/assets/postprocess.py",
        "lamb_oseen_plot_layer_contract",
    )

    assert set(diagnostics.SCHEME_DRAW_ORDER) == set(diagnostics.SCHEMES)
    assert diagnostics.SCHEME_DRAW_ORDER[-1] == "cs"
    assert diagnostics.scheme_zorder("cs") > max(
        diagnostics.scheme_zorder(scheme) for scheme in diagnostics.SCHEMES if scheme != "cs"
    )
    assert diagnostics.scheme_zorder("cs") > 100  # analytic/reference curves


def test_vpm_tutorials_keep_their_compact_sampling_budgets():
    lamb_oseen = (TUTORIALS / "vpm/lamb_oseen_vortex/lamb_oseen_setup.py").read_text(
        encoding="utf-8"
    )
    vortex_ring = (TUTORIALS / "vpm/vortex_ring/ring_setup.py").read_text(encoding="utf-8")

    assert "FIELD_SPACING = 0.16 * CORE_RADIUS" in lamb_oseen
    assert "MERGING_SAMPLE_INTERVAL_STEPS = 6" in lamb_oseen
    assert 'field_interval_steps = MERGING_SAMPLE_INTERVAL_STEPS if physics == "merging"' in lamb_oseen
    assert "every_n_steps=field_interval_steps" in lamb_oseen
    assert 'field_padding = 0.0 if physics == "vortex" else 3.0 * final_core_radius' in lamb_oseen
    assert "bounds=field_bounds" in lamb_oseen
    assert "vpm.Backup(" in lamb_oseen
    assert "n_steps = round(TOTAL_TIME / solver.time_step_size)" in lamb_oseen
    assert "if solver.step % sample_steps != 0:" in lamb_oseen
    assert "SAMPLE_INTERVAL_TIME = 0.1" in vortex_ring
    assert "BACKUP_INTERVAL_TIME = 0.5" in vortex_ring
    assert "vpm.Backup(" in vortex_ring


def test_vortex_ring_backup_contract_follows_each_run_horizon():
    postprocess = _load_module(
        TUTORIALS / "vpm/vortex_ring/assets/postprocess.py",
        "vortex_ring_postprocess_contract",
    )

    assert postprocess._expected_backup_steps(45, 25) == {0, 25, 45}
    assert postprocess._expected_backup_steps(1629, 25) == {
        0,
        1629,
        *range(25, 1626, 25),
    }
    assert postprocess._expected_backup_steps(3000, 25) == {
        0,
        *range(25, 3001, 25),
    }

    with pytest.raises(ValueError, match="positive"):
        postprocess._expected_backup_steps(3000, 0)


def test_vpm_tutorials_use_the_single_backup_constructor():
    setup_files = (
        "vpm/delta_wing/delta_wing_setup.py",
        "vpm/flat_plate/setup_plate.py",
        "vpm/lamb_oseen_vortex/lamb_oseen_setup.py",
        "vpm/quadcopter/quadcopter_setup.py",
        "vpm/rotor_flow/rotor_setup.py",
        "vpm/vortex_interactions/interactions_setup.py",
        "vpm/vortex_ring/ring_setup.py",
    )

    for relative_path in setup_files:
        source = (TUTORIALS / relative_path).read_text(encoding="utf-8")
        assert 'write_precision="f32"' in source
        assert "vpm.Backup(" in source
        assert "checkpoint" not in source.lower()
