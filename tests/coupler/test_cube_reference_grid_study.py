"""Focused tests for the compact cube-reference grid study."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ANALYSER = (
    ROOT
    / "tutorials"
    / "coupled_fvm_vpm"
    / "cube_flow"
    / "reference_flow"
    / "assets"
    / "analyse_grid_study.py"
)


def _load_analyser():
    spec = spec_from_file_location("cube_reference_grid_study", ANALYSER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_second_order_drag_sequence_produces_a_conservative_dyadic_recommendation():
    analyser = _load_analyser()
    values = {grid: 1.0 + spacing**2 for grid, spacing in analyser.GRID_SPACING.items()}

    result = analyser._spatial_convergence(values, tolerance=0.001)

    assert result["monotone"]
    assert result["observed_order"] == pytest.approx(2.0)
    assert result["richardson_extrapolated_value"] == pytest.approx(1.0)
    assert result["passed"]
    assert result["recommended_surface_cell_size"] == pytest.approx(1.0 / 64.0)


def test_existing_fine_compaction_keeps_only_requested_line_times(tmp_path):
    analyser = _load_analyser()
    source = tmp_path / "existing"
    destination = tmp_path / "compact" / "h64"
    source.mkdir()
    (source / "forces_history.csv").write_text(
        "time,patch,drag_coefficient,lift_coefficient,side_force_coefficient\n"
        "0.05,cube,1.0,0.0,0.0\n"
        "0.50,cube,1.1,0.0,0.0\n",
        encoding="utf-8",
    )
    line_header = "time,step,position_x,velocity_x\n"
    line_rows = "0.05,5,1.0,0.1\n0.25,25,1.0,0.2\n0.50,50,1.0,0.3\n"
    for name in analyser.LINE_FILES:
        (source / name).write_text(line_header + line_rows, encoding="utf-8")

    analyser.prepare_existing_fine(source, destination, line_interval=0.25)

    for name in analyser.LINE_FILES:
        lines = (destination / name).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert lines[1].startswith("0.25,")
        assert lines[2].startswith("0.50,")
    metadata = json.loads((destination / "grid_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["grid"] == "h64"
    assert metadata["execution"]["reused_existing_samples"] is True
