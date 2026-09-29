"""Missing or empty native histories cannot masquerade as successful plots."""

import ast
from pathlib import Path

import pytest

from openonda import plotting
from tests._tutorial_helpers import load_tutorial_module


def test_fvm_plot_text_uses_shared_thesis_font_size():
    tutorials = Path(__file__).resolve().parents[2] / "tutorials" / "fvm"
    for script in tutorials.glob("*/assets/plot_*.py"):
        tree = ast.parse(script.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "fontsize":
                assert not isinstance(node.value, ast.Constant), (
                    script,
                    "Literal text sizes override the fixed thesis font and fail real plots",
                )


@pytest.mark.parametrize("case", ["airfoil_flow", "boundary_layer", "step_profile"])
def test_required_column_history_rejects_header_only_csv(tmp_path, monkeypatch, case):
    monkeypatch.setattr(plotting, "set_thesis_style", lambda: None)
    common = load_tutorial_module(f"fvm/{case}", "assets._common")
    path = tmp_path / "profiles.csv"
    path.write_text("time,velocity_x\n")
    with pytest.raises(ValueError, match="has no records"):
        common.load_csv_columns(path)
    path.write_text("time,velocity_x\n0,1\n0.1,0.5\n")
    data = common.load_csv_columns(path)
    assert data["time"].tolist() == [0, 0.1]
    assert data["velocity_x"].tolist() == [1, 0.5]


@pytest.mark.parametrize("case", ["airfoil_flow", "cube_flow", "cylinder_ibm"])
def test_required_force_history_rejects_header_only_csv(tmp_path, monkeypatch, case):
    monkeypatch.setattr(plotting, "set_thesis_style", lambda: None)
    common = load_tutorial_module(f"fvm/{case}", "assets._common")
    ibm = case == "cylinder_ibm"
    identifier = "body_id" if ibm else "patch"
    filename = "ibm_forces_history.csv" if ibm else "forces_history.csv"
    samples = tmp_path / "samples"
    samples.mkdir()
    path = samples / filename
    path.write_text(f"{identifier},time,drag_coefficient\n")
    load = common.load_ibm_forces_csv if ibm else common.load_forces_csv
    with pytest.raises(ValueError, match="has no records"):
        load(tmp_path / "solution")
    path.write_text(f"{identifier},time,drag_coefficient\ncylinder,0,1.25\n")
    assert load(tmp_path / "solution")["cylinder"]["drag_coefficient"].tolist() == [1.25]
