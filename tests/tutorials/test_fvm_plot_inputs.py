"""Missing or empty native histories cannot masquerade as successful plots."""

import pytest

from openonda import plotting
from tests._tutorial_helpers import load_tutorial_module


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
