"""Reference comparison uses common physical times and known field errors."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = (
    Path(__file__).resolve().parents[2]
    / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets/compare_reference.py"
)
SPEC = importlib.util.spec_from_file_location("cylinder_comparison", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_coupled_force_area_matches_the_supplied_finite_cylinder_geometry():
    import pyvista as pv

    setup_path = PATH.parents[1] / "setup.py"
    tree = ast.parse(setup_path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"DIAMETER", "CYLINDER_LENGTH", "REFERENCE_AREA"}:
                constants[name] = eval(
                    compile(ast.Expression(node.value), str(setup_path), "eval"),
                    {"__builtins__": {}},
                    constants,
                )
    surface = pv.read(PATH.parent / "cylinder_long.stl")
    xmin, xmax, _, _, zmin, zmax = surface.bounds
    assert constants["REFERENCE_AREA"] == pytest.approx((xmax - xmin) * (zmax - zmin))


def test_history_uses_only_shared_time_interval_and_preserves_coefficient_scale():
    reference = pd.DataFrame({"time": [0.0, 0.5, 1.0], "force": [0.0, 1.0, 2.0]})
    candidate = pd.DataFrame({"time": [0.25, 0.75, 1.25], "force": [0.625, 1.625, 2.625]})
    times, actual, expected, errors = MODULE.common_history(candidate, reference, ["force"])
    np.testing.assert_allclose(times[[0, -1]], [0.25, 1.0])
    np.testing.assert_allclose(actual - expected, 0.125)
    assert errors["force"]["rms"] == pytest.approx(0.125)
    assert errors["force"]["maximum"] == pytest.approx(0.125)
    with pytest.raises(ValueError, match="strictly increasing"):
        MODULE.common_history(pd.concat([candidate, candidate]), reference, ["force"])


def test_full_comparison_uses_common_snapshot_and_reports_missing_curves(tmp_path):
    samples = tmp_path / "samples"
    reference = tmp_path / "reference_flow/samples/medium"
    samples.mkdir()
    reference.mkdir(parents=True)
    forces = pd.DataFrame(
        {"time": [0, 1, 2], "drag_coefficient": [1, 1, 1], "lift_coefficient": [0, 0.1, 0]}
    )
    forces.to_csv(samples / "forces_history.csv", index=False)
    forces.to_csv(reference / "forces_history.csv", index=False)
    rows = [
        {
            "time": t,
            "position_x": 1,
            "position_y": y,
            "position_z": 0,
            "velocity_x": 1 - y * y + t,
            "velocity_y": y,
            "velocity_z": 0,
        }
        for t in (0, 1, 2)
        for y in (-1, 0, 1)
    ]
    ref = pd.DataFrame(rows)
    ref.to_csv(reference / "transverse_x1.csv", index=False)
    coupled = ref[ref.time <= 1].copy()
    coupled.velocity_y += 0.125
    coupled.to_csv(samples / "fvm_transverse_x1.csv", index=False)
    MODULE.compare(tmp_path, "medium")
    report = json.loads((tmp_path / "solution/cylinder_reference_comparison.json").read_text())
    assert report["profile_time"] == 1
    assert report["force_errors"]["drag_coefficient"]["rms"] == 0
    assert report["profile_errors"]["fvm_x1"]["velocity_rms_over_Uinf"] == pytest.approx(0.125)
    assert "vpm_transverse_x1" in report["missing_profiles"]
    assert (tmp_path / "figures/cylinder_reference_forces.png").is_file()
    assert (tmp_path / "figures/cylinder_reference_profiles.png").is_file()
