"""Current cylinder comparisons preserve native area, time and field-error scales."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.panel_removal import compare_cylinder_run as comparison

PATH = (
    Path(__file__).resolve().parents[2]
    / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets"
)


def test_coupled_force_area_uses_the_resolved_span_and_native_cylinder_diameter():
    import pyvista as pv

    setup_path = PATH.parent / "setup.py"
    tree = ast.parse(setup_path.read_text())
    constants = {}
    physical_names = {
        "DIAMETER",
        "CYLINDER_LENGTH",
        "FVM_RESOLVED_SPAN",
        "FVM_HALF_SPAN",
        "FVM_BOX",
        "REFERENCE_AREA",
        "PANEL_REFERENCE_AREA",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in physical_names:
                constants[name] = eval(
                    compile(ast.Expression(node.value), str(setup_path), "eval"),
                    {"__builtins__": {}},
                    constants,
                )
    surface = pv.read(PATH / "cylinder_long.stl")
    xmin, xmax, _, _, zmin, zmax = surface.bounds
    diameter = xmax - xmin
    resolved_span = constants["FVM_BOX"][5] - constants["FVM_BOX"][4]
    assert resolved_span == pytest.approx(constants["FVM_RESOLVED_SPAN"])
    assert constants["REFERENCE_AREA"] == pytest.approx(diameter * resolved_span)
    assert constants["PANEL_REFERENCE_AREA"] == pytest.approx(diameter * (zmax - zmin))
    assert resolved_span < zmax - zmin
    assert constants["REFERENCE_AREA"] != pytest.approx(constants["PANEL_REFERENCE_AREA"])

    force_samplers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "ForceSampler"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "fvm"
    ]
    assert len(force_samplers) == 1
    area = next(
        keyword.value for keyword in force_samplers[0].keywords if keyword.arg == "reference_area"
    )
    assert isinstance(area, ast.Name) and area.id == "REFERENCE_AREA"


def test_history_uses_only_shared_time_interval_and_preserves_coefficient_scale(tmp_path):
    trial, reference = tmp_path / "trial", tmp_path / "reference"
    trial.mkdir()
    reference.mkdir()
    # Integer ticks avoid introducing a tolerance or time interpolation into the test.
    for directory, ticks, offset in (
        (reference, np.arange(101), 0.0),
        (trial, np.arange(25, 126), 0.125),
    ):
        times = ticks / 100
        pd.DataFrame(
            {
                "time": times,
                "drag_coefficient": 1 + 2 * times + offset,
                "lift_coefficient": np.sin(times),
            }
        ).to_csv(directory / "forces_history.csv", index=False)
    result = comparison.force_comparison(trial, reference, 0, 2)
    assert result["window"] == [0.25, 1.0]
    assert result["common_samples"] == 76
    assert result["instantaneous_cd_difference_rms"] == pytest.approx(0.125)
    assert result["trial"]["mean_drag"] - result["reference"]["mean_drag"] == pytest.approx(0.125)


def test_comparison_uses_common_snapshot_and_reports_missing_curves(tmp_path):
    samples, reference = tmp_path / "samples", tmp_path / "reference"
    samples.mkdir()
    reference.mkdir()
    forces = pd.DataFrame(
        {"time": [0, 1, 2], "drag_coefficient": [1, 1, 1], "lift_coefficient": [0, 0.1, 0]}
    )
    forces.to_csv(samples / "forces_history.csv", index=False)
    forces.to_csv(reference / "forces_history.csv", index=False)
    records = pd.DataFrame(
        [
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
    )
    records.to_csv(reference / "transverse_x1.csv", index=False)
    candidate = records[records.time <= 1].copy()
    candidate.velocity_y += 0.125
    candidate.to_csv(samples / "fvm_transverse_x1.csv", index=False)
    metrics = comparison.profile_comparison(samples, reference, "transverse_x1", 0, 2)
    assert metrics["time_window"] == [0, 1]
    assert metrics["mean_velocity_rms_Uinf"] == pytest.approx(0.125)
    with pytest.raises(FileNotFoundError):
        comparison.profile_comparison(samples, reference, "transverse_x1", 0, 2, prefix="vpm_")
    report = comparison.plot_comparison(tmp_path, reference, tmp_path / "figures/comparison.png")
    assert report["snapshot_time"] == 1
    assert report["available_profiles"]["transverse_x1"]
    assert not report["available_profiles"]["transverse_x2"]
    assert not report["available_profiles"]["transverse_x4"]
    assert report["startup"] and not report["qualification_changed"]
    assert not report["phase_adjustment"] and not report["time_interpolation"]
    assert Path(report["path"]).is_file()
    assert all(Path(path).is_file() for path in report["companion_paths"])
