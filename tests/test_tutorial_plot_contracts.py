"""Regression contracts for tutorial post-processing after nomenclature changes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import numpy as np

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


def test_vortex_interaction_validators_use_canonical_cadence_keys():
    for relative_path in (
        "vpm/vortex_interactions/assets/check_run.py",
        "vpm/vortex_interactions/assets/validate_plot_inputs.py",
    ):
        source = (TUTORIALS / relative_path).read_text(encoding="utf-8")
        assert re.search(r'manifest\.get\("checkpoint_interval_steps"(?:,\s*0)?\)', source)
        assert 'manifest.get("snapshot_frequency")' not in source
        assert re.search(
            r"range\(\s*checkpoint_interval_steps,\s*completed_steps\s*\+\s*1",
            source,
        )
