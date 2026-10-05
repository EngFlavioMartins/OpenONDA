"""Parameter-study plotting never silently falls back to stale successful results."""

import json
import os
from pathlib import Path

import pytest

from openonda.tutorial_runner import load_case_module


def test_latest_incomplete_parameter_study_requires_explicit_older_selection(tmp_path):
    assets = Path(__file__).resolve().parents[2] / "tests/support/cylinder"
    module = load_case_module(assets, "plot_parameter_study")
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    complete = {"reference": {"grids": []}, "runs": [{"kind": "reference"}]}
    old_checkpoint_info = old / "solver_comparison.json"
    old_checkpoint_info.write_text(json.dumps(complete))
    os.utime(old_checkpoint_info, (1, 1))
    latest = new / "solver_comparison.json"
    latest.write_text(json.dumps({"pilot": True}))
    with pytest.raises(ValueError, match="no production grid report"):
        module.parameter_study_report(tmp_path, None)
    assert module.parameter_study_report(tmp_path, old) == (old, complete)
    latest.write_text(json.dumps({**complete, "runs": [{"kind": "coupled"}]}))
    with pytest.raises(ValueError, match="incomplete coupled comparison"):
        module.parameter_study_report(tmp_path, None)
