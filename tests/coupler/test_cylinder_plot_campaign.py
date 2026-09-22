"""Campaign plotting never silently falls back to stale successful results."""

import json
import os
from pathlib import Path

import pytest

from openonda.tutorial_runner import load_case_module


def test_latest_incomplete_campaign_requires_explicit_older_selection(tmp_path):
    assets = (
        Path(__file__).resolve().parents[2]
        / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets"
    )
    module = load_case_module(assets, "plot_campaign")
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    complete = {"reference": {"grids": []}, "runs": [{"kind": "reference"}]}
    old_manifest = old / "pipeline_manifest.json"
    old_manifest.write_text(json.dumps(complete))
    os.utime(old_manifest, (1, 1))
    latest = new / "pipeline_manifest.json"
    latest.write_text(json.dumps({"pilot": True}))
    with pytest.raises(ValueError, match="no production grid report"):
        module.campaign_report(tmp_path, None)
    assert module.campaign_report(tmp_path, old) == (old, complete)
    latest.write_text(json.dumps({**complete, "runs": [{"kind": "coupled"}]}))
    with pytest.raises(ValueError, match="incomplete coupled comparison"):
        module.campaign_report(tmp_path, None)
