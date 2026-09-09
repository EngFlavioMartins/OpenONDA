"""Plotters select solver fields by recorded time, never by filename order."""

import json

from openonda.plotting import latest_fvm_snapshot


def test_latest_snapshot_uses_solver_metadata_and_physical_time(tmp_path):
    (tmp_path / "fvm_metadata.json").write_text(json.dumps({"case_name": "edited_case"}))
    (tmp_path / "edited_case.pvd").write_text(
        "<VTKFile><Collection>"
        '<DataSet timestep="2.0" file="a_final.vtu"/>'
        '<DataSet timestep="0.0" file="z_initial.vtu"/>'
        "</Collection></VTKFile>"
    )
    for name in ("a_final.vtu", "z_initial.vtu", "mesh.vtu"):
        (tmp_path / name).touch()
    assert latest_fvm_snapshot(tmp_path) == tmp_path / "a_final.vtu"


def test_no_field_series_does_not_select_geometry(tmp_path):
    (tmp_path / "mesh.vtu").touch()
    assert latest_fvm_snapshot(tmp_path) is None
