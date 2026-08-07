"""Rewinding sampler histories on restart (blocker: line/surface + PVD).

``SolverIO.rewind_histories`` must strip rows/frames past the resumed ``flow_time``
from *every* sampler product, not just the classic ``forces_history`` / IBM CSVs:
line samplers write a ``flow_time`` time column (not ``time``), and surface
samplers write a ``.pvd`` index alongside their per-step ``.vts`` files.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from source.solvers.FVM.io.solver_io import SolverIO


@pytest.fixture
def io(tmp_path):
    fake_solver = SimpleNamespace(case_dir=str(tmp_path), parallel=None)
    return SolverIO(fake_solver)


def _write(tmp_path, relpath, text):
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_rewind_trims_flow_time_line_csv(io, tmp_path):
    path = _write(
        tmp_path,
        "samples/centerline.csv",
        "flow_time,time_step,x\n0.00,0,0.5\n0.02,2,0.5\n0.04,4,0.5\n0.06,6,0.5\n",
    )
    io.rewind_histories(0.04)
    assert (
        path.read_text(encoding="utf-8")
        == "flow_time,time_step,x\n0.00,0,0.5\n0.02,2,0.5\n0.04,4,0.5\n"
    )


def test_rewind_trims_time_column_forces_csv(io, tmp_path):
    path = _write(
        tmp_path,
        "samples/forces_history.csv",
        "time,step,Cd\n1.0,1,0.5\n2.0,2,0.6\n3.0,3,0.7\n",
    )
    io.rewind_histories(2.0)
    assert path.read_text(encoding="utf-8") == "time,step,Cd\n1.0,1,0.5\n2.0,2,0.6\n"


def test_rewind_trims_surface_pvd_frames(io, tmp_path):
    path = _write(
        tmp_path,
        "samples/slice.pvd",
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" byte_order="LittleEndian">\n'
        "  <Collection>\n"
        '    <DataSet timestep="1.0" file="slice_000001.vts"/>\n'
        '    <DataSet timestep="2.0" file="slice_000002.vts"/>\n'
        '    <DataSet timestep="3.0" file="slice_000003.vts"/>\n'
        "  </Collection>\n"
        "</VTKFile>\n",
    )
    io.rewind_histories(2.0)
    text = path.read_text(encoding="utf-8")
    assert "slice_000001.vts" in text
    assert "slice_000002.vts" in text
    assert "slice_000003.vts" not in text


def test_rewind_ignores_csv_without_time_column(io, tmp_path):
    path = _write(
        tmp_path,
        "samples/other.csv",
        "x,y\n0.5,1.0\n0.6,2.0\n",
    )
    io.rewind_histories(0.0)
    assert path.read_text(encoding="utf-8") == "x,y\n0.5,1.0\n0.6,2.0\n"
