"""Bounded asynchronous FVM output behavior."""

import numpy as np
import pytest

from source.solvers.fvm.io import async_output


class _Exporter:
    writes = []

    def __init__(self, mesh_data, output=None):
        self.mesh_data = mesh_data
        self.output = output

    def export(self, filename, fields):
        self.writes.append((filename, fields["kinematic_pressure"].copy()))


class _PVD:
    entries = []

    def __init__(self, filename):
        self.filename = filename

    def add_step(self, time, filename):
        self.entries.append((time, filename))


def test_buffered_writer_snapshots_and_flushes(monkeypatch, tmp_path):
    monkeypatch.setattr(async_output, "VTKExporter", _Exporter)
    monkeypatch.setattr(async_output, "PVDManager", _PVD)
    _Exporter.writes.clear()
    _PVD.entries.clear()
    values = np.array([1.0, 2.0])
    writer = async_output.BufferedVTKWriter({}, str(tmp_path / "case.pvd"))
    writer.submit(str(tmp_path / "one.vtu"), 0.1, {"kinematic_pressure": values})
    values[:] = -1.0
    writer.flush()
    writer.close()

    np.testing.assert_array_equal(_Exporter.writes[0][1], [1.0, 2.0])
    assert _PVD.entries == [(0.1, str(tmp_path / "one.vtu"))]


def test_buffered_writer_propagates_background_failure(monkeypatch, tmp_path):
    class _FailingExporter(_Exporter):
        def export(self, filename, fields):
            raise OSError("disk full")

    monkeypatch.setattr(async_output, "VTKExporter", _FailingExporter)
    writer = async_output.BufferedVTKWriter({}, str(tmp_path / "case.pvd"))
    writer.submit(
        str(tmp_path / "one.vtu"),
        0.1,
        {"kinematic_pressure": np.ones(1)},
    )
    with pytest.raises(OSError, match="disk full"):
        writer.close()
