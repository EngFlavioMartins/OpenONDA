"""Restart output history reconciliation keeps interrupted branches inspectable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from source.coupler.backup import _rewind_coupler_diagnostics
from source.solvers.fvm.io.solver_io import SolverIO
from source.solvers.vpm.io.sampler import OutputManager


def _jsonl(path: Path, records: list[object]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_coupler_diagnostics_rewind_archives_future_and_partial_records(tmp_path):
    path = tmp_path / "solution" / "coupler_diagnostics.jsonl"
    path.parent.mkdir()
    _jsonl(path, [{"step": 1, "time": 0.1}, {"step": 2, "time": 0.2}])
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"step": 3, "time": 0.3}\n{"step": 4')

    history = [
        {"step": 1, "time": 0.1},
        {"step": 2, "time": 0.2},
        {"step": 3, "time": 0.3},
    ]
    _rewind_coupler_diagnostics(path, 0.2, history)

    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [1, 2]
    assert [record["step"] for record in history] == [1, 2]
    archives = list(
        (path.parent / "restart-branches").glob("before-*/coupler_diagnostics.jsonl.superseded")
    )
    assert len(archives) == 1
    assert '"step": 4' in archives[0].read_text()


def test_coupler_diagnostics_rewind_normalizes_complete_final_record(tmp_path):
    path = tmp_path / "coupler_diagnostics.jsonl"
    path.write_text('{"step": 1, "time": 0.1}', encoding="utf-8")

    _rewind_coupler_diagnostics(path, 0.1)

    assert path.read_text().endswith("\n")
    archives = list(
        (tmp_path / "restart-branches").glob("before-*/coupler_diagnostics.jsonl.superseded")
    )
    assert len(archives) == 1


def test_fvm_jsonl_rewind_archives_superseded_history_and_recovers_partial_tail(tmp_path):
    path = tmp_path / "performance.jsonl"
    _jsonl(path, [{"time": 0.1}, {"time": 0.2}, {"time": 0.3}])
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"time": 0.4')

    SolverIO._rewind_jsonl(path, 0.2)

    assert [json.loads(line)["time"] for line in path.read_text().splitlines()] == [0.1, 0.2]
    archives = list((tmp_path / "restart-branches").glob("before-*/performance.jsonl.superseded"))
    assert len(archives) == 1
    assert '"time": 0.4' in archives[0].read_text()


def test_coupler_diagnostics_rewind_rejects_malformed_interior_record(tmp_path):
    path = tmp_path / "coupler_diagnostics.jsonl"
    path.write_text('{"time": 0.1}\nnot-json\n{"time": 0.2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid record"):
        _rewind_coupler_diagnostics(path, 0.1)


def test_rewind_preserves_superseded_partitioned_frames_before_replay(tmp_path):
    directory = tmp_path / "solution"
    frames = directory / "fvm"
    frames.mkdir(parents=True)
    (frames / "flow_000001.vtu").write_text("kept")
    (frames / "flow_000002-rank-00000.vtu").write_text("old rank frame")
    (frames / "flow_000002.pvtu").write_text(
        '<VTKFile><PUnstructuredGrid><Piece Source="flow_000002-rank-00000.vtu"/>'
        "</PUnstructuredGrid></VTKFile>"
    )
    index = directory / "fvm.pvd"
    index.write_text(
        "<VTKFile><Collection>"
        '<DataSet timestep="0.1" file="fvm/flow_000001.vtu"/>'
        '<DataSet timestep="0.2" file="fvm/flow_000002.pvtu"/>'
        "</Collection></VTKFile>"
    )

    SolverIO._rewind_pvd(index, 0.1)

    assert "flow_000002" not in index.read_text()
    assert (frames / "flow_000001.vtu").read_text() == "kept"
    assert not (frames / "flow_000002.pvtu").exists()
    branch = next((directory / "restart-branches").glob("before-*"))
    assert (branch / "fvm/flow_000002.pvtu").is_file()
    assert (branch / "fvm/flow_000002-rank-00000.vtu").read_text() == "old rank frame"


@pytest.mark.parametrize("payload", ["[1, 2]\n", '{"time": NaN}\n'])
def test_coupler_diagnostics_rewind_rejects_invalid_final_record(tmp_path, payload):
    path = tmp_path / "coupler_diagnostics.jsonl"
    path.write_text('{"time": 0.1}\n' + payload, encoding="utf-8")

    with pytest.raises(ValueError):
        _rewind_coupler_diagnostics(path, 0.1)


@pytest.mark.parametrize("rewinder", [SolverIO._rewind_csv, OutputManager._rewind_csv])
def test_csv_rewind_discards_only_an_unterminated_short_final_row(tmp_path, rewinder):
    path = tmp_path / "history.csv"
    path.write_text("time,value\n0.1,1\n0.2", encoding="utf-8")

    rewinder(path, 0.2)

    assert path.read_text() == "time,value\n0.1,1\n"


@pytest.mark.parametrize("rewinder", [SolverIO._rewind_csv, OutputManager._rewind_csv])
def test_csv_rewind_normalizes_complete_unterminated_final_row(tmp_path, rewinder):
    path = tmp_path / "history.csv"
    path.write_text("time,value\n0.1,1\n0.2,2", encoding="utf-8")

    rewinder(path, 0.2)

    assert path.read_text().endswith("0.2,2\n")


@pytest.mark.parametrize("rewinder", [SolverIO._rewind_csv, OutputManager._rewind_csv])
@pytest.mark.parametrize("row", ["0.2", "not-a-time,2", "nan,2"])
def test_csv_rewind_rejects_complete_or_interior_malformed_final_rows(tmp_path, rewinder, row):
    path = tmp_path / "history.csv"
    path.write_text(f"time,value\n0.1,1\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        rewinder(path, 0.2)
