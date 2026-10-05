"""Native checkpoints govern continuation independently of visualization history."""

import json

import pytest

from source.restart import archive_run_metadata, reset_run_outputs, select_backup


def test_coupled_latest_retries_metadata_only_at_zero(tmp_path):
    path = tmp_path / "fvm_metadata.json"
    path.write_text(json.dumps({"state": {"step": 0, "time": 0.0}}))
    archive_run_metadata(path)
    (tmp_path / "fvm.pvd").write_text(
        '<VTKFile><Collection><DataSet timestep="0"/></Collection></VTKFile>'
    )
    assert select_backup("latest", directory=tmp_path, kind="coupled") is None
    assert select_backup("initial", directory=tmp_path, kind="coupled") is None


@pytest.mark.parametrize("kind", ["fvm", "vpm", "coupled"])
def test_initial_ignores_even_invalid_native_backups(tmp_path, kind):
    (tmp_path / "vpm").mkdir()
    (tmp_path / "vpm/vpm_999999.h5").write_bytes(b"not HDF5")
    (tmp_path / "backups").mkdir()
    (tmp_path / "backups/checkpoint_info.json").write_bytes(b"not JSON")
    (tmp_path / "backup").write_bytes(b"not a native state")
    assert select_backup("initial", directory=tmp_path, kind=kind, backup_path="backup") is None


def test_initial_output_reset_retires_all_vpm_candidates_and_preserves_other_files(tmp_path):
    solution = tmp_path / "solution"
    (solution / "vpm").mkdir(parents=True)
    samples = tmp_path / "samples"
    samples.mkdir()
    for relative in ("vpm/vpm_000012.h5", "vpm_000050.h5", "vpm.pvd", "vlm_000012.vtu"):
        (solution / relative).write_text("old or interrupted data")
    for name in ("forces.csv", "forces.pvd", "unrelated.csv"):
        (samples / name).write_text("old data")
    (solution / "vpm_metadata.json").write_text("current constructor metadata")
    (solution / "vpm.log").write_text("open log")
    reset_run_outputs(solution, kind="vpm", samples_dir=samples, sample_names=["forces"])
    assert select_backup("latest", directory=solution, kind="vpm") is None
    assert (samples / "unrelated.csv").read_text() == "old data"
    assert not (samples / "forces.csv").exists()
    assert (solution / "vpm_metadata.json").read_text() == "current constructor metadata"
    assert (solution / "vpm.log").read_text() == "open log"
    assert list((solution / "restart_history").glob("initial-before-*/solution/vpm/vpm_000012.h5"))


@pytest.mark.parametrize(
    "metadata",
    [
        {"state": {"step": 1, "time": 0.01}},
        {"state": {"step": 0, "time": 0.01}},
        {"state": {}},
        {},
    ],
)
def test_latest_without_checkpoint_ignores_archived_metadata(tmp_path, metadata):
    path = tmp_path / "fvm_metadata.json"
    path.write_text(json.dumps(metadata))
    archive_run_metadata(path)
    assert select_backup("latest", directory=tmp_path, kind="coupled") is None


def test_repeated_failed_initialization_ignores_requested_stop_in_coupler_metadata(tmp_path):
    path = tmp_path / "run_metadata.json"
    path.write_text(
        json.dumps(
            {
                "execution": {
                    "start_coupling_step": 0,
                    "start_time": 0.0,
                    "stop_coupling_step": 600,
                }
            }
        )
    )
    archive_run_metadata(path)
    assert select_backup("latest", directory=tmp_path, kind="coupled") is None


@pytest.mark.parametrize(
    "filename,content",
    [
        ("fvm.pvd", '<VTKFile><Collection><DataSet timestep="0.01"/></Collection></VTKFile>'),
        ("coupler_diagnostics.jsonl", '{"step": 1, "time": 0.01}\n'),
        ("coupler_diagnostics.jsonl", '{"step":'),
    ],
)
def test_latest_without_checkpoint_ignores_output_history(tmp_path, filename, content):
    path = tmp_path / "fvm_metadata.json"
    path.write_text(json.dumps({"state": {"step": 0, "time": 0.0}}))
    archive_run_metadata(path)
    (tmp_path / filename).write_text(content)
    assert select_backup("latest", directory=tmp_path, kind="coupled") is None


@pytest.mark.parametrize("kind", ["fvm", "vpm", "coupled"])
def test_none_preserves_in_memory_selection_despite_corrupt_output(tmp_path, kind):
    (tmp_path / "backup").write_text("corrupt")
    (tmp_path / "backups").mkdir()
    (tmp_path / "backups/checkpoint_info.json").write_text("corrupt")
    (tmp_path / "vpm").mkdir()
    (tmp_path / "vpm/vpm_000001.h5").write_text("corrupt")
    assert select_backup(None, directory=tmp_path, kind=kind, backup_path="backup") is None


@pytest.mark.parametrize("kind", ["fvm", "coupled"])
def test_latest_selects_present_corrupt_checkpoint_for_strict_reader(tmp_path, kind):
    backup = tmp_path / "backup"
    backup.write_text("corrupt")
    bundle = tmp_path / "backups"
    bundle.mkdir()
    (bundle / "checkpoint_info.json").write_text("corrupt")
    assert select_backup("latest", directory=tmp_path, kind=kind, backup_path="backup") == (
        backup if kind == "fvm" else bundle
    )
