"""Checkpoint rotation must never discard the last committed restart on failure."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.fvm.io import backup, partitioned


@pytest.fixture
def solver(monkeypatch):
    monkeypatch.setattr(backup, "_solver_setup", lambda solver: None)
    monkeypatch.setattr(backup, "config_hash", lambda setup: "test-config")
    monkeypatch.setattr(partitioned, "require_free_space", lambda *args: None)
    parallel = SimpleNamespace(
        is_root=True,
        rank=0,
        bcast=lambda value, root: value,
        global_sum=lambda value: value,
        comm=SimpleNamespace(allgather=lambda value: [value]),
        partition=SimpleNamespace(local_global_ids=np.arange(2), size=1, rank=0, n_global_cells=2),
    )
    result = SimpleNamespace(
        parallel=parallel,
        mesh_data={"global_face_id": np.arange(3), "global_mesh_hash": "mesh"},
        _kinematic_viscosity=0.001,
        eddy_viscosity=None,
        time=0.0,
        step=0,
        _n_committed_time_steps=0,
        time_step_size=0.01,
        _accepted_time_step_size=0.01,
        _previous_time_step_size=0.01,
        max_courant_number=0.1,
        _n_consecutive_accepted_steps={},
    )
    for key in (
        "velocity",
        "velocity_old",
        "velocity_older",
        "kinematic_pressure",
        "volumetric_face_flux",
        "volumetric_face_flux_old",
        "volumetric_face_flux_older",
    ):
        setattr(result, key, np.ones(2))
    # The writer's getattr default is eagerly evaluated.
    monkeypatch.setattr(
        backup,
        "_solver_setup",
        lambda solver: SimpleNamespace(transport=SimpleNamespace(kinematic_viscosity=0.001)),
    )
    return result


def test_writer_keeps_two_committed_generations_and_unrelated_files(tmp_path, solver):
    unrelated = tmp_path / "rank-00000-legacy.npz"
    unrelated.write_text("legacy data")
    generations = []
    for step in range(4):
        solver.step = step
        partitioned.save_partitioned_solver_backup(solver, tmp_path)
        generations.append(json.loads((tmp_path / "manifest.json").read_text()))
    assert json.loads((tmp_path / "manifest.previous.json").read_text()) == generations[-2]
    for manifest in generations[-2:]:
        assert all((tmp_path / name).is_file() for name in manifest["files"])
    for manifest in generations[:-2]:
        assert all(not (tmp_path / name).exists() for name in manifest["files"])
    assert unrelated.read_text() == "legacy data"


def test_failed_archive_preserves_current_and_previous(tmp_path, solver, monkeypatch):
    for _ in range(2):
        partitioned.save_partitioned_solver_backup(solver, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def fail(*args):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(partitioned, "_atomic_npz", fail)
    with pytest.raises(RuntimeError, match="simulated storage failure"):
        partitioned.save_partitioned_solver_backup(solver, tmp_path)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_failed_manifest_commit_never_prunes_committed_generations(tmp_path, solver, monkeypatch):
    for _ in range(2):
        partitioned.save_partitioned_solver_backup(solver, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    replace = partitioned.os.replace

    def fail_manifest(source, destination):
        if str(destination).endswith("/manifest.json"):
            raise OSError("simulated manifest publication failure")
        return replace(source, destination)

    monkeypatch.setattr(partitioned.os, "replace", fail_manifest)
    with pytest.raises(RuntimeError, match="simulated manifest publication failure"):
        partitioned.save_partitioned_solver_backup(solver, tmp_path)
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
