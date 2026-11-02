"""The isolated test runner runs exactly the files the command line names."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

from tests import run_isolated

ROOT = Path(run_isolated.__file__).resolve().parent


def _commands(monkeypatch, arguments):
    """Run the runner's main with ``arguments`` and return the pytest commands it starts."""
    commands = []

    def record(command):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(ROOT.parent)
    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr("sys.argv", ["run_isolated.py", *arguments])
    assert run_isolated.main() == 0
    return commands


def test_named_files_and_directories_select_the_processes(monkeypatch):
    commands = _commands(
        monkeypatch,
        ["tests/fvm/test_mesh_motion.py", "tests/tutorials", "-m", "not gpu", "-x"],
    )
    targets = [command[3] for command in commands]
    tutorial_files = sorted(str(path) for path in (ROOT / "tutorials").rglob("test_*.py"))
    assert targets == [str(ROOT / "fvm/test_mesh_motion.py"), *tutorial_files]
    # Each process runs one target; the remaining arguments go to pytest unchanged
    for command in commands:
        assert command[-3:] == ["-m", "not gpu", "-x"]
        assert not any(argument.startswith("tests/") for argument in command[4:])


def test_node_ids_keep_their_test_names(monkeypatch):
    node = "tests/fvm/test_mesh_motion.py::test_does_not_exist"
    commands = _commands(monkeypatch, [node])
    assert [command[3] for command in commands] == [
        str(ROOT / "fvm/test_mesh_motion.py") + "::test_does_not_exist"
    ]


def test_without_paths_every_test_file_runs(monkeypatch):
    commands = _commands(monkeypatch, ["-m", "not gpu"])
    assert [command[3] for command in commands] == [
        str(path) for path in sorted(ROOT.rglob("test_*.py"))
    ]
