"""Distribution contracts for installed, user-owned tutorial workspaces."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from openonda.cli import main
import openonda.tutorials as tutorial_api
from openonda.tutorials import (
    TUTORIALS,
    execute_tutorial,
    materialize_tutorial,
    tutorial_case_path,
)


def test_catalog_has_every_maintained_launcher() -> None:
    maintained = {
        str(path.parent.relative_to("tutorials")) for path in Path("tutorials").rglob("allrun.sh")
    }
    assert {tutorial.name for tutorial in TUTORIALS} == maintained


def test_materialized_lamb_oseen_case_is_self_contained(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    case_path = materialize_tutorial("vpm/lamb-oseen-vortex", workspace)

    assert case_path == tutorial_case_path(workspace, "vpm/lamb_oseen_vortex")
    assert (case_path / "setup.py").is_file()
    assert (case_path / "allrun.sh").stat().st_mode & 0o100
    assert (case_path / "assets/postprocess.py").is_file()
    assert (workspace / "tutorials/__init__.py").is_file()
    assert (workspace / "tutorials/vpm/__init__.py").is_file()
    assert (workspace / "docs/themes/matplotlib_setup.py").is_file()
    assert (workspace / "docs/themes/DejaVuSerif.ttf").is_file()

    assert not (case_path / "samples").exists()
    assert not (case_path / "solution").exists()
    assert not (case_path / "assets/schematics.pvsm").exists()
    assert not list(case_path.rglob("paraview_state.py"))
    assert not list(case_path.rglob("paraview_tracer.py"))
    assert not (
        case_path / "assets/references/the-physical-mechanism-for-vortex-merging.pdf"
    ).exists()


def test_materializer_never_overwrites_existing_case(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    materialize_tutorial("fvm/taylor_green", workspace)
    marker = tutorial_case_path(workspace, "fvm/taylor_green") / "user-change.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        materialize_tutorial("fvm/taylor_green", workspace)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_lists_tutorials_and_renders_api_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tutorial", "list"]) == 0
    assert "vpm/lamb_oseen_vortex" in capsys.readouterr().out

    assert main(["api", "tutorials.materialize_tutorial"]) == 0
    output = capsys.readouterr().out
    assert "materialize_tutorial" in output
    assert "user-owned workspace" in output


def test_launcher_uses_the_console_scripts_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    materialize_tutorial("fvm/taylor_green", workspace)
    captured: dict[str, object] = {}

    class Result:
        returncode = 0

    def fake_run(command, *, cwd, env, check):
        captured.update(command=command, cwd=cwd, env=env, check=check)
        return Result()

    monkeypatch.setattr(tutorial_api.subprocess, "run", fake_run)
    assert execute_tutorial("fvm/taylor_green", workspace) == 0

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["OPENONDA_PYTHON"] == sys.executable
    assert environment["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert environment["MPLCONFIGDIR"] == str(workspace / ".matplotlib")
    assert environment["XDG_CACHE_HOME"] == str(workspace / ".cache")
    assert environment["TI_OFFLINE_CACHE_FILE_PATH"] == str(workspace / ".cache/taichi")
