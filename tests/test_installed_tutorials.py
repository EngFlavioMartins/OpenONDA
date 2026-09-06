"""Tests for installed, user-owned tutorial workspaces."""

from __future__ import annotations

import importlib.util
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


def _load_lamb_oseen_setup():
    path = Path("tutorials/vpm/lamb_oseen_vortex/setup.py")
    spec = importlib.util.spec_from_file_location("lamb_oseen_setup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert (case_path / "README.md").is_file()
    assert (case_path / "allrun.sh").stat().st_mode & 0o100
    assert (case_path / "allplot.sh").stat().st_mode & 0o100
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

    # `openonda tutorial` passes the console command's interpreter through this
    # variable. The copied launchers must retain that handoff rather than
    # accidentally selecting another `python` on the user's PATH.
    for launcher in (case_path / "allrun.sh", case_path / "allplot.sh"):
        contents = launcher.read_text(encoding="utf-8")
        assert 'PYTHON_BIN="${OPENONDA_PYTHON:-python}"' in contents
        assert '"${PYTHON_BIN}" -m' in contents

    allrun = (case_path / "allrun.sh").read_text(encoding="utf-8")
    assert 'mktemp -d "${CACHE_PARENT%/}/lamb-oseen.XXXXXX"' in allrun
    assert "run_physics_case vortex" in allrun
    assert "run_physics_case dipole" in allrun
    assert "run_physics_case merging" in allrun
    assert '--aggregate-rwm-case "${physics}"' in allrun
    assert '--validate-case "${physics}"' in allrun
    assert "--induction" not in allrun
    assert " CS DIRECT" not in allrun
    assert " DVH TREECODE" not in allrun
    assert " GBD TREECODE" not in allrun

    allplot = (case_path / "allplot.sh").read_text(encoding="utf-8")
    assert "MPLCONFIGDIR:-${SCRIPT_DIR}/.cache/matplotlib" in allplot
    assert allplot.rstrip().endswith('"${PYTHON_BIN}" -m "${MODULE}.assets.postprocess"')


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


def test_lamb_oseen_initial_core_radius_matches_regeneration_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _load_lamb_oseen_setup()

    observed: dict[str, float] = {}

    class DistributionReachedError(Exception):
        pass

    def capture_distribution(*, core_radius_ratio: float, **_kwargs):
        observed["core_radius_ratio"] = core_radius_ratio
        raise DistributionReachedError

    monkeypatch.setattr(setup.vpm, "TriangularPrismDistribution", capture_distribution)
    with pytest.raises(DistributionReachedError):
        setup.run_case("vortex", "GBD")

    assert observed["core_radius_ratio"] == pytest.approx(setup.CORE_RADIUS_RATIO)
    assert pytest.approx(setup.PARTICLE_RADIUS) == setup.CORE_RADIUS_RATIO * setup.SPACING


def test_lamb_oseen_numerical_setup() -> None:
    setup = _load_lamb_oseen_setup()

    assert pytest.approx(0.60) == setup.SPACING / setup.CORE_RADIUS
    assert setup.RWM_ENSEMBLE_SIZE == 10
    assert setup.induction_config("CS").method == "DIRECT"
    assert setup.induction_config("RWM").method == "DIRECT"
    assert setup.induction_config("DVH").method == "TREECODE"
    assert setup.induction_config("GBD").method == "TREECODE"


@pytest.mark.parametrize("physics", ["vortex", "dipole", "merging"])
def test_lamb_oseen_workspace_reserves_full_time_diffusion(physics, monkeypatch):
    setup = _load_lamb_oseen_setup()
    captured = []

    class CaseCapturedError(Exception):
        pass

    def capture(case):
        captured.append(case)
        raise CaseCapturedError

    monkeypatch.setattr(setup.vpm, "VPMSolver", capture)
    with pytest.raises(CaseCapturedError):
        setup.run_case(physics, "DVH", surfaces=False)
    case = captured[0]
    viscosity = case.numerics.viscous.kinematic_viscosity
    heat_margin = 3.6 * (4.0 * viscosity * setup.TOTAL_TIME) ** 0.5
    xmin, xmax, ymin, ymax, zmin, zmax = case.numerics.domain_bounds
    for condition in case.initial_conditions:
        position = condition.build().position
        for axis, (lower, upper) in enumerate(((xmin, xmax), (ymin, ymax), (zmin, zmax))):
            assert lower <= position[:, axis].min() - heat_margin
            assert upper >= position[:, axis].max() + heat_margin
