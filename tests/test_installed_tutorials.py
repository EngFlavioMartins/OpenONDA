"""Tests for installed, user-owned tutorial workspaces."""

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


def _load_lamb_oseen_setup():
    from importlib.resources import files

    from openonda.tutorial_runner import load_case_module

    return load_case_module(Path(files("tutorials")) / "vpm/01_lamb_oseen_vortex")


def test_catalog_has_every_maintained_launcher() -> None:
    from importlib.resources import files

    root = Path(files("tutorials"))
    maintained = {str(path.parent.relative_to(root)) for path in root.rglob("allrun.sh")}
    # Retained experiments can be reproduced, but are not public tutorials.
    maintained.discard("vpm/07_quadcopter_PENDING/studies")
    assert {tutorial.relative_path.as_posix() for tutorial in TUTORIALS} == maintained


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


@pytest.mark.parametrize("custom_taichi_cache", [False, True])
def test_launcher_uses_the_console_scripts_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_taichi_cache: bool
) -> None:
    workspace = tmp_path / "workspace"
    expected_cache = workspace / ".cache/taichi"
    monkeypatch.delenv("TI_OFFLINE_CACHE_FILE_PATH", raising=False)
    if custom_taichi_cache:
        expected_cache = tmp_path / "custom Taichi cache"
        monkeypatch.setenv("TI_OFFLINE_CACHE_FILE_PATH", str(expected_cache))
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
    assert environment["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert environment["MPLCONFIGDIR"] == str(workspace / ".matplotlib")
    assert environment["XDG_CACHE_HOME"] == str(workspace / ".cache")
    assert environment["TI_OFFLINE_CACHE_FILE_PATH"] == str(expected_cache)


@pytest.mark.parametrize("physics", ["vortex", "dipole", "merging"])
def test_lamb_oseen_workspace_reserves_full_time_diffusion(physics, monkeypatch, tmp_path):
    setup = _load_lamb_oseen_setup()
    monkeypatch.setattr(setup, "TUTORIAL_DIR", tmp_path)
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


def test_every_template_materializes_without_generated_results(tmp_path):
    """One catalog-wide contract replaces case-specific file/string snapshots."""
    import subprocess

    for tutorial in TUTORIALS:
        case = materialize_tutorial(tutorial.name, tmp_path / tutorial.slug)
        assert (case / "setup.py").is_file(), tutorial.name
        assert (case / "allrun.sh").is_file(), tutorial.name
        for script in case.rglob("*.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
        for path in case.rglob("*"):
            assert not set(path.relative_to(case).parts) & {
                "solution",
                "solutions",
                "samples",
                "figures",
                "study_results",
                "__pycache__",
            }, path


def test_local_module_runner_uses_edited_case_and_propagates_exit_code(tmp_path):
    import subprocess

    case = tmp_path / "case with spaces"
    (case / "assets").mkdir(parents=True)
    (case / "settings.py").write_text("VALUE = 73\n")
    (case / "assets/check.py").write_text(
        "from ..settings import VALUE\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(__file__).with_suffix('.txt').write_text(str(VALUE))\n"
        "raise SystemExit(int(sys.argv[1]))\n"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-m", "openonda.tutorial_runner", str(case), "assets.check", "23"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 23, result.stderr
    assert (case / "assets/check.txt").read_text() == "73"


def test_direct_asset_script_reads_the_edited_local_setup(tmp_path):
    import subprocess

    case = materialize_tutorial("vpm/lamb_oseen_vortex", tmp_path / "case with spaces")
    # The ensemble imports this setting from the local setup. Its --help output
    # exposes the default without launching a GPU simulation.
    setup = case / "setup.py"
    setup.write_text(setup.read_text().replace("RWM_ENSEMBLE_SIZE = 10", "RWM_ENSEMBLE_SIZE = 37"))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", str(case / "assets/rwm_ensemble.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "default: 37" in result.stdout


@pytest.mark.parametrize("script", ["plot_core_sections.py", "assess_lbm_agreement.py"])
def test_interaction_plotters_run_directly_from_a_copied_case(tmp_path, script):
    import subprocess

    case = materialize_tutorial("vpm/vortex_interactions", tmp_path / "copied case")
    result = subprocess.run(
        [sys.executable, "-I", str(case / "assets" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_cleaners_only_remove_their_own_outputs_from_an_unrelated_directory(tmp_path):
    """Use copied scripts so this test can never clean repository results."""
    from importlib.resources import files
    import subprocess

    root = Path(files("tutorials"))
    caller = tmp_path / "unrelated directory"
    caller.mkdir()
    for name in ("solution", "samples", "figures", "study_results", "constant", "assets"):
        (caller / name).mkdir()
        (caller / name / "keep.txt").write_text("keep")
    for index, original in enumerate(root.rglob("allclean.sh")):
        case = tmp_path / f"case {index}"
        case.mkdir()
        script = case / "allclean.sh"
        script.write_text(original.read_text())
        (case / "solution").mkdir()
        (case / "solution/generated.txt").write_text("generated")
        (case / "setup.py").write_text("# keep user configuration")
        subprocess.run(["bash", str(script)], cwd=caller, check=True, capture_output=True)
        assert not (case / "solution").exists(), original
        assert (case / "setup.py").is_file()
    assert all(path.read_text() == "keep" for path in caller.glob("*/keep.txt"))


@pytest.mark.parametrize(
    "backend, stretching",
    [
        ("DIRECT", "transposed"),
        ("DIRECT", "mixed"),
        ("FMM", "mixed"),
        ("TREECODE", "direct"),
    ],
)
def test_lamb_oseen_selects_backend_and_stretching_independently(monkeypatch, backend, stretching):
    setup = _load_lamb_oseen_setup()
    monkeypatch.setitem(setup.COMPUTE_METHOD, "CS", backend)
    monkeypatch.setattr(setup, "STRETCHING_SCHEME", stretching)
    runtime = setup.induction_config("CS").build()
    assert runtime.method == backend
    assert runtime.stretching_scheme.lower() == stretching


def test_all_vpm_tutorials_construct_cases_with_the_installed_api(tmp_path, monkeypatch):
    from openonda.tutorial_runner import load_case_module

    class CaseCapturedError(Exception):
        def __init__(self, case):
            self.case = case

    def capture(case):
        raise CaseCapturedError(case)

    builders = {
        "delta_wing": lambda setup: setup.vpm.VPMSolver(setup.build_case()),
        "quadcopter": lambda setup: setup.run(),
        "flat_plate": lambda setup: setup.run("static", 8),
        "rotor_flow": lambda setup: setup.run(),
        "lamb_oseen_vortex": lambda setup: setup.run_case("vortex", "CS", surfaces=False),
        "vortex_ring": lambda setup: setup.run_case("dns_mixed", n_steps=1),
        "vortex_interactions": lambda setup: setup.build_case("baseline", n_steps=1),
    }
    for name, build in builders.items():
        case_dir = materialize_tutorial(f"vpm/{name}", tmp_path / name)
        setup = load_case_module(case_dir)
        with monkeypatch.context() as patch:
            # Exercise real case/geometry construction without starting a
            # long GPU campaign or writing into installed tutorial resources.
            patch.setattr(setup.vpm, "VPMSolver", capture)
            try:
                case = build(setup)
            except CaseCapturedError as captured:
                case = captured.case
        assert case.directory == case_dir
        configured = case.numerics.induction
        runtime = configured.build()
        assert runtime.method == configured.method
        assert runtime.stretching_scheme == configured.stretching_scheme


@pytest.mark.parametrize("fail_first", [False, True])
def test_every_launcher_runs_direct_python_and_stops_on_failure(tmp_path, fail_first):
    """Execute copied launchers with an interpreter probe, never repository cleanup."""
    from importlib.resources import files
    import json
    import shutil
    import subprocess

    root = Path(files("tutorials"))
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    probe = executable_dir / "python"
    probe.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['OPENONDA_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "raise SystemExit(23 if os.environ['OPENONDA_TEST_FAIL'] == '1' else 0)\n"
    )
    probe.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(executable_dir) + os.pathsep + os.environ["PATH"],
        "OPENONDA_TEST_FAIL": str(int(fail_first)),
    }
    for index, original in enumerate(sorted(root.rglob("allrun.sh"))):
        case = tmp_path / f"case {index} with spaces"
        case.mkdir()
        launcher = case / "allrun.sh"
        shutil.copy2(original, launcher)
        output = case / "solution"
        output.mkdir()
        (output / "existing.txt").write_text("keep")
        calls_file = case / "calls.jsonl"
        result = subprocess.run(
            [str(launcher)],
            cwd=case,
            env={**environment, "OPENONDA_TEST_CALLS": str(calls_file)},
            capture_output=True,
            text=True,
        )
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
        expected_count = sum(
            line.startswith("python ") for line in original.read_text().splitlines()
        )
        assert result.returncode == (23 if fail_first else 0), (original, result.stderr)
        assert len(calls) == (1 if fail_first else expected_count), original
        assert (output / "existing.txt").read_text() == "keep"
        if original.parent.name == "01_lamb_oseen_vortex" and not fail_first:
            assert [call[:3] for call in calls] == [
                arguments
                for physics in ("vortex", "dipole", "merging")
                for arguments in (
                    ["setup.py", physics, "CS"],
                    ["assets/rwm_ensemble.py", physics, "--number-of-realizations"],
                    ["setup.py", physics, "DVH"],
                    ["setup.py", physics, "GBD"],
                )
            ]
