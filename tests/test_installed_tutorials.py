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

    return load_case_module(Path(files("tutorials")) / "vpm/lamb_oseen_vortex")


def test_catalog_has_every_maintained_launcher() -> None:
    from importlib.resources import files

    root = Path(files("tutorials"))
    maintained = {str(path.parent.relative_to(root)) for path in root.rglob("allrun.sh")}
    assert {tutorial.name for tutorial in TUTORIALS} == maintained


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


def test_lamb_oseen_reuses_results_only_for_the_selected_backend_and_stretching(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    setup = _load_lamb_oseen_setup()
    monkeypatch.setattr(setup, "TUTORIAL_DIR", tmp_path)
    name = "vortex_cs"
    folder = tmp_path / "samples" / name
    folder.mkdir(parents=True)
    (folder / "flow_integrals.csv").touch()
    (folder / f"{name}_zq.pvd").touch()
    solution = tmp_path / "solution" / name
    solution.mkdir(parents=True)
    steps = round(setup.TOTAL_TIME / setup.TIME_STEP_SIZE)
    (solution / f"vpm_{steps:06d}.h5").touch()

    for backend, stretching in (
        ("DIRECT", "transposed"),
        ("DIRECT", "mixed"),
        ("FMM", "mixed"),
        ("TREECODE", "direct"),
    ):
        monkeypatch.setitem(setup.COMPUTE_METHOD, "CS", backend)
        monkeypatch.setattr(setup, "STRETCHING_SCHEME", stretching)
        assert not setup.completed_run_matches("vortex", "CS", name, 42)
        solver = SimpleNamespace(
            induction=setup.induction_config("CS"),
            integrator_tableau=SimpleNamespace(name="RK2", order=2, stages=2),
            time=setup.TOTAL_TIME,
            particles=SimpleNamespace(n_particles_total=0),
        )
        setup.write_run_metadata(
            physics="vortex",
            scheme="CS",
            sample_directory=name,
            circulations=setup.PHYSICS_CIRCULATIONS["vortex"],
            kinematic_viscosity=1.0 / setup.CIRCULATION_REYNOLDS_NUMBER,
            spacing=setup.SPACING,
            particle_core_radius=setup.PARTICLE_RADIUS,
            field_spacing=setup.FIELD_SPACING,
            n_steps=steps,
            random_seed=42,
            initial_n_particles_total=0,
            solver=solver,
        )
        assert setup.completed_run_matches("vortex", "CS", name, 42)


def test_all_vpm_tutorials_construct_cases_with_the_installed_api(tmp_path, monkeypatch):
    from openonda.tutorial_runner import load_case_module

    class CaseCapturedError(Exception):
        def __init__(self, case):
            self.case = case

    def capture(case):
        raise CaseCapturedError(case)

    builders = {
        "delta_wing": lambda setup: setup.run(),
        "quadcopter": lambda setup: setup.run(),
        "flat_plate": lambda setup: setup.run("static", 8),
        "rotor_flow": lambda setup: setup.build_rotor_case(steps=1, directory=setup.TUTORIAL_DIR),
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
