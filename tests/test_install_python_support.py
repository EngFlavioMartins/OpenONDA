"""Python version checks and installer execution."""

from pathlib import Path
import tomllib
from types import SimpleNamespace

from packaging.specifiers import SpecifierSet
import pytest

import install
from source.version import PYTHON_REQUIRES

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_python_requirement_matches_package_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert PYTHON_REQUIRES == project["requires-python"] == "==3.11.*"


@pytest.mark.parametrize("version", ["3.11.0", "3.11.15", "3.11.99"])
def test_metadata_accepts_security_patch_updates(version):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert version in SpecifierSet(project["requires-python"])
    assert "Programming Language :: Python :: Implementation :: CPython" in project["classifiers"]


@pytest.mark.parametrize("version", ["3.10.19", "3.12.0", "3.13.0", "3.14.0", "4.0.0"])
def test_metadata_rejects_other_minors(version):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert version not in SpecifierSet(project["requires-python"])


@pytest.mark.parametrize(
    ("implementation", "version"),
    [
        ("cpython", (3, 10, 19)),
        ("cpython", (3, 12, 0)),
        ("cpython", (3, 13, 0)),
        ("cpython", (3, 14, 0)),
        ("pypy", (3, 11, 0)),
    ],
)
def test_installer_rejects_unsupported_interpreter_before_pip(
    monkeypatch, capsys, implementation, version
):
    monkeypatch.setattr(
        install,
        "sys",
        SimpleNamespace(implementation=SimpleNamespace(name=implementation), version_info=version),
    )
    monkeypatch.setattr(
        install.subprocess, "run", lambda *args, **kwargs: pytest.fail("pip must not run")
    )
    with pytest.raises(SystemExit) as error:
        install.main([])
    assert error.value.code == 2
    assert "requires CPython 3.11" in capsys.readouterr().err


def test_installer_runs_install_check_and_isolated_verification(monkeypatch):
    calls = []

    def run(command, *, cwd, check):
        assert Path(cwd).is_dir()
        assert Path(cwd) != ROOT
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(install.subprocess, "run", run)
    assert install.main([]) == 0
    assert calls[0][:4] == [install.sys.executable, "-m", "pip", "install"]
    assert calls[0][4:] == ["-e", f"{ROOT}[dev]"]
    assert calls[1] == [install.sys.executable, "-m", "pip", "check"]
    assert calls[2] == [install.sys.executable, "-I", "-m", "openonda.verify_install"]


def test_installer_stops_after_failed_install(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(install.subprocess, "run", run)
    assert install.main([]) == 7
    assert len(calls) == 1
