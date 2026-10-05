"""Local modules retain case ownership through execution and cleanup."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from openonda.tutorial_runner import run_case


@pytest.mark.parametrize("module", ["setup", "assets.plot_field"])
def test_modules_execute_their_edited_case_with_relative_imports(tmp_path, module):
    for value in (3, 7):
        case = tmp_path / f"edited case {value}"
        assets = case / "assets"
        assets.mkdir(parents=True)
        (assets / "field.py").write_text(f"VALUE = {value}\n")
        (case / "setup.py").write_text(
            "from .assets.field import VALUE\n"
            "if __name__ == '__main__':\n"
            "    from .assets.output import save\n"
            "    save(VALUE)\n"
        )
        (assets / "plot_field.py").write_text(
            "from ..setup import VALUE\nfrom .output import save\nsave(VALUE)\n"
        )
        (assets / "output.py").write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "def save(value):\n"
            "    Path('result.json').write_text(json.dumps([value, sys.argv[1:]]))\n"
        )
        subprocess.run(
            [sys.executable, "-B", "-m", "openonda.tutorial_runner", str(case), module, "selected"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            timeout=15,
        )
        assert json.loads((case / "result.json").read_text()) == [value, ["selected"]]


def test_clean_removes_case_outputs_and_preserves_inputs_and_neighbours(tmp_path):
    case = tmp_path / "case"
    outputs = (
        "solution/fvm/mesh.npz",
        "solutions/grid/backup",
        "samples/forces.csv",
        "study_results/report.json",
        ".matplotlib/fontlist.json",
        "__pycache__/setup.pyc",
        "assets/__pycache__/field.pyc",
        "run.log",
    )
    retained = (
        "assets/body.stl",
        "reference_flow/solution/backup",
        "previous_runs/old/solution/backup",
        "drag_recovery/evidence.json",
    )
    for name in (*outputs, *retained):
        path = case / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    external = tmp_path / "external figure.png"
    external.write_text("external output")
    (case / "figures").symlink_to(external)
    subprocess.run(
        [sys.executable, "-B", "-m", "openonda.tutorial_runner", str(case), "clean"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=15,
    )
    assert all(not (case / name).exists() for name in outputs)
    assert not (case / "figures").is_symlink()
    assert external.read_text() == "external output"
    for name in retained:
        assert (case / name).read_text() == name


@pytest.mark.parametrize("backend", [None, "SVG"])
def test_copied_module_inherits_native_default_or_explicit_backend(tmp_path, backend):
    case = tmp_path / "case with spaces"
    assets = case / "assets"
    assets.mkdir(parents=True)
    (case / "physical_input.txt").write_text("case-owned input")
    (assets / "__init__.py").write_text(
        "import json, os\nimport matplotlib\nfrom pathlib import Path\n"
        "Path(__file__).with_name('backend.json').write_text(\n"
        "    json.dumps([os.environ['MPLBACKEND'], matplotlib.get_backend(),\n"
        "                Path('physical_input.txt').read_text()]))\n"
    )
    (assets / "plot_field.py").write_text("VALUE = 1\n")
    environment = os.environ.copy()
    environment.pop("MPLBACKEND", None)
    if backend is not None:
        environment["MPLBACKEND"] = backend
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "openonda.tutorial_runner",
            str(case),
            "assets.plot_field",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        timeout=15,
    )
    configured, actual, physical_input = json.loads((assets / "backend.json").read_text())
    assert configured == (backend or "Agg")
    assert actual.casefold() == configured.casefold()
    assert physical_input == "case-owned input"


def test_in_process_execution_restores_the_callers_environment_and_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    (tmp_path / "setup.py").write_text("raise SystemExit(3)\n")
    directory, arguments = Path.cwd(), sys.argv
    with pytest.raises(SystemExit) as stopped:
        run_case(tmp_path, "setup", ["selected"])
    assert stopped.value.code == 3
    assert Path.cwd() == directory
    assert sys.argv is arguments
    assert "MPLBACKEND" not in os.environ
