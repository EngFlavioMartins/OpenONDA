"""Installation verification checks the same files selected by the tutorial copier."""

from pathlib import PurePosixPath

import pytest

from openonda.tutorials import _include_resource
from openonda.verify_install import _verify_tutorial_commands, _verify_tutorial_source_paths


@pytest.mark.parametrize(
    "relative",
    [
        "case/study_results/old/worker.py",
        "case/solution/restart_history/setup.py",
        "case/samples/generated.py",
        "case/figures/postprocess.sh",
        "case/__pycache__/generated.py",
        "case/results/generated.py",
        "case/paraview_state.py",
        "case/paraview_tracer.py",
    ],
)
def test_generated_excluded_source_does_not_block_install_verification(tmp_path, relative):
    assert not _include_resource(PurePosixPath(relative))
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("OLD_CHECKOUT = '/home/example/old-checkout'\n")
    ordinary = tmp_path / "case/setup.py"
    ordinary.write_text("# Portable maintained template.\n")
    _verify_tutorial_source_paths(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "case/setup.py",
        "case/allrun.sh",
        "case/assets/worker.py",
        "case/reference_flow/setup.py",
        "case/results.py",
    ],
)
@pytest.mark.parametrize("marker", ["/home/example/project", "/Users/example/project"])
def test_included_source_still_rejects_machine_specific_paths(tmp_path, relative, marker):
    assert _include_resource(PurePosixPath(relative))
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(f"CHECKOUT = {marker!r}\n")
    with pytest.raises(RuntimeError, match="machine-specific path") as error:
        _verify_tutorial_source_paths(tmp_path)
    assert str(path) in str(error.value)


def test_portable_nested_sources_pass_installation_checks(tmp_path):
    source = tmp_path / "case/reference_flow/assets/plot.py"
    source.parent.mkdir(parents=True)
    source.write_text("from pathlib import Path\nROOT = Path(__file__).parent\n")
    _verify_tutorial_source_paths(tmp_path)


def test_installed_tutorial_commands_resolve_from_copied_cases_without_python_path(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert _verify_tutorial_commands() > 0
