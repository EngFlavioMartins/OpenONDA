"""Command-line contract for the minimal cylinder reference case."""

from pathlib import Path
import subprocess

CASE_DIR = (
    Path(__file__).resolve().parents[2]
    / "tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow"
)


def test_reference_source_is_the_four_file_interface():
    generated = {"__pycache__", "samples", "solution"}
    assert {path.name for path in CASE_DIR.iterdir()} - generated == {
        "README.md",
        "allclean.sh",
        "allrun.sh",
        "setup.py",
    }


def run_launcher(tmp_path, *arguments):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    python.chmod(0o755)
    return subprocess.run(
        [str(CASE_DIR / "allrun.sh"), *arguments],
        env={"PYTHON": str(python)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_allrun_runs_the_complete_study(tmp_path):
    result = run_launcher(tmp_path)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "-u",
        "setup.py",
        "--name",
        "coarse",
        "--dx",
        "0.125",
        "-u",
        "setup.py",
        "--name",
        "medium",
        "--dx",
        "0.0625",
        "-u",
        "setup.py",
        "--name",
        "fine",
        "--dx",
        "0.03125",
    ]
