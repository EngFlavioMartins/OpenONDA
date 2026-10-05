"""The generic tutorial runner excludes duplicate launches across solver execs."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
OUTPUT_FILES = (
    "solution/fvm/mesh.npz",
    "solution/backups/checkpoint",
    "solutions/grid/backup",
    "samples/forces_history.csv",
    "figures/forces.png",
    "run.log",
)


def fake_case(tmp_path, reference):
    case = tmp_path / "case with spaces"
    case.mkdir()
    source = CASE / "reference_flow" if reference else CASE
    for name in ("allrun.sh", "allcontinue.sh"):
        shutil.copy2(source / name, case / name)
    for name in OUTPUT_FILES:
        path = case / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    (case / "setup.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['CALL_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "assert '--fresh' not in sys.argv\n"
        "os.closerange(3, 1024)\n"
        "os.execv(sys.executable, [sys.executable, '-c',\n"
        '    "import os,time; from pathlib import Path; "\n'
        "    \"Path(os.environ['READY']).touch(); \"\n"
        '    "exec(\\"while not Path(os.environ[\'RELEASE\']).exists():\\\\n time.sleep(.01)\\")"] )\n'
    )
    environment = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "CALL_LOG": str(tmp_path / "calls.jsonl"),
        "READY": str(tmp_path / "ready"),
        "RELEASE": str(tmp_path / "release"),
    }
    return case, environment


@pytest.mark.parametrize("reference", [False, True], ids=["coupled", "reference"])
@pytest.mark.parametrize("duplicate_arguments", [[], ["--fresh"]], ids=["continue", "fresh"])
def test_duplicate_launch_rejected_after_child_closes_fds_and_execs(
    tmp_path, reference, duplicate_arguments
):
    case, environment = fake_case(tmp_path, reference)
    first = subprocess.Popen(
        ["bash", str(case / "allcontinue.sh")],
        cwd="/tmp",
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not Path(environment["READY"]).exists() and time.monotonic() < deadline:
            assert first.poll() is None, first.communicate()
            time.sleep(0.01)
        assert Path(environment["READY"]).exists()
        duplicate = subprocess.run(
            ["bash", str(case / "allcontinue.sh"), *duplicate_arguments],
            cwd="/tmp",
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert duplicate.returncode == 75, duplicate.stderr
        assert "already running" in duplicate.stderr
        assert len(Path(environment["CALL_LOG"]).read_text().splitlines()) == 1
        assert not (case / "previous_runs").exists()
        for name in OUTPUT_FILES:
            assert (case / name).read_text() == name
    finally:
        Path(environment["RELEASE"]).touch()
        stdout, stderr = first.communicate(timeout=10)
    assert first.returncode == 0, (stdout, stderr)
    assert (case / ".openonda-run.lock").is_file()
    # Exiting releases the lock without deleting its inode; fresh launch can proceed.
    next_run = subprocess.run(
        ["bash", str(case / "allrun.sh"), "--fresh"],
        cwd="/tmp",
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert next_run.returncode == 0, next_run.stderr
    assert Path(environment["CALL_LOG"]).read_text().splitlines() == ["[]", "[]"]
    archives = list((case / "previous_runs").iterdir())
    assert len(archives) == 1
    for name in OUTPUT_FILES:
        assert not (case / name).exists()
        assert (archives[0] / name).read_text() == name


@pytest.mark.parametrize("reference", [False, True], ids=["coupled", "reference"])
def test_guard_preserves_solver_failure_exit_code(tmp_path, reference):
    case, environment = fake_case(tmp_path, reference)
    (case / "setup.py").write_text("raise SystemExit(17)\n")
    result = subprocess.run(
        ["bash", str(case / "allcontinue.sh")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 17
    assert "already running" not in result.stderr
