"""Plot launchers propagate a failed command under direct and bash invocation."""

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = sorted((ROOT / "tutorials").glob("*/*/allplot.sh"))


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda path: str(path.relative_to(ROOT)))
@pytest.mark.parametrize("shell", [False, True], ids=["direct", "bash"])
def test_plot_launcher_stops_at_first_failure(tmp_path, launcher, shell):
    copied = tmp_path / "allplot.sh"
    copied.write_bytes(launcher.read_bytes())
    copied.chmod(0o755)
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\nexit 17\n')
    python.chmod(0o755)
    log = tmp_path / "calls.log"
    environment = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", CALL_LOG=str(log))
    command = ["bash", str(copied)] if shell else [str(copied)]
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, timeout=10)
    assert result.returncode == 17, result.stderr.decode()
    assert len(log.read_text().splitlines()) == 1
