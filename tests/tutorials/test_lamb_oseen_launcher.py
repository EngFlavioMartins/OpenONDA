"""Campaign logging must preserve command ordering and failure exit status."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("fail_dvh", [False, True])
def test_campaign_phase_logs_preserve_exit_status(tmp_path, monkeypatch, fail_dvh):
    root = tmp_path / "checkout"
    tutorial = root / "tutorials/vpm/lamb_oseen_vortex"
    tutorial.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "tutorials/vpm/lamb_oseen_vortex/allrun.sh"
    shutil.copy2(source, tutorial / "allrun.sh")
    for name in ("allclean.sh", "allplot.sh"):
        script = tutorial / name
        script.write_text(f"#!/usr/bin/env bash\nprintf '[stub] {name}\\n'\n")
        script.chmod(0o755)
    interpreter = tmp_path / "python-stub"
    interpreter.write_text(
        '#!/usr/bin/env bash\nprintf "[stub] %s\\n" "$*"\n'
        'if [[ "$*" == *"setup vortex DVH"* && "${FAIL_DVH}" == 1 ]]; then exit 23; fi\n'
        "exit 0\n"
    )
    interpreter.chmod(0o755)
    monkeypatch.setenv("OPENONDA_PYTHON", str(interpreter))
    monkeypatch.setenv("TI_OFFLINE_CACHE_FILE_PATH", str(tmp_path / "cache"))
    monkeypatch.setenv("FAIL_DVH", "1" if fail_dvh else "0")
    result = subprocess.run(
        ["bash", str(tutorial / "allrun.sh")], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == (23 if fail_dvh else 0)
    assert "[campaign] START | vortex / DVH" in result.stdout
    assert result.stdout.count("[stub] allclean.sh") == 1
    assert not list((tmp_path / "cache").iterdir())
    if fail_dvh:
        assert "FAILED | vortex / DVH | exit 23" in result.stderr
        assert "DONE  | vortex / DVH" not in result.stdout
        assert "vortex / GBD" not in result.stdout
        assert "[campaign] COMPLETED" not in result.stdout
    else:
        assert "[campaign] COMPLETED" in result.stdout
        assert result.stdout.count(" / validate |") == 3
