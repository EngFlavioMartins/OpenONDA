"""Exercise installation orchestration without changing the user's environment."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_tex", ROOT / "scripts/install/install_tex.py")
tex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tex)


def executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


@pytest.mark.parametrize("failure", ["", "conda", "tex", "package"])
@pytest.mark.parametrize("shell", ["bash", "zsh"])
def test_sourced_install_activates_only_after_success(tmp_path, failure, shell):
    shell_executable = shutil.which(shell)
    if shell_executable is None:
        pytest.skip(f"{shell} is not installed")
    checkout = tmp_path / "checkout with spaces"
    (checkout / "scripts/install").mkdir(parents=True)
    shutil.copy(ROOT / "install.sh", checkout / "install.sh")
    shutil.copy(ROOT / "scripts/install/install_conda.sh", checkout / "scripts/install")
    conda_root = tmp_path / "conda with spaces"
    environment = conda_root / "envs/OpenONDA"
    log = tmp_path / "calls.log"
    executable(conda_root / "bin/conda", '''#!/bin/bash
printf 'conda:%s\n' "$*" >> "$INSTALL_TEST_LOG"
case "$1" in
  info) printf '%s\n' "$INSTALL_TEST_ROOT" ;;
  env) if [[ "$INSTALL_TEST_FAILURE" == conda ]]; then exit 17; fi ;;
esac
''')
    hook = conda_root / "etc/profile.d/conda.sh"
    hook.parent.mkdir(parents=True)
    hook.write_text('''conda() {
  if [[ "$1" == activate ]]; then
    export PATH="$INSTALL_TEST_ENV/bin:$PATH"
    export CONDA_DEFAULT_ENV="$2"
    export CONDA_PREFIX="$INSTALL_TEST_ENV"
    if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/openonda.sh" ]]; then
      source "$CONDA_PREFIX/etc/conda/activate.d/openonda.sh"
    fi
  else
    "$CONDA_EXE" "$@"
  fi
}
''')
    executable(environment / "bin/python", '''#!/bin/bash
printf 'python:%s\n' "$*" >> "$INSTALL_TEST_LOG"
if [[ "$1" == -c ]]; then exec "$INSTALL_TEST_PYTHON" "$@"; fi
case "$1:$INSTALL_TEST_FAILURE" in
  */install_tex.py:tex) exit 18 ;;
  */install.py:package) exit 19 ;;
esac
''')
    result = subprocess.run(
        [shell_executable, "-f", "-c", '''
before_flags=$-
source "$1/install.sh"
install_exit=$?
printf 'status=%s env=%s cwd=%s flags=%s/%s\n' "$install_exit" "$CONDA_DEFAULT_ENV" "$PWD" "$before_flags" "$-"
printf 'first_path=%s\n' "${PATH%%:*}"
type _openonda_install >/dev/null 2>&1 && exit 80
exit "$install_exit"
''', shell, str(checkout)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
        env={**os.environ, "CONDA_EXE": str(conda_root / "bin/conda"),
             "CONDA_DEFAULT_ENV": "previous", "SHELL": f"/bin/{shell}",
             "INSTALL_TEST_ROOT": str(conda_root), "INSTALL_TEST_ENV": str(environment),
             "INSTALL_TEST_LOG": str(log), "INSTALL_TEST_FAILURE": failure,
             "INSTALL_TEST_PYTHON": sys.executable},
    )
    calls = log.read_text()
    assert "env update --name OpenONDA --file" in calls
    assert f"cwd={tmp_path}" in result.stdout
    if failure:
        assert result.returncode == {"conda": 17, "tex": 18, "package": 19}[failure]
        assert "env=previous" in result.stdout
        assert "OpenONDA is ready" not in result.stdout
        assert "conda:init" not in calls
        if failure != "package":
            assert "install.py --with-environment" not in calls
    else:
        assert result.returncode == 0, result.stderr
        assert "env=OpenONDA" in result.stdout
        assert f"first_path={environment / 'bin'}" in result.stdout
        assert "install.py --with-environment" in calls
        assert f"conda:init --quiet {shell}" in calls
        assert calls.index("install_tex.py") < calls.index("install.py --with-environment")
    flags = result.stdout.split("flags=", 1)[1].splitlines()[0].split("/")
    assert flags[0] == flags[1]


def tinytex_archive(member=".TinyTeX/bin/x86_64-linux/tlmgr"):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:xz") as archive:
        data = b"#!/bin/sh\n"
        entry = tarfile.TarInfo(member)
        entry.size = len(data)
        entry.mode = 0o755
        archive.addfile(entry, io.BytesIO(data))
    return stream.getvalue()


def prepare_tex(monkeypatch, tmp_path, *, bad_digest=False, member=None):
    prefix = tmp_path / "environment with spaces"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "bin").mkdir()
    payload = tinytex_archive(member) if member else tinytex_archive()
    downloads, commands = [], []

    def download(url, destination):
        downloads.append(url)
        if destination.suffix == ".json":
            digest = "0" * 64 if bad_digest else hashlib.sha256(payload).hexdigest()
            destination.write_text(json.dumps({"assets": [{
                "name": "TinyTeX-1-linux-x86_64-v2026.10.tar.xz",
                "digest": "sha256:" + digest, "browser_download_url": "https://example.invalid/tex",
            }]}))
        else:
            destination.write_bytes(payload)

    monkeypatch.setattr(tex, "download", download)
    monkeypatch.setattr(tex.platform, "system", lambda: "Linux")
    monkeypatch.setattr(tex.subprocess, "run", lambda command, **kwargs: commands.append(command))
    return prefix, downloads, commands


def test_tex_is_verified_contained_and_reusable(monkeypatch, tmp_path):
    prefix, downloads, commands = prepare_tex(monkeypatch, tmp_path)
    tex.install_tex(prefix)
    link = prefix / "bin/tlmgr"
    assert link.resolve().is_relative_to(prefix / "share/openonda-tex")
    assert len(downloads) == 2
    assert any(command[1:] == ["install", *tex.PACKAGES] for command in commands)
    tex.install_tex(prefix)
    assert len(downloads) == 2
    assert len(commands) == 2


def test_tex_rejects_corrupt_download_before_extracting(monkeypatch, tmp_path):
    prefix, _, commands = prepare_tex(monkeypatch, tmp_path, bad_digest=True)
    with pytest.raises(RuntimeError, match="SHA-256"):
        tex.install_tex(prefix)
    assert not commands
    assert not (prefix / "share/openonda-tex").exists()


def test_tex_rejects_archive_path_escape(monkeypatch, tmp_path):
    prefix, _, commands = prepare_tex(monkeypatch, tmp_path, member="../../../escaped")
    with pytest.raises(tarfile.FilterError):
        tex.install_tex(prefix)
    assert not commands
    assert not (tmp_path / "escaped").exists()


def test_tex_does_not_replace_an_existing_command(monkeypatch, tmp_path):
    prefix, _, _ = prepare_tex(monkeypatch, tmp_path)
    command = prefix / "bin/tlmgr"
    command.write_text("existing installation")
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        tex.install_tex(prefix)
    assert command.read_text() == "existing installation"
