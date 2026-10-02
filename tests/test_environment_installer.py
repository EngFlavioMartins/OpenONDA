"""Exercise installation orchestration without changing the user's environment."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "install_tex", ROOT / "scripts/install/install_tex.py"
)
assert spec is not None and spec.loader is not None
tex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tex)


def executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


@pytest.mark.parametrize(
    "system,architecture,checksum",
    [
        ("Linux", "x86_64", "sha256sum"),
        ("Darwin", "x86_64", "shasum"),
        ("Darwin", "arm64", "shasum"),
    ],
)
@pytest.mark.parametrize("valid_digest", [False, True])
def test_fresh_bootstrap_uses_pinned_asset_and_checks_before_execution(
    tmp_path,
    system,
    architecture,
    checksum,
    valid_digest,
):
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("bash", "dirname", "mktemp", "rm", "cat", checksum):
        binary = shutil.which(name)
        if binary is None:
            pytest.skip(f"{name} is not installed")
        (commands / name).symlink_to(binary)
    executable(
        commands / "uname",
        f"""#!/bin/bash
case "$1" in -s) echo {system};; -m) echo {architecture};; esac
""",
    )
    payload = b'#!/bin/bash\nprintf installed > "$INSTALL_TEST_MARKER"\nexit 37\n'
    archive = tmp_path / "fixture.sh"
    archive.write_bytes(payload)
    executable(
        commands / "curl",
        """#!/bin/bash
while [[ $# -gt 0 ]]; do
    case "$1" in --output) destination="$2"; shift;; esac
    url="$1"
    shift
done
printf '%s\n' "$url" >> "$INSTALL_TEST_LOG"
cat "$INSTALL_TEST_ARCHIVE" > "$destination"
""",
    )
    source = (ROOT / "scripts/install/install_conda.sh").read_text()
    if valid_digest:
        source = re.sub(
            r"MINIFORGE_SHA256=[0-9a-f]{64}",
            "MINIFORGE_SHA256=" + hashlib.sha256(payload).hexdigest(),
            source,
        )
    worker = tmp_path / "scripts/install/install_conda.sh"
    executable(worker, source)
    marker, log = tmp_path / "installed", tmp_path / "downloads"
    result = subprocess.run(
        [str(commands / "bash"), str(worker), str(tmp_path / "activation")],
        env={
            "HOME": str(tmp_path),
            "PATH": str(commands),
            "INSTALL_TEST_ARCHIVE": str(archive),
            "INSTALL_TEST_MARKER": str(marker),
            "INSTALL_TEST_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    platform = "Linux" if system == "Linux" else "MacOSX"
    assert log.read_text().splitlines() == [
        "https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/"
        f"Miniforge3-26.7.2-0-{platform}-{architecture}.sh"
    ]
    assert marker.exists() == valid_digest
    assert result.returncode == (37 if valid_digest else 1), result.stderr
    if not valid_digest:
        assert "FAILED" in result.stdout


@pytest.mark.parametrize("failure", ["", "conda", "tex", "package"])
@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("egl_dirs", [None, "", "/custom/gpu"])
def test_sourced_install_activates_only_after_success(tmp_path, failure, shell, egl_dirs):
    shell_executable = shutil.which(shell)
    if shell_executable is None:
        pytest.skip(f"{shell} is not installed")
    checkout = tmp_path / "checkout with spaces"
    (checkout / "scripts/install").mkdir(parents=True)
    shutil.copy(ROOT / "install.sh", checkout / "install.sh")
    shutil.copy(ROOT / "scripts/install/install_conda.sh", checkout / "scripts/install")
    conda_root = tmp_path / "conda with spaces"
    environment = conda_root / "envs/OpenONDA"
    mesa = environment / "share/glvnd/egl_vendor.d/50_mesa.json"
    mesa.parent.mkdir(parents=True)
    mesa.write_text("{}")
    log = tmp_path / "calls.log"
    executable(
        conda_root / "bin/conda",
        """#!/bin/bash
printf 'conda:%s\n' "$*" >> "$INSTALL_TEST_LOG"
case "$1" in
  info) printf '%s\n' "$INSTALL_TEST_ROOT" ;;
  env) if [[ "$INSTALL_TEST_FAILURE" == conda ]]; then exit 17; fi ;;
esac
""",
    )
    hook = conda_root / "etc/profile.d/conda.sh"
    hook.parent.mkdir(parents=True)
    hook.write_text("""conda() {
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
""")
    executable(
        environment / "bin/python",
        """#!/bin/bash
printf 'python:%s\n' "$*" >> "$INSTALL_TEST_LOG"
if [[ "$1" == -c ]]; then exec "$INSTALL_TEST_PYTHON" "$@"; fi
case "$1:$INSTALL_TEST_FAILURE" in
  */install_tex.py:tex) exit 18 ;;
  */install.py:package) exit 19 ;;
esac
""",
    )
    process_environment = {
        **os.environ,
        "CONDA_EXE": str(conda_root / "bin/conda"),
        "CONDA_DEFAULT_ENV": "previous",
        "SHELL": f"/bin/{shell}",
        "INSTALL_TEST_ROOT": str(conda_root),
        "INSTALL_TEST_ENV": str(environment),
        "INSTALL_TEST_LOG": str(log),
        "INSTALL_TEST_FAILURE": failure,
        "INSTALL_TEST_PYTHON": sys.executable,
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/custom/nvidia.json",
    }
    for variable in (
        "__EGL_VENDOR_LIBRARY_DIRS",
        "_OPENONDA_EGL_DIRS_SAVED",
        "_OPENONDA_EGL_DIRS_VALUE",
    ):
        process_environment.pop(variable, None)
    if egl_dirs is not None:
        process_environment["__EGL_VENDOR_LIBRARY_DIRS"] = egl_dirs
    result = subprocess.run(
        [
            shell_executable,
            "-f",
            "-c",
            """
before_flags=$-
source "$1/install.sh"
install_exit=$?
printf 'status=%s env=%s cwd=%s flags=%s/%s\n' "$install_exit" "$CONDA_DEFAULT_ENV" "$PWD" "$before_flags" "$-"
printf 'first_path=%s\n' "${PATH%%:*}"
if [[ "$install_exit" == 0 ]]; then
    printf 'egl_active=%s\n' "$__EGL_VENDOR_LIBRARY_DIRS"
    source "$CONDA_PREFIX/etc/conda/activate.d/openonda.sh"
    printf 'egl_again=%s\n' "$__EGL_VENDOR_LIBRARY_DIRS"
    source "$CONDA_PREFIX/etc/conda/deactivate.d/openonda.sh"
fi
printf 'egl_restored=%s:%s\n' "${__EGL_VENDOR_LIBRARY_DIRS+x}" "${__EGL_VENDOR_LIBRARY_DIRS-}"
printf 'egl_files=%s\n' "$__EGL_VENDOR_LIBRARY_FILENAMES"
[[ "${_OPENONDA_EGL_DIRS_SAVED+x}" == x ]] && exit 81
type _openonda_install >/dev/null 2>&1 && exit 80
exit "$install_exit"
""",
            shell,
            str(checkout),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=process_environment,
    )
    calls = log.read_text()
    assert "env update --name OpenONDA --file" in calls
    assert f"cwd={tmp_path}" in result.stdout
    restored = ":" if egl_dirs is None else f"x:{egl_dirs}"
    assert f"egl_restored={restored}\n" in result.stdout
    assert "egl_files=/custom/nvidia.json" in result.stdout
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
        drivers = (
            "/etc/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d"
            if egl_dirs is None
            else egl_dirs
        )
        expected = f"{drivers + ':' if drivers else ''}{mesa.parent}"
        assert f"egl_active={expected}\n" in result.stdout
        assert f"egl_again={expected}\n" in result.stdout
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
            destination.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "name": "TinyTeX-1-linux-x86_64-v2026.10.tar.xz",
                                "digest": "sha256:" + digest,
                                "browser_download_url": "https://example.invalid/tex",
                            }
                        ]
                    }
                )
            )
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
