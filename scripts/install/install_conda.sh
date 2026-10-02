#!/usr/bin/env bash
# Internal worker; install.sh keeps activation in the caller's shell.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVATION_FILE="${1:?Run source install.sh from the repository root.}"

# Pin the bootstrap and its upstream SHA-256 together. The unversioned
# Miniforge asset has no .sha256 companion, and "latest" can change mid-download.
MINIFORGE_VERSION=26.7.2-0
case "$(uname -s):$(uname -m)" in
    Linux:x86_64)
        PLATFORM=Linux; ARCH=x86_64
        MINIFORGE_SHA256=281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05 ;;
    Darwin:x86_64)
        PLATFORM=MacOSX; ARCH=x86_64
        MINIFORGE_SHA256=b00e7798658f92721a3ae2f6b9832695ffc6baf07758894d726268055359f6c5 ;;
    Darwin:arm64)
        PLATFORM=MacOSX; ARCH=arm64
        MINIFORGE_SHA256=d70bfa2e97afcda96927c9b9ca0e2316cb7750e4ce651c94388267cbe9588711 ;;
    *) echo 'OpenONDA supports Linux x86-64 and macOS (Intel/Apple Silicon).' >&2; exit 1 ;;
esac

CONDA_COMMAND=""
for candidate in \
    "${CONDA_EXE:-}" \
    "$(type -P conda || true)" \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/mambaforge/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "$HOME/miniconda3/bin/conda"
do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        CONDA_COMMAND="$candidate"
        break
    fi
done

if [[ -z "$CONDA_COMMAND" ]]; then
    INSTALLER="Miniforge3-${MINIFORGE_VERSION}-${PLATFORM}-${ARCH}.sh"
    DOWNLOAD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/openonda-miniforge.XXXXXX")"
    trap 'rm -rf "$DOWNLOAD_DIR"' EXIT
    URL="https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_VERSION/$INSTALLER"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --connect-timeout 20 --speed-limit 1024 --speed-time 60 --retry 3 --output "$DOWNLOAD_DIR/$INSTALLER" "$URL"
    elif command -v wget >/dev/null 2>&1; then
        wget --timeout=60 --tries=4 --output-document="$DOWNLOAD_DIR/$INSTALLER" "$URL"
    else
        echo 'Downloading Miniforge requires curl or wget.' >&2
        exit 1
    fi
    printf '%s  %s\n' "$MINIFORGE_SHA256" "$INSTALLER" > "$DOWNLOAD_DIR/$INSTALLER.sha256"
    (
        cd "$DOWNLOAD_DIR"
        if command -v sha256sum >/dev/null 2>&1; then
            sha256sum --check "$INSTALLER.sha256"
        else
            shasum -a 256 --check "$INSTALLER.sha256"
        fi
    )
    bash "$DOWNLOAD_DIR/$INSTALLER" -b -p "$HOME/miniforge3"
    CONDA_COMMAND="$HOME/miniforge3/bin/conda"
fi

CONDA_ROOT="$("$CONDA_COMMAND" info --base)"
echo 'Creating or updating OpenONDA...'
"$CONDA_COMMAND" env update --name OpenONDA --file "$REPO_ROOT/scripts/environment/environment.yml"
# Activation supplies compiler and MPI library settings as well as Python.
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate OpenONDA
# Conda replaces the previous environment's PATH entry in place. Existing
# OpenFOAM/ParaView startup entries may therefore still precede it. Activate
# this environment with its own tools first, without leaving duplicate entries
# behind when Conda subsequently deactivates it.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/openonda.sh" <<'HOOK'
export PATH="$("$CONDA_PREFIX/bin/python" -c 'import os; p=os.path.join(os.environ["CONDA_PREFIX"], "bin"); print(os.pathsep.join([p] + [v for v in os.environ["PATH"].split(os.pathsep) if v != p]))')"
# GLVND searches system vendor directories by default, so merely installing
# Conda's Mesa does not expose its software renderer on a bare Linux machine.
# Keep system/custom drivers available and add the environment's Mesa vendor.
if [[ -f "$CONDA_PREFIX/share/glvnd/egl_vendor.d/50_mesa.json" ]]; then
    if [[ "${_OPENONDA_EGL_DIRS_SAVED+x}" != x ]]; then
        export _OPENONDA_EGL_DIRS_SAVED="${__EGL_VENDOR_LIBRARY_DIRS+x}"
        export _OPENONDA_EGL_DIRS_VALUE="${__EGL_VENDOR_LIBRARY_DIRS-}"
    fi
    case ":${__EGL_VENDOR_LIBRARY_DIRS-}:" in
        *":$CONDA_PREFIX/share/glvnd/egl_vendor.d:"*) ;;
        *)
            export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS-/etc/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d}"
            export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:+${__EGL_VENDOR_LIBRARY_DIRS}:}$CONDA_PREFIX/share/glvnd/egl_vendor.d"
            ;;
    esac
fi
HOOK
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/openonda.sh" <<'HOOK'
if [[ "${_OPENONDA_EGL_DIRS_SAVED+x}" == x ]]; then
    if [[ "$_OPENONDA_EGL_DIRS_SAVED" == x ]]; then
        export __EGL_VENDOR_LIBRARY_DIRS="$_OPENONDA_EGL_DIRS_VALUE"
    else
        unset __EGL_VENDOR_LIBRARY_DIRS
    fi
    unset _OPENONDA_EGL_DIRS_SAVED _OPENONDA_EGL_DIRS_VALUE
fi
HOOK
source "$CONDA_PREFIX/etc/conda/activate.d/openonda.sh"
"$CONDA_PREFIX/bin/python" "$REPO_ROOT/scripts/install/install_tex.py"
"$CONDA_PREFIX/bin/python" "$REPO_ROOT/install.py" --with-environment

case "${SHELL:-/bin/bash}" in
    */zsh) "$CONDA_COMMAND" init --quiet zsh ;;
    *) "$CONDA_COMMAND" init --quiet bash ;;
esac
printf '%s\n' "$CONDA_ROOT" > "$ACTIVATION_FILE"
