#!/usr/bin/env bash
# Internal worker; install.sh keeps activation in the caller's shell.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVATION_FILE="${1:?Run source install.sh from the repository root.}"

case "$(uname -s):$(uname -m)" in
    Linux:x86_64) PLATFORM=Linux; ARCH=x86_64 ;;
    Darwin:x86_64) PLATFORM=MacOSX; ARCH=x86_64 ;;
    Darwin:arm64) PLATFORM=MacOSX; ARCH=arm64 ;;
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
    INSTALLER="Miniforge3-${PLATFORM}-${ARCH}.sh"
    DOWNLOAD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/openonda-miniforge.XXXXXX")"
    trap 'rm -rf "$DOWNLOAD_DIR"' EXIT
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER"
    for suffix in '' '.sha256'; do
        if command -v curl >/dev/null 2>&1; then
            curl --fail --location --retry 3 --output "$DOWNLOAD_DIR/$INSTALLER$suffix" "$URL$suffix"
        elif command -v wget >/dev/null 2>&1; then
            wget --output-document="$DOWNLOAD_DIR/$INSTALLER$suffix" "$URL$suffix"
        else
            echo 'Downloading Miniforge requires curl or wget.' >&2
            exit 1
        fi
    done
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
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/openonda.sh" <<'HOOK'
export PATH="$("$CONDA_PREFIX/bin/python" -c 'import os; p=os.path.join(os.environ["CONDA_PREFIX"], "bin"); print(os.pathsep.join([p] + [v for v in os.environ["PATH"].split(os.pathsep) if v != p]))')"
HOOK
source "$CONDA_PREFIX/etc/conda/activate.d/openonda.sh"
"$CONDA_PREFIX/bin/python" "$REPO_ROOT/scripts/install/install_tex.py"
"$CONDA_PREFIX/bin/python" "$REPO_ROOT/install.py" --with-environment

case "${SHELL:-/bin/bash}" in
    */zsh) "$CONDA_COMMAND" init --quiet zsh ;;
    *) "$CONDA_COMMAND" init --quiet bash ;;
esac
printf '%s\n' "$CONDA_ROOT" > "$ACTIVATION_FILE"
