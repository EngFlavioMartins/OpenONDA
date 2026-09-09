#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_DIR="$REPO_ROOT/scripts/environment"
ENV_NAME="${OPENONDA_CONDA_ENV:-OpenONDA}"
ENV_FILE="$ENV_DIR/environment.yml"
AUTO_YES=1
EDITABLE=0

usage() {
    printf '%s\n' \
        "Usage: scripts/install/install_conda.sh [options]" \
        "" \
        "  -y, --yes       install Miniforge without prompting (the default)" \
        "  --prompt        ask before installing Miniforge if Conda is absent" \
        "  --parallel      install the MPI/PETSc environment" \
        "  --dev          editable install with development tools" \
        "  --no-editable   install a fixed copy instead of linking the repository" \
        "  --name NAME     choose the Conda environment name" \
        "  -h, --help      show this help"
}

while (($#)); do
    case "$1" in
        -y|--yes) AUTO_YES=1 ;;
        --prompt) AUTO_YES=0 ;;
        --parallel)
            ENV_FILE="$ENV_DIR/environment-parallel.yml"
            if [[ "$ENV_NAME" == "OpenONDA" ]]; then ENV_NAME="OpenONDA-parallel"; fi
            ;;
        --no-editable) EDITABLE=0 ;;
        --dev) EDITABLE=1 ;;
        --name)
            shift
            [[ $# -gt 0 ]] || { echo "--name requires a value" >&2; exit 2; }
            ENV_NAME="$1"
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

case "$(uname -s)" in
    Linux) PLATFORM="Linux" ;;
    Darwin) PLATFORM="MacOSX" ;;
    *) echo "OpenONDA supports Linux and macOS." >&2; exit 1 ;;
esac

CONDA_EXE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
else
    for candidate in \
        "${OPENONDA_CONDA_EXE:-}" \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/mambaforge/bin/conda" \
        "$HOME/anaconda3/bin/conda"
    do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            CONDA_EXE="$candidate"
            break
        fi
    done
fi

if [[ -z "$CONDA_EXE" ]]; then
    case "$(uname -m)" in
        x86_64) ARCH="x86_64" ;;
        arm64|aarch64)
            if [[ "$PLATFORM" == "Linux" ]]; then ARCH="aarch64"; else ARCH="arm64"; fi
            ;;
        *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    MINIFORGE_ROOT="${OPENONDA_MINIFORGE_ROOT:-$HOME/miniforge3}"
    INSTALLER="Miniforge3-${PLATFORM}-${ARCH}.sh"
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER"
    if [[ $AUTO_YES -ne 1 ]]; then
        printf 'Conda was not found. Install Miniforge at %s? [y/N] ' "$MINIFORGE_ROOT"
        read -r response
        [[ "$response" =~ ^[Yy]([Ee][Ss])?$ ]] || exit 1
    fi
    TEMP_INSTALLER="$(mktemp "${TMPDIR:-/tmp}/openonda-miniforge.XXXXXX.sh")"
    trap 'rm -f "$TEMP_INSTALLER"' EXIT
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "$TEMP_INSTALLER" "$URL"
    elif command -v wget >/dev/null 2>&1; then
        wget --output-document="$TEMP_INSTALLER" "$URL"
    else
        echo "Install curl or wget, then re-run this installer." >&2
        exit 1
    fi
    bash "$TEMP_INSTALLER" -b -p "$MINIFORGE_ROOT"
    CONDA_EXE="$MINIFORGE_ROOT/bin/conda"
fi

CONDA_ROOT="$("$CONDA_EXE" info --base)"

echo "Creating or updating Conda environment '$ENV_NAME'..."
"$CONDA_EXE" env update --name "$ENV_NAME" --file "$ENV_FILE"

ENV_PREFIX="$("$CONDA_EXE" env list | awk -v name="$ENV_NAME" '$1 == name {print $NF; exit}')"
ENV_PYTHON="$ENV_PREFIX/bin/python"
if [[ -z "$ENV_PREFIX" || ! -x "$ENV_PYTHON" ]]; then
    echo "Could not locate Python in Conda environment '$ENV_NAME'." >&2
    exit 1
fi

REQUIRED_PYTHON="$(sed -n 's/^[[:space:]]*-[[:space:]]*python=\([0-9][0-9.]*\).*/\1/p' "$ENV_FILE" | head -1)"
ACTUAL_PYTHON="$("$ENV_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ -n "$REQUIRED_PYTHON" && "$ACTUAL_PYTHON" != "$REQUIRED_PYTHON" ]]; then
    echo "Environment '$ENV_NAME' has Python $ACTUAL_PYTHON, but OpenONDA requires $REQUIRED_PYTHON." >&2
    echo "Remove it and re-run this installer:  conda env remove --name $ENV_NAME" >&2
    exit 1
fi
echo "Using Python $ACTUAL_PYTHON from '$ENV_NAME'."

INSTALL_ARGS=()
if [[ $EDITABLE -eq 1 ]]; then INSTALL_ARGS+=(--dev); fi
"$ENV_PYTHON" "$REPO_ROOT/install.py" ${INSTALL_ARGS[@]+"${INSTALL_ARGS[@]}"}

echo
echo "OpenONDA is ready in '$ENV_NAME'."
printf 'Activate it with:\n  source %q\n  conda activate %q\n' \
    "$CONDA_ROOT/etc/profile.d/conda.sh" "$ENV_NAME"
echo "The package can then be imported from any directory; PYTHONPATH is not needed."
echo "Inspect the installation with: openonda info"
echo "List packaged tutorials with: openonda tutorial list"
if [[ $EDITABLE -eq 1 ]]; then
    echo "Edits under $REPO_ROOT take effect immediately; no reinstall is needed."
fi
