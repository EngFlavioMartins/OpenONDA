#!/bin/bash
# install_anaconda.sh - Install Miniforge and the complete OpenONDA environment
#
# Usage: ./install_anaconda.sh [-y]
#   -y: Auto-confirm Miniforge installation when Conda is absent

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/scripts/environment/environment.yml"
OPENVSP_INSTALLER="$SCRIPT_DIR/install_openvsp.sh"
CONDA_ENV="${OPENONDA_CONDA_ENV:-OpenONDA}"

# Large PETSc/VTK/Gmsh packages can exceed Conda's default network timeout.
export CONDA_REMOTE_CONNECT_TIMEOUT_SECS="${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-30}"
export CONDA_REMOTE_READ_TIMEOUT_SECS="${CONDA_REMOTE_READ_TIMEOUT_SECS:-120}"
export CONDA_REMOTE_MAX_RETRIES="${CONDA_REMOTE_MAX_RETRIES:-5}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "OpenONDA Environment Installer"
echo "=============================================="
echo ""

# Reuse an existing Conda installation; otherwise install Miniforge.
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✅ Conda is already installed at: $(which conda)${NC}"
    OS="$(uname -s)"
else
    # Detect OS and architecture for the Miniforge installer.
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux) OS_TYPE="Linux" ;;
        Darwin) OS_TYPE="MacOSX" ;;
        *)
            echo -e "${RED}❌ Unsupported OS: $OS${NC}"
            exit 1
            ;;
    esac

    case "$ARCH" in
        x86_64) ARCH_TYPE="x86_64" ;;
        arm64|aarch64)
            ARCH_TYPE="arm64"
            if [ "$OS_TYPE" = "Linux" ]; then
                ARCH_TYPE="aarch64"
            fi
            ;;
        *)
            echo -e "${RED}❌ Unsupported Architecture: $ARCH${NC}"
            exit 1
            ;;
    esac

    INSTALLER_NAME="Miniforge3-${OS_TYPE}-${ARCH_TYPE}.sh"
    DOWNLOAD_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER_NAME}"
    INSTALL_DIR="$HOME/miniforge3"

    echo "Detected System: $OS_TYPE ($ARCH_TYPE)"
    echo "Target: $INSTALL_DIR"
    echo ""

    if [ "${1:-}" != "-y" ]; then
        echo -e "${YELLOW}This script will download and install Miniforge3.${NC}"
        echo -e "${YELLOW}It requires approx 500MB of disk space.${NC}"
        read -r -p "Do you want to proceed? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY]) ;;
            *) echo "Installation aborted."; exit 1 ;;
        esac
    fi

    echo ""
    echo "Downloading $INSTALLER_NAME..."
    if command -v curl &> /dev/null; then
        curl -L -o "$INSTALLER_NAME" "$DOWNLOAD_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$INSTALLER_NAME" "$DOWNLOAD_URL"
    else
        echo -e "${RED}❌ Error: Neither curl nor wget found.${NC}"
        exit 1
    fi

    echo "Installing Miniforge..."
    bash "$INSTALLER_NAME" -b -p "$INSTALL_DIR"
    rm "$INSTALLER_NAME"

    # shellcheck disable=SC1091
    source "$INSTALL_DIR/bin/activate" base
    conda init bash
    if [ "$OS" = "Darwin" ]; then
        conda init zsh
    fi
fi

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ Environment file not found: $ENV_FILE${NC}"
    exit 1
fi

echo ""
echo "Creating/updating Conda environment '$CONDA_ENV'..."
conda env update --name "$CONDA_ENV" --file "$ENV_FILE"

echo "Installing OpenONDA in editable mode..."
conda run -n "$CONDA_ENV" python -m pip install --no-deps -e "$REPO_ROOT"

if [ "${OPENONDA_INSTALL_OPENVSP:-1}" = "1" ]; then
    echo "Installing OpenVSP and its Python API..."
    "$OPENVSP_INSTALLER"
fi

if [ -d "$REPO_ROOT/.git" ]; then
    if conda run -n "$CONDA_ENV" pre-commit --version >/dev/null 2>&1; then
        echo "Installing pre-commit hooks..."
        conda run -n "$CONDA_ENV" pre-commit install --install-hooks
    else
        echo "Skipping pre-commit hooks: development tooling is not installed."
    fi
fi

echo "Verifying the solver environment..."
for module in gmsh h5py mpi4py numba petsc4py pyamg pydantic pygit2 pytest pyvista ruff scipy taichi vtk; do
    conda run -n "$CONDA_ENV" python -c "import $module; print('$module: OK')"
done
conda run -n "$CONDA_ENV" python -m pip check
conda run -n "$CONDA_ENV" ruff --version
conda run -n "$CONDA_ENV" pyrefly --version
conda run -n "$CONDA_ENV" mpiexec --version

# ── Make new shells open the project env by default (instead of base) ─────────
configure_default_env() {
    local env_name="$1"
    echo ""
    echo "Configuring '$env_name' as the default environment for new shells..."

    # Stop Conda from auto-activating 'base' when a shell starts.
    conda config --set auto_activate_base false || true

    local marker_begin="# >>> OpenONDA default env >>>"
    local marker_end="# <<< OpenONDA default env <<<"
    local rc
    for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
        [ -f "$rc" ] || continue
        if grep -qF "$marker_begin" "$rc"; then
            echo "  $rc already configured — skipping."
            continue
        fi
        if [ ! -w "$rc" ]; then
            # e.g. a dotfile left root-owned by an earlier sudo edit.
            echo -e "  ${YELLOW}⚠ $rc is not writable (owner $(stat -f '%Su' "$rc" 2>/dev/null || echo '?')).${NC}"
            echo "    Fix ownership, then add the activation line manually:"
            echo "      sudo chown \"$(id -un)\" \"$rc\""
            echo "      printf '\\nconda activate %s\\n' \"$env_name\" >> \"$rc\""
            continue
        fi
        {
            printf '\n%s\n' "$marker_begin"
            printf '# Managed by scripts/install/install_anaconda.sh — activate the project env by default.\n'
            printf 'conda activate %s\n' "$env_name"
            printf '%s\n' "$marker_end"
        } >> "$rc"
        echo "  Added default activation of '$env_name' to $rc."
    done
}

configure_default_env "$CONDA_ENV"

echo ""
echo -e "${GREEN}✅ OpenONDA environment '$CONDA_ENV' is ready.${NC}"
echo "New terminals will open '$CONDA_ENV' automatically."
echo "Activate it in this shell with: conda activate $CONDA_ENV"
