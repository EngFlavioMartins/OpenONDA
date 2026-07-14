#!/bin/bash
# install_anaconda.sh - Install Miniforge and the complete OpenONDA environment
#
# Usage: ./install_anaconda.sh [-y]
#   -y: Auto-confirm Miniforge installation when Conda is absent

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/scripts/environment/environment.yml"
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

if [ -d "$REPO_ROOT/.git" ]; then
    echo "Installing pre-commit hooks..."
    conda run -n "$CONDA_ENV" pre-commit install --install-hooks
fi

echo "Verifying the environment..."
for module in bandit gmsh mpi4py petsc4py pyamg pydantic pyvista pytest taichi vtk vulture; do
    conda run -n "$CONDA_ENV" python -c "import $module; print('$module: OK')"
done
conda run -n "$CONDA_ENV" python -m pip check
conda run -n "$CONDA_ENV" ruff --version
conda run -n "$CONDA_ENV" bandit --version
conda run -n "$CONDA_ENV" vulture --version

echo ""
echo -e "${GREEN}✅ OpenONDA environment '$CONDA_ENV' is ready.${NC}"
echo "Activate it with: conda activate $CONDA_ENV"
