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
OPENFOAM_INSTALLER="$SCRIPT_DIR/install_openfoam.sh"
CONDA_ENV="${OPENONDA_CONDA_ENV:-OpenONDA}"

# Single source of truth for the runtime Python: the `python=3.11` pin in
# environment.yml (matching the README badge and pyproject's requires-python).
REQUIRED_PYTHON="$(sed -nE 's/^[[:space:]]*-[[:space:]]*python[[:space:]]*=[[:space:]]*([0-9]+\.[0-9]+).*/\1/p' "$ENV_FILE" 2>/dev/null | head -n1)"
REQUIRED_PYTHON="${REQUIRED_PYTHON:-3.11}"

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

# Resolve the environment's own interpreter without consulting PATH.  Both
# `conda run -n ENV which python` and `conda run -n ENV python` pick `python`
# off PATH, so any *other* environment sitting earlier on PATH silently wins —
# which is how a stray env caused installs and version checks to target the
# wrong Python.  Ask conda for the target prefix and use an absolute path.
ENV_PREFIX="$(conda run -n "$CONDA_ENV" printenv CONDA_PREFIX 2>/dev/null | tr -d '\r')"
if [ -z "$ENV_PREFIX" ]; then
    ENV_PREFIX="$(conda env list | awk -v e="$CONDA_ENV" '$1==e {print $NF}')"
fi
ENV_PYTHON="$ENV_PREFIX/bin/python"
if [ ! -x "$ENV_PYTHON" ]; then
    echo -e "${RED}❌ Could not locate the '$CONDA_ENV' interpreter (looked for $ENV_PYTHON).${NC}"
    exit 1
fi

# Fail loudly here rather than at some later import.  A Python other than the
# pin in environment.yml breaks the compiled wheels (Taichi, PETSc, OpenVSP's
# _vsp.so) with obscure "symbol not found" errors at run time.
ACTUAL_PYTHON="$("$ENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$ACTUAL_PYTHON" != "$REQUIRED_PYTHON" ]; then
    echo -e "${RED}❌ '$CONDA_ENV' has Python $ACTUAL_PYTHON but OpenONDA requires $REQUIRED_PYTHON.${NC}"
    echo "   Remove the environment and re-run this script:"
    echo "     conda env remove -n $CONDA_ENV"
    exit 1
fi
echo -e "${GREEN}✅ $CONDA_ENV uses Python $ACTUAL_PYTHON ($ENV_PYTHON)${NC}"

echo "Installing OpenONDA in editable mode..."
"$ENV_PYTHON" -m pip install --no-deps -e "$REPO_ROOT"

if [ "${OPENONDA_INSTALL_OPENVSP:-1}" = "1" ]; then
    echo "Installing OpenVSP and its Python API..."
    "$OPENVSP_INSTALLER"
else
    echo "Skipping OpenVSP (OPENONDA_INSTALL_OPENVSP=0)."
fi

# ── Optional: OpenFOAM (needed only for the OFW wrapper and coupled_OFW_VPM) ──
# Not part of the Conda environment: it is a system package, needs root, and is
# packaged for Debian/Ubuntu only.  Everything else in OpenONDA (native FVM,
# VPM, VLM) runs without it, so this is opt-in.
install_openfoam_if_requested() {
    local choice="${OPENONDA_INSTALL_OPENFOAM:-}"

    if [ "$(uname -s)" != "Linux" ]; then
        if [ "$choice" = "1" ]; then
            echo -e "${YELLOW}⚠ OpenFOAM packages are Debian/Ubuntu-only; skipping on $(uname -s).${NC}"
            echo "  Use an Ubuntu VM/container to run the OFW and coupled_OFW_VPM tutorials."
        fi
        return 0
    fi

    if [ -z "$choice" ]; then
        if [ -t 0 ]; then
            echo ""
            echo -e "${YELLOW}OpenFOAM is optional. It is required only for the OFW wrapper${NC}"
            echo -e "${YELLOW}and the coupled_OFW_VPM tutorials, and installing it needs sudo.${NC}"
            read -r -p "Install OpenFOAM now? [y/N] " response
            case "$response" in
                [yY][eE][sS]|[yY]) choice=1 ;;
                *) choice=0 ;;
            esac
        else
            # Non-interactive (CI): default to skipping the privileged install.
            choice=0
        fi
    fi

    if [ "$choice" = "1" ]; then
        echo "Installing OpenFOAM..."
        "$OPENFOAM_INSTALLER"
    else
        echo "Skipping OpenFOAM. Install it later with:"
        echo "  OPENONDA_INSTALL_OPENFOAM=1 scripts/install/install_openfoam.sh"
    fi
}

install_openfoam_if_requested

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
    "$ENV_PYTHON" -c "import $module; print('$module: OK')"
done

# OpenVSP is optional, but a *broken* install is worse than an absent one: the
# blade generators fall back to re-launching a helper interpreter, so an ABI
# mismatch surfaces as a confusing failure far from here.  Check it explicitly.
if [ "${OPENONDA_INSTALL_OPENVSP:-1}" = "1" ]; then
    if "$ENV_PREFIX/bin/openvsp-python" -c "import openvsp; print('openvsp:', openvsp.GetVSPVersion())" 2>/dev/null; then
        :
    else
        echo -e "${YELLOW}⚠ OpenVSP is installed but its Python API does not import.${NC}"
        echo "  Tutorials that ship a cached blade JSON still run; regeneration will not."
        echo "  Re-run: scripts/install/install_openvsp.sh"
    fi
fi

"$ENV_PYTHON" -m pip check
"$ENV_PREFIX/bin/ruff" --version
"$ENV_PREFIX/bin/pyrefly" --version
"$ENV_PREFIX/bin/mpiexec" --version

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
