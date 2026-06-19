#!/bin/bash
# install_anaconda.sh - Safe and cross-platform Miniconda installer
#
# Usage: ./install_anaconda.sh [-y]
#   -y: Auto-confirm installation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Miniforge Installation Script"
echo "=============================================="
echo ""

# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✅ Conda is already installed at: $(which conda)${NC}"
    exit 0
fi

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        OS_TYPE="Linux"
        ;;
    Darwin)
        OS_TYPE="MacOSX"
        ;;
    *)
        echo -e "${RED}❌ Unsupported OS: $OS${NC}"
        exit 1
        ;;
esac

# Detect Architecture
case "$ARCH" in
    x86_64)
        ARCH_TYPE="x86_64"
        ;;
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

# User Confirmation
if [ "$1" != "-y" ]; then
    echo -e "${YELLOW}This script will download and install Miniforge3.${NC}"
    echo -e "${YELLOW}It requires approx 500MB of disk space.${NC}"
    read -p "Do you want to proceed? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            ;;
        *)
            echo "Installation aborted."
            exit 1
            ;;
    esac
fi

# Download
echo ""
echo "Downloading $INSTALLER_NAME..."
if command -v curl &> /dev/null; then
    curl -L -O "$DOWNLOAD_URL"
elif command -v wget &> /dev/null; then
    wget "$DOWNLOAD_URL"
else
    echo -e "${RED}❌ Error: Neither curl nor wget found. Please install one of them.${NC}"
    exit 1
fi

if [ ! -f "$INSTALLER_NAME" ]; then
    echo -e "${RED}❌ Download failed.${NC}"
    exit 1
fi


# Handle existing directory
if [ -d "$INSTALL_DIR" ]; then
    echo "Directory $INSTALL_DIR exists. Backing up..."
    mv "$INSTALL_DIR" "${INSTALL_DIR}_backup_$(date +%s)"
fi

# Install
echo ""
echo "Installing Miniforge..."
bash "$INSTALLER_NAME" -b -p "$INSTALL_DIR"

# Cleanup
rm "$INSTALLER_NAME"

# Initialize
echo ""
echo "Initializing Conda..."
source "$INSTALL_DIR/bin/activate" base
conda init bash
if [ "$OS" = "Darwin" ]; then
    conda init zsh
fi

echo ""
echo -e "${GREEN}✅ Miniforge installed successfully!${NC}"
echo "Please restart your terminal."
