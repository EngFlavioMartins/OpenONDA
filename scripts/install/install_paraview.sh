#!/usr/bin/env bash
# ============================================================
# ParaView 6.x installer for Ubuntu
# Tested on Ubuntu 20.04, 22.04, 24.04
# ============================================================

set -e  # stop on error

# ---- Settings ----
PARAVIEW_VERSION="6.0.1"
PARAVIEW_TAR="ParaView-${PARAVIEW_VERSION}-MPI-Linux-Python3.12-x86_64.tar.gz"
PARAVIEW_URL="https://www.paraview.org/files/v${PARAVIEW_VERSION%.*}/${PARAVIEW_TAR}"
INSTALL_DIR="/opt/paraview6"
SYMLINK_PATH="/usr/local/bin/paraview"

# ---- Check dependencies ----
echo "[INFO] Checking for wget and tar..."
sudo apt-get update -qq
sudo apt-get install -y wget tar libglu1-mesa libxrender1 libxt6 libsm6

# ---- Download ParaView ----
echo "[INFO] Downloading ParaView ${PARAVIEW_VERSION}..."
mkdir -p /tmp/paraview_install
cd /tmp/paraview_install
wget -q --show-progress "$PARAVIEW_URL"

# ---- Extract ----
echo "[INFO] Extracting ParaView to ${INSTALL_DIR}..."
sudo mkdir -p "$INSTALL_DIR"
sudo tar -xzf "$PARAVIEW_TAR" -C "$INSTALL_DIR" --strip-components=1

# ---- Create symlink ----
echo "[INFO] Creating symlink at ${SYMLINK_PATH}..."
sudo ln -sf "${INSTALL_DIR}/bin/paraview" "$SYMLINK_PATH"

# ---- Cleanup ----
cd ~
rm -rf /tmp/paraview_install

# ---- Verify ----
echo "[INFO] Checking ParaView installation..."
"$SYMLINK_PATH" --version || echo "[WARN] ParaView installed, but version check failed."

echo ""
echo "✅ ParaView ${PARAVIEW_VERSION} installed successfully!"
echo "You can run it with:  paraview"
