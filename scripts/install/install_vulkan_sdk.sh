#!/usr/bin/env bash
# ===========================================================
# Vulkan SDK Installer for Ubuntu 24.04+ (including 25.04)
# Author: Flavio Martins (OpenONDA)
# ===========================================================

set -e  # exit on error

SDK_VERSION="1.4.328.1"
SDK_DIR="$HOME/VulkanSDK/$SDK_VERSION"
ARCHIVE_NAME="vulkansdk-linux-x86_64-${SDK_VERSION}.tar.xz"
DOWNLOAD_URL="https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/${ARCHIVE_NAME}"

echo ">>> Installing Vulkan system dependencies..."
sudo apt update -y
sudo apt install -y build-essential mesa-vulkan-drivers vulkan-tools vulkan-validationlayers libvulkan-dev wget jq

mkdir -p "$HOME/VulkanSDK"
cd "$HOME/VulkanSDK"

if [ ! -f "$ARCHIVE_NAME" ]; then
    echo ">>> Downloading Vulkan SDK ${SDK_VERSION}..."
    wget --quiet "$DOWNLOAD_URL" -O "$ARCHIVE_NAME"
else
    echo ">>> Archive already exists: $ARCHIVE_NAME"
fi

echo ">>> Extracting Vulkan SDK..."
tar -xf "$ARCHIVE_NAME" -C "$HOME/VulkanSDK"

echo ">>> Configuring Vulkan environment..."

cat <<EOF > "$HOME/.vulkan_env"
# Vulkan SDK environment (auto-generated)
export VULKAN_SDK="$SDK_DIR/x86_64"
export PATH="\$PATH:\$VULKAN_SDK/bin"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$VULKAN_SDK/lib"
export VK_ADD_LAYER_PATH="\$VULKAN_SDK/share/vulkan/explicit_layer.d"
export VK_LAYER_PATH="\$VK_ADD_LAYER_PATH"  # backward compatibility
EOF

if [ -f "$SDK_DIR/setup-env.sh" ]; then
    echo "source \"$SDK_DIR/setup-env.sh\"" >> "$HOME/.vulkan_env"
fi

if ! grep -q "source ~/.vulkan_env" "$HOME/.bashrc"; then
    echo "source ~/.vulkan_env" >> "$HOME/.bashrc"
fi

# --- FIXED VS CODE BLOCK ---
if command -v code >/dev/null 2>&1; then
    echo ">>> Setting up VS Code Vulkan environment..."
    VSCODE_SETTINGS_DIR="$HOME/.config/Code/User"
    mkdir -p "$VSCODE_SETTINGS_DIR"
    SETTINGS_FILE="$VSCODE_SETTINGS_DIR/settings.json"

    if [ ! -f "$SETTINGS_FILE" ]; then
        echo "{}" > "$SETTINGS_FILE"
    fi

    tmpfile=$(mktemp)
    jq --arg sdk "$SDK_DIR/x86_64" \
       '.["terminal.integrated.env.linux"] = {
           "VULKAN_SDK": $sdk,
           "PATH": "${env:PATH}:${VULKAN_SDK}/bin",
           "LD_LIBRARY_PATH": "${VULKAN_SDK}/lib:${env:LD_LIBRARY_PATH}",
           "VK_ADD_LAYER_PATH": "${VULKAN_SDK}/share/vulkan/explicit_layer.d",
           "VK_LAYER_PATH": "${VULKAN_SDK}/share/vulkan/explicit_layer.d"
       }' "$SETTINGS_FILE" > "$tmpfile" && mv "$tmpfile" "$SETTINGS_FILE"
fi
# --- END FIX ---

echo ">>> Reloading environment..."
source "$HOME/.vulkan_env"

echo ">>> Vulkan SDK installed successfully!"
echo "    Version: $SDK_VERSION"
echo "    SDK path: $VULKAN_SDK"
echo
echo ">>> Testing validation layer visibility..."
if vulkaninfo | grep -q "VK_LAYER_KHRONOS_validation"; then
    echo "✅ Validation layers detected!"
else
    echo "⚠️  Validation layers not detected — open a new terminal or restart VS Code."
fi
