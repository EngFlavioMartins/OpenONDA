#!/usr/bin/env bash
# Taichi ships its own runtime. This helper only checks whether a Vulkan driver
# is visible; it deliberately does not edit shell startup files or install SDKs.
set -euo pipefail

if command -v vulkaninfo >/dev/null 2>&1; then
    if vulkaninfo --summary >/dev/null 2>&1; then
        echo "Vulkan is available. Taichi may use the GPU_VULKAN backend."
        exit 0
    fi
    echo "vulkaninfo is installed, but no working Vulkan device was detected." >&2
    exit 1
fi

case "$(uname -s)" in
    Linux)
        echo "No vulkaninfo command was found. Install your GPU vendor's Vulkan driver" >&2
        echo "and the vulkan-tools package, then run this check again." >&2
        ;;
    Darwin)
        echo "Vulkan is optional on macOS; Taichi can use CPU or Metal." >&2
        ;;
    *) echo "Unsupported operating system." >&2 ;;
esac
exit 1
