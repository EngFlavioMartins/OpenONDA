#!/usr/bin/env bash
# install_openvsp.sh - Wire OpenVSP into the OpenONDA conda workflow.
#
# OpenVSP binary releases are tied to the Python version they were built with,
# so the download, the ABI check, and the environment must all agree.  The
# required version is derived from scripts/environment/environment.yml rather
# than hardcoded here — a mismatch shows up at import time as
# "symbol not found ... _PyDict_GetItemStringRef", not at install time.
#
# Usage:
#   scripts/install/install_openvsp.sh
#   OPENVSP_ROOT=/path/to/OpenVSP-3.51.0 scripts/install/install_openvsp.sh
#   OPENVSP_ARCHIVE_URL=https://.../OpenVSP.zip scripts/install/install_openvsp.sh

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/install/install_openvsp.sh

Install the optional OpenVSP application and Python API for the OpenONDA Conda
environment. Configure non-default locations with OPENONDA_CONDA_ENV,
OPENVSP_VERSION, OPENVSP_ROOT, or OPENVSP_ARCHIVE_URL.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/scripts/environment/environment.yml"

CONDA_ENV="${OPENONDA_CONDA_ENV:-OpenONDA}"

# Single source of truth for the runtime Python: the `python=3.11` pin in
# environment.yml (which matches the README badge and pyproject's
# requires-python).  Deriving it here keeps the OpenVSP ABI check from drifting
# away from the environment the project actually builds.
REQUIRED_PYTHON="$(sed -nE 's/^[[:space:]]*-[[:space:]]*python[[:space:]]*=[[:space:]]*([0-9]+\.[0-9]+).*/\1/p' "$ENV_FILE" | head -n1)"
REQUIRED_PYTHON="${REQUIRED_PYTHON:-3.11}"

OPENVSP_VERSION="${OPENVSP_VERSION:-3.51.0}"
OPENVSP_ROOT="${OPENVSP_ROOT:-$HOME/.local/share/openonda/OpenVSP-${OPENVSP_VERSION}-Python${REQUIRED_PYTHON}}"
OPENVSP_ARCHIVE_URL="${OPENVSP_ARCHIVE_URL:-}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found on PATH. Install/activate conda first." >&2
    exit 1
fi

official_download_url() {
    os="$(uname -s)"
    arch="$(uname -m)"

    if [ "$os" = "Darwin" ] && [ "$arch" = "arm64" ]; then
        printf '%s\n' "https://openvsp.org/download.php?file=zips/current/mac/OpenVSP-${OPENVSP_VERSION}-macos-14-ARM64-Python${REQUIRED_PYTHON}.zip"
        return
    fi
    if [ "$os" = "Darwin" ] && [ "$arch" = "x86_64" ]; then
        printf '%s\n' "https://openvsp.org/download.php?file=zips/current/mac/OpenVSP-${OPENVSP_VERSION}-macos-15-intel-X64-Python${REQUIRED_PYTHON}.zip"
        return
    fi
    if [ "$os" = "Linux" ] && [ "$arch" = "x86_64" ] && [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        if [ "${ID:-}" = "ubuntu" ] && { [ "${VERSION_ID:-}" = "24.04" ] || [ "${VERSION_ID:-}" = "26.04" ]; }; then
            printf '%s\n' "https://openvsp.org/download.php?file=zips/current/linux/OpenVSP-${OPENVSP_VERSION}-Ubuntu-${VERSION_ID}_amd64.deb"
            return
        fi
    fi

    echo "ERROR: OpenVSP ${OPENVSP_VERSION} has no supported binary for $os/$arch." >&2
    echo "Set OPENVSP_ROOT to a compatible installation or OPENVSP_ARCHIVE_URL to an official binary archive." >&2
    exit 1
}

extract_openvsp() {
    archive="$1"
    destination="$2"

    case "$archive" in
        *.zip) unzip -q "$archive" -d "$destination" ;;
        *.deb) dpkg-deb -x "$archive" "$destination" ;;
        *.tar|*.tar.gz|*.tgz|*.tar.xz) tar -xf "$archive" -C "$destination" ;;
        *)
            if unzip -q "$archive" -d "$destination" 2>/dev/null; then
                return
            fi
            if command -v dpkg-deb >/dev/null 2>&1 && dpkg-deb -x "$archive" "$destination" 2>/dev/null; then
                return
            fi
            tar -xf "$archive" -C "$destination"
            ;;
    esac
}

download_openvsp() {
    if [ -d "$OPENVSP_ROOT" ]; then
        return 0
    fi

    if [ -z "$OPENVSP_ARCHIVE_URL" ]; then
        OPENVSP_ARCHIVE_URL="$(official_download_url)"
    fi

    tmpdir="$(mktemp -d)"
    archive_name="${OPENVSP_ARCHIVE_URL%%\?*}"
    archive_name="${archive_name##*/}"
    if [ -z "$archive_name" ] || [ "$archive_name" = "download.php" ]; then
        case "$OPENVSP_ARCHIVE_URL" in
            *file=*.zip*) archive_name="OpenVSP.zip" ;;
            *file=*.deb*) archive_name="OpenVSP.deb" ;;
            *) archive_name="OpenVSP.archive" ;;
        esac
    fi
    archive="$tmpdir/$archive_name"
    extracted="$tmpdir/extracted"
    echo "Downloading OpenVSP from $OPENVSP_ARCHIVE_URL"
    curl --fail --location --retry 3 "$OPENVSP_ARCHIVE_URL" -o "$archive"
    mkdir -p "$extracted"
    extract_openvsp "$archive" "$extracted"

    discovered_root="$(find "$extracted" -type f -name vspscript -print | head -n 1 | xargs -I{} dirname '{}')"
    if [ -z "$discovered_root" ]; then
        echo "ERROR: Downloaded OpenVSP archive does not contain vspscript." >&2
        rm -rf "$tmpdir"
        exit 1
    fi

    mkdir -p "$(dirname "$OPENVSP_ROOT")"
    mv "$discovered_root" "$OPENVSP_ROOT"
    rm -rf "$tmpdir"
}

check_openvsp_tree() {
    for path in \
        "$OPENVSP_ROOT/vsp" \
        "$OPENVSP_ROOT/vspscript" \
        "$OPENVSP_ROOT/python/openvsp" \
        "$OPENVSP_ROOT/python/degen_geom" \
        "$OPENVSP_ROOT/python/openvsp_config" \
        "$OPENVSP_ROOT/python/utilities"
    do
        if [ ! -e "$path" ]; then
            echo "ERROR: OpenVSP install is missing required path: $path" >&2
            exit 1
        fi
    done
}

# Resolve the interpreter of $CONDA_ENV without consulting PATH.
#
# `conda run -n ENV which python` (and even `conda run -n ENV python`) resolve
# `python` through PATH, so a *different* environment appearing earlier on PATH
# silently wins — that is how a stray env made this installer report the wrong
# Python version and refuse to install.  Ask conda for the target env's own
# prefix instead, and invoke its interpreter by absolute path.
openonda_prefix() {
    local prefix
    prefix="$(conda run -n "$CONDA_ENV" printenv CONDA_PREFIX 2>/dev/null | tr -d '\r')"
    if [ -z "$prefix" ]; then
        prefix="$(conda env list | awk -v e="$CONDA_ENV" '$1==e {print $NF}')"
    fi
    if [ -z "$prefix" ] || [ ! -x "$prefix/bin/python" ]; then
        echo "ERROR: could not locate the '$CONDA_ENV' environment's interpreter." >&2
        echo "       Create it first: scripts/install/install_conda.sh" >&2
        exit 1
    fi
    printf '%s\n' "$prefix"
}

install_python_api() {
    openonda_python="$(openonda_prefix)/bin/python"
    python_version="$($openonda_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$python_version" != "$REQUIRED_PYTHON" ]; then
        echo "ERROR: OpenVSP ${OPENVSP_VERSION} requires Python ${REQUIRED_PYTHON}; '$CONDA_ENV' uses $python_version." >&2
        echo "       Recreate the environment from scripts/environment/environment.yml." >&2
        exit 1
    fi

    "$openonda_python" -m pip install --no-deps --force-reinstall "$OPENVSP_ROOT/python/openvsp_config"
    "$openonda_python" -m pip install --no-deps --force-reinstall "$OPENVSP_ROOT/python/utilities"
    "$openonda_python" -m pip install --no-deps --force-reinstall "$OPENVSP_ROOT/python/degen_geom"
    "$openonda_python" -m pip install --no-deps --force-reinstall "$OPENVSP_ROOT/python/openvsp"
}

install_openonda_hooks() {
    openonda_prefix="$(openonda_prefix)"
    openonda_python="$openonda_prefix/bin/python"

    mkdir -p "$openonda_prefix/bin"
    cat > "$openonda_prefix/bin/openvsp-python" <<EOF
#!/usr/bin/env bash
export OPENVSP_ROOT="$OPENVSP_ROOT"
export OPENVSP_PATH="\$OPENVSP_ROOT"
export PYTHONPATH="\$OPENVSP_ROOT/python/openvsp:\$OPENVSP_ROOT/python/degen_geom:\$OPENVSP_ROOT/python/openvsp_config:\$OPENVSP_ROOT/python/utilities\${PYTHONPATH:+:\$PYTHONPATH}"
export LD_LIBRARY_PATH="\$OPENVSP_ROOT/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "$openonda_python" "\$@"
EOF
    chmod +x "$openonda_prefix/bin/openvsp-python"

    mkdir -p "$openonda_prefix/etc/conda/activate.d" "$openonda_prefix/etc/conda/deactivate.d"
    cat > "$openonda_prefix/etc/conda/activate.d/openvsp.sh" <<EOF
#!/usr/bin/env bash
export OPENVSP_ROOT="$OPENVSP_ROOT"
export OPENVSP_PATH="\$OPENVSP_ROOT"
export OPENONDA_OPENVSP_PYTHON="$openonda_prefix/bin/openvsp-python"
export _OPENONDA_OPENVSP_OLD_PATH="\${PATH:-}"
export PATH="\$OPENVSP_ROOT:\$PATH"
EOF

    cat > "$openonda_prefix/etc/conda/deactivate.d/openvsp.sh" <<'EOF'
#!/usr/bin/env bash
if [ -n "${_OPENONDA_OPENVSP_OLD_PATH:-}" ]; then
    export PATH="$_OPENONDA_OPENVSP_OLD_PATH"
fi
unset OPENVSP_ROOT
unset OPENVSP_PATH
unset OPENONDA_OPENVSP_PYTHON
unset _OPENONDA_OPENVSP_OLD_PATH
EOF
}

verify_install() {
    local prefix
    prefix="$(openonda_prefix)"
    "$prefix/bin/openvsp-python" -c "import openvsp as vsp; print('openvsp:', vsp.GetVSPVersion())"
    "$OPENVSP_ROOT/vspscript" -help >/dev/null 2>&1 || true
}

download_openvsp
check_openvsp_tree
install_python_api
install_openonda_hooks
verify_install

cat <<EOF
OpenVSP is wired into conda env '$CONDA_ENV'.
  OPENVSP_ROOT: $OPENVSP_ROOT
  wrapper     : $(openonda_prefix)/bin/openvsp-python

Reactivate the env to pick up PATH/OPENVSP_ROOT:
  conda deactivate
  conda activate $CONDA_ENV
EOF
