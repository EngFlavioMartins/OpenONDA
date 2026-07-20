#!/usr/bin/env bash
# install_openvsp.sh - Wire OpenVSP into the OpenONDA conda workflow.
#
# OpenVSP binary releases are tied to the Python version they were built with.
# The current official OpenVSP 3.51.0 package needs Python 3.13.  The
# FVM/VPM environment intentionally uses Python 3.10 for Taichi compatibility,
# so install_anaconda.sh skips this optional component there.
#
# Usage:
#   scripts/install/install_openvsp.sh
#   OPENVSP_ROOT=/path/to/OpenVSP-3.51.0 scripts/install/install_openvsp.sh
#   OPENVSP_ARCHIVE_URL=https://.../OpenVSP.zip scripts/install/install_openvsp.sh

set -euo pipefail

CONDA_ENV="${OPENONDA_CONDA_ENV:-OpenONDA}"
OPENVSP_VERSION="${OPENVSP_VERSION:-3.51.0}"
OPENVSP_ROOT="${OPENVSP_ROOT:-$HOME/.local/share/openonda/OpenVSP-${OPENVSP_VERSION}}"
OPENVSP_ARCHIVE_URL="${OPENVSP_ARCHIVE_URL:-}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found on PATH. Install/activate conda first." >&2
    exit 1
fi

official_download_url() {
    os="$(uname -s)"
    arch="$(uname -m)"

    if [ "$os" = "Darwin" ] && [ "$arch" = "arm64" ]; then
        printf '%s\n' "https://openvsp.org/download.php?file=zips/current/mac/OpenVSP-${OPENVSP_VERSION}-macos-14-ARM64-Python3.13.zip"
        return
    fi
    if [ "$os" = "Darwin" ] && [ "$arch" = "x86_64" ]; then
        printf '%s\n' "https://openvsp.org/download.php?file=zips/current/mac/OpenVSP-${OPENVSP_VERSION}-macos-15-intel-X64-Python3.13.zip"
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

install_python_api() {
    openonda_python="$(conda run -n "$CONDA_ENV" which python)"
    python_version="$($openonda_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$python_version" != "3.13" ]; then
        echo "ERROR: OpenVSP ${OPENVSP_VERSION} requires Python 3.13; '$CONDA_ENV' uses $python_version." >&2
        exit 1
    fi

    "$openonda_python" -m pip install --no-deps --force-reinstall "$OPENVSP_ROOT/python/openvsp_config"
    "$openonda_python" -m pip install --no-deps --force-reinstall "$OPENVSP_ROOT/python/utilities"
    "$openonda_python" -m pip install --no-deps --force-reinstall "$OPENVSP_ROOT/python/degen_geom"
    "$openonda_python" -m pip install --no-deps --force-reinstall "$OPENVSP_ROOT/python/openvsp"
}

install_openonda_hooks() {
    openonda_prefix="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.prefix)')"
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
    openonda_prefix="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.prefix)')"
    "$openonda_prefix/bin/openvsp-python" -c "import openvsp as vsp; print(vsp.GetVSPVersion())"
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
  wrapper     : $(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.prefix)')/bin/openvsp-python

Reactivate the env to pick up PATH/OPENVSP_ROOT:
  conda deactivate
  conda activate $CONDA_ENV
EOF
