#!/usr/bin/env bash
# install_openvsp.sh - Wire OpenVSP into the OpenONDA conda workflow.
#
# OpenVSP binary releases are tied to the Python version they were built with.
# The OpenONDA VPM runtime currently uses Python 3.13 for Taichi, while the
# OpenVSP 3.51.0 Python API distributed locally is built for Python 3.14.
# This script therefore keeps the solver env unchanged and creates a small
# helper env/wrapper that OpenONDA can call for OpenVSP geometry generation.
#
# Usage:
#   scripts/install/install_openvsp.sh
#   OPENVSP_ROOT=/path/to/OpenVSP-3.51.0 scripts/install/install_openvsp.sh
#   OPENVSP_ARCHIVE_URL=https://.../OpenVSP.tar.gz scripts/install/install_openvsp.sh

set -euo pipefail

CONDA_ENV="${OPENONDA_CONDA_ENV:-OpenONDA}"
OPENVSP_VERSION="${OPENVSP_VERSION:-3.51.0}"
OPENVSP_ROOT="${OPENVSP_ROOT:-$HOME/OpenVSP-${OPENVSP_VERSION}}"
OPENVSP_HELPER_ENV="${OPENVSP_HELPER_ENV:-openonda-openvsp}"
OPENVSP_ARCHIVE_URL="${OPENVSP_ARCHIVE_URL:-}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found on PATH. Install/activate conda first." >&2
    exit 1
fi

download_openvsp() {
    if [ -d "$OPENVSP_ROOT" ]; then
        return 0
    fi

    if [ -z "$OPENVSP_ARCHIVE_URL" ]; then
        cat >&2 <<EOF
ERROR: OpenVSP was not found at:
  $OPENVSP_ROOT

Set OPENVSP_ROOT to an existing OpenVSP binary installation, or set
OPENVSP_ARCHIVE_URL to a Linux OpenVSP binary archive for this version.
EOF
        exit 1
    fi

    tmpdir="$(mktemp -d)"
    archive="$tmpdir/openvsp-archive"
    echo "Downloading OpenVSP from $OPENVSP_ARCHIVE_URL"
    curl -L "$OPENVSP_ARCHIVE_URL" -o "$archive"
    mkdir -p "$(dirname "$OPENVSP_ROOT")"
    mkdir -p "$OPENVSP_ROOT"
    tar -xf "$archive" -C "$OPENVSP_ROOT" --strip-components=1
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

ensure_helper_env() {
    if ! conda env list | awk '{print $1}' | grep -qx "$OPENVSP_HELPER_ENV"; then
        echo "Creating helper conda env: $OPENVSP_HELPER_ENV"
        conda create -y -n "$OPENVSP_HELPER_ENV" -c conda-forge python=3.14 pip
    fi

    helper_python="$(conda run -n "$OPENVSP_HELPER_ENV" which python)"
    "$helper_python" -m pip install --no-deps -e "$OPENVSP_ROOT/python/openvsp_config"
    "$helper_python" -m pip install --no-deps -e "$OPENVSP_ROOT/python/utilities"
    "$helper_python" -m pip install --no-deps -e "$OPENVSP_ROOT/python/degen_geom"
    "$helper_python" -m pip install --no-deps -e "$OPENVSP_ROOT/python/openvsp"
}

install_openonda_hooks() {
    openonda_prefix="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.prefix)')"
    helper_python="$(conda run -n "$OPENVSP_HELPER_ENV" which python)"

    mkdir -p "$openonda_prefix/bin"
    cat > "$openonda_prefix/bin/openvsp-python" <<EOF
#!/usr/bin/env bash
export OPENVSP_ROOT="$OPENVSP_ROOT"
export OPENVSP_PATH="\$OPENVSP_ROOT"
export PYTHONPATH="\$OPENVSP_ROOT/python/openvsp:\$OPENVSP_ROOT/python/degen_geom:\$OPENVSP_ROOT/python/openvsp_config:\$OPENVSP_ROOT/python/utilities\${PYTHONPATH:+:\$PYTHONPATH}"
export LD_LIBRARY_PATH="\$OPENVSP_ROOT/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "$helper_python" "\$@"
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
ensure_helper_env
install_openonda_hooks
verify_install

cat <<EOF
OpenVSP is wired into conda env '$CONDA_ENV'.
  OPENVSP_ROOT: $OPENVSP_ROOT
  helper env  : $OPENVSP_HELPER_ENV
  wrapper     : $(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.prefix)')/bin/openvsp-python

Reactivate the env to pick up PATH/OPENVSP_ROOT:
  conda deactivate
  conda activate $CONDA_ENV
EOF
