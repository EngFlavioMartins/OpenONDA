#!/usr/bin/env bash
# Source this file in Bash or Zsh to install and activate OpenONDA.
# Options --dev and --gaussian-mesh-cuda12 pass to install.py: source install.sh --dev
_openonda_install() {
    local installer_path activation_file conda_root install_status option
    local -a installer_options
    unset -f _openonda_install
    # A sourced file inherits the caller's arguments; keep only installer options.
    for option in "$@"; do
        case "$option" in
            --dev|--gaussian-mesh-cuda12) installer_options+=("$option") ;;
        esac
    done
    if [ -n "${ZSH_VERSION:-}" ]; then
        eval 'installer_path=${(%):-%x}'
    else
        installer_path="${BASH_SOURCE[0]}"
    fi
    installer_path="$(cd "$(dirname "$installer_path")" && pwd)" || return
    activation_file="$(mktemp "${TMPDIR:-/tmp}/openonda-activation.XXXXXX")" || return
    if bash "$installer_path/scripts/install/install_conda.sh" "$activation_file" ${installer_options[@]+"${installer_options[@]}"} </dev/null; then
        conda_root="$(cat "$activation_file")"
        rm -f "$activation_file"
        . "$conda_root/etc/profile.d/conda.sh" || return
        conda activate OpenONDA || return
        printf '\nOpenONDA is ready. Python edits take effect immediately.\n'
        printf 'In a new terminal: conda activate OpenONDA\n'
    else
        install_status=$?
        rm -f "$activation_file"
        return "$install_status"
    fi
}
_openonda_install "$@"
