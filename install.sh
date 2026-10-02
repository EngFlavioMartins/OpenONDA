#!/usr/bin/env bash
# Source this file in Bash or Zsh to install and activate OpenONDA.
_openonda_install() {
    local installer_path activation_file conda_root install_status
    unset -f _openonda_install
    if [ -n "${ZSH_VERSION:-}" ]; then
        eval 'installer_path=${(%):-%x}'
    else
        installer_path="${BASH_SOURCE[0]}"
    fi
    installer_path="$(cd "$(dirname "$installer_path")" && pwd)" || return
    activation_file="$(mktemp "${TMPDIR:-/tmp}/openonda-activation.XXXXXX")" || return
    if bash "$installer_path/scripts/install/install_conda.sh" "$activation_file"; then
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
