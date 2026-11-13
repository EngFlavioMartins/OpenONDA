#!/bin/bash
# One-time setup of the sample sync on this (simulation) machine. Needs `sudo apt install syncthing`.
#   1. starts Syncthing as a user service (no default folder),
#   2. shares ~/Sync/openonda_samples send-only as folder "openonda-samples",
#   3. installs a user timer that runs scripts/publish_samples.sh every 15 minutes,
#   4. with a device ID argument, pairs the plotting machine and shares the folder with it.
# Usage: scripts/setup_sample_sync.sh [<device ID of the plotting machine>]
# Rerunning is harmless; run it again with the device ID once the other machine is installed.
set -euo pipefail

folder="$HOME/Sync/openonda_samples"
publish="$(cd "$(dirname "$0")" && pwd)/publish_samples.sh"
command -v syncthing >/dev/null || { echo "Install Syncthing first: sudo apt install syncthing" >&2; exit 1; }

if [ ! -f "$HOME/.local/state/syncthing/config.xml" ] && [ ! -f "$HOME/.config/syncthing/config.xml" ]; then
    # Without a default ~/Sync folder, which would contain the shared folder.
    STNODEFAULTFOLDER=1 syncthing generate --no-default-folder 2>/dev/null \
        || STNODEFAULTFOLDER=1 syncthing generate
fi
systemctl --user enable --now syncthing.service
until syncthing cli show system >/dev/null 2>&1; do sleep 1; done

mkdir -p "$folder"
if ! syncthing cli config folders list | grep -qx openonda-samples; then
    syncthing cli config folders add --id openonda-samples --label "OpenONDA samples" --path "$folder"
fi
syncthing cli config folders openonda-samples type set sendonly
if syncthing cli config folders list | grep -qx default; then
    echo "Remove the 'Default Folder' (~/Sync) in the web interface: it would contain this one." >&2
fi

units="$HOME/.config/systemd/user"
mkdir -p "$units"
cat > "$units/openonda-publish-samples.service" <<EOF
[Unit]
Description=Copy OpenONDA tutorial samples into the Syncthing folder

[Service]
Type=oneshot
Nice=10
IOSchedulingClass=idle
ExecStart=$publish $folder
EOF
cat > "$units/openonda-publish-samples.timer" <<EOF
[Unit]
Description=Publish OpenONDA tutorial samples every 15 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
Persistent=true

[Install]
WantedBy=timers.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now openonda-publish-samples.timer
# Keep the user services running when logged out (simulations run unattended).
loginctl enable-linger "$USER" 2>/dev/null || true

if [ $# -ge 1 ]; then
    syncthing cli config devices add --device-id "$1" --name plotting || true
    syncthing cli config folders openonda-samples devices add --device-id "$1" || true
fi
echo "This machine's device ID (enter it on the plotting machine):"
syncthing cli show system | grep -o '"myID": *"[^"]*"' | cut -d'"' -f4
echo "Web interface: http://127.0.0.1:8384"
