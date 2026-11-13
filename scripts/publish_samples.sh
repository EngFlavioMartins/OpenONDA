#!/bin/bash
# Copy the sampled results of the VPM, VLM-VPM and coupled FVM-VPM tutorials into a
# folder that Syncthing shares with the plotting machine (send-only on this side).
#
# Copied, with the tutorial layout kept: every samples/ directory, and from solution/
# only the run metadata (*_metadata.json), coupler.log and coupler_diagnostics.jsonl.
# Field output, backups and the FVM-only tutorials (tutorials/fvm) are left out.
# Symbolic links (cases that run under ~/openonda_runs) are followed. Nothing is
# deleted from the shared folder, so cleaning a case before a rerun keeps the
# last results on the other machine until the rerun overwrites them.
#
# Usage: scripts/publish_samples.sh [target]   (default: $OPENONDA_SAMPLES or ~/Sync/openonda_samples)
# A systemd user timer runs it every 15 minutes (see scripts/README.md).
set -euo pipefail

source="$(cd "$(dirname "$0")/../tutorials" && pwd)"
target="${1:-${OPENONDA_SAMPLES:-$HOME/Sync/openonda_samples}}"
mkdir -p "$target"

rsync -a --copy-links --prune-empty-dirs \
    --exclude='/fvm/' --exclude='*backup*/' --exclude='restart-branches/' \
    --exclude='__pycache__/' \
    --include='*/' \
    --include='samples/***' \
    --include='solution/**_metadata.json' \
    --include='solution/**coupler.log' \
    --include='solution/**coupler_diagnostics.jsonl' \
    --exclude='*' \
    "$source/" "$target/" || status=$?
# 23/24: some files could not be read (a dangling link) or vanished during a running case.
case "${status:-0}" in 0) ;; 23|24) echo "publish_samples: some files were skipped (see above)" >&2 ;; *) exit "$status" ;; esac
date -Is > "$target/LAST_PUBLISHED"
