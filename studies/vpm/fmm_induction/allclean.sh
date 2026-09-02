#!/usr/bin/env bash
set -euo pipefail

study_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
find "${study_dir}/results" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
find "${study_dir}/figures" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
