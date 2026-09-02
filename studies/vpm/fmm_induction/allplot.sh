#!/usr/bin/env bash
set -euo pipefail

study_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${study_dir}/../../.."
python studies/vpm/fmm_induction/setup.py plot
