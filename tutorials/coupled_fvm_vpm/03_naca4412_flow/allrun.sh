#!/bin/bash
set -e

cd "$(dirname "$0")"

# MPI ranks keep the CPUs of this shell (Open MPI otherwise binds them from core 0 up, onto other
# runs); WING_COUPLED_CPUS pins them to a CPU list
export PRTE_MCA_hwloc_default_binding_policy=none
if [ -n "${WING_COUPLED_CPUS:-}" ]; then
    taskset -cp "$WING_COUPLED_CPUS" $$ > /dev/null
fi

# Remove previous output and simulate from t = 0.
./allclean.sh
python setup.py "$@"
