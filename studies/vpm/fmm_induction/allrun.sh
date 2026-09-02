#!/usr/bin/env bash
set -euo pipefail

study_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${study_dir}/../../.."

"${study_dir}/allclean.sh"
python studies/vpm/fmm_induction/setup.py init

for kernel in GAUSSIAN HIGH_ORDER_GAUSSIAN SUPER_GAUSSIAN WINCKELMANS; do
    python studies/vpm/fmm_induction/setup.py accuracy "${kernel}" CPU 512 two_rings FMM
    python studies/vpm/fmm_induction/setup.py accuracy "${kernel}" VULKAN 512 two_rings FMM
done

for count in 1000 4000 14080 35000 70200; do
    python studies/vpm/fmm_induction/setup.py scaling FMM VULKAN "${count}" uniform
    python studies/vpm/fmm_induction/setup.py scaling TREE VULKAN "${count}" uniform
done

for distribution in uniform clustered elongated ring two_rings leapfrog rotor; do
    python studies/vpm/fmm_induction/setup.py accuracy GAUSSIAN VULKAN 14080 "${distribution}" FMM
done

python studies/vpm/fmm_induction/setup.py evolution VULKAN 512 200 two_rings
python studies/vpm/fmm_induction/setup.py comparison VULKAN 14080 10 leapfrog
python studies/vpm/fmm_induction/setup.py evolution VULKAN 14080 100 leapfrog
python studies/vpm/fmm_induction/setup.py evolution VULKAN 70200 20 rotor

python studies/vpm/fmm_induction/coupled_qualification.py vlm FMM VULKAN 100
python studies/vpm/fmm_induction/coupled_qualification.py vlm TREECODE VULKAN 100
python studies/vpm/fmm_induction/coupled_qualification.py compare-vlm
python studies/vpm/fmm_induction/coupled_qualification.py fvm FMM CPU 10
python studies/vpm/fmm_induction/coupled_qualification.py fvm TREECODE CPU 10
python studies/vpm/fmm_induction/coupled_qualification.py compare-fvm

python studies/vpm/fmm_induction/setup.py plot
