# Cylinder mesh convergence

Status: **inconclusive**. Selected mesh: **none**.

Grid independence of listed force statistics for this fixed domain, geometry and model; not physical validation. Coarse is a convergence anchor, not a qualified selection.

| Run | Cells | mean Cd | Cd RMS | mean Cl | Cl RMS | Cl amplitude | St | Cycles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Qualification / next action

- RuntimeError: Independent checkMesh is unavailable; activate OpenFOAM first
- coarse mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/meshes/coarse/independent_check.json'
- medium mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/meshes/medium/mesh_manifest.json'
- fine mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/meshes/fine/mesh_manifest.json'
- fine_domain mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/meshes/fine_domain/mesh_manifest.json'
- fine_span mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/meshes/fine_span/mesh_manifest.json'
- coarse: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/coarse/progress.json'
- medium: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/medium/progress.json'
- medium_dt2: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/medium_dt2/progress.json'
- medium_dt4: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/medium_dt4/progress.json'
- medium_tight: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/medium_tight/progress.json'
- fine: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/fine/progress.json'
- fine_dt2: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/fine_dt2/progress.json'
- fine_dt4: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/fine_dt4/progress.json'
- fine_tight: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/fine_tight/progress.json'
- fine_domain: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/fine_domain/progress.json'
- fine_span: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop_v2/runs/fine_span/progress.json'

See grid_study.json for per-metric budgets and figures/ for plots.

Conservative engineering estimates; block intervals are not rigorous confidence bounds or independent proof of the asymptotic range.
