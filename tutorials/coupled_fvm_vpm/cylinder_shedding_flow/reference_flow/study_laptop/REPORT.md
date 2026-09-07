# Cylinder mesh convergence

Status: **inconclusive**. Selected mesh: **none**.

Grid independence of listed force statistics for this fixed domain, geometry and model; not physical validation. Coarse is a convergence anchor, not a qualified selection.

| Run | Cells | mean Cd | Cd RMS | mean Cl | Cl RMS | Cl amplitude | St | Cycles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Qualification / next action

- CalledProcessError: Command '['/opt/anaconda3/envs/OpenONDA/bin/python', '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/mesh.py', '--case', 'coarse', '--output-dir', '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/meshes/coarse', '--backup-dir', '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/solution/coarse']' returned non-zero exit status 1.
- coarse mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/meshes/coarse/mesh_manifest.json'
- medium mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/meshes/medium/mesh_manifest.json'
- fine mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/meshes/fine/mesh_manifest.json'
- fine_domain mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/meshes/fine_domain/mesh_manifest.json'
- fine_span mesh: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/meshes/fine_span/mesh_manifest.json'
- coarse: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/coarse/progress.json'
- medium: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/medium/progress.json'
- medium_dt2: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/medium_dt2/progress.json'
- medium_dt4: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/medium_dt4/progress.json'
- medium_tight: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/medium_tight/progress.json'
- fine: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/fine/progress.json'
- fine_dt2: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/fine_dt2/progress.json'
- fine_dt4: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/fine_dt4/progress.json'
- fine_tight: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/fine_tight/progress.json'
- fine_domain: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/fine_domain/progress.json'
- fine_span: [Errno 2] No such file or directory: '/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/study_laptop/runs/fine_span/progress.json'

See grid_study.json for per-metric budgets and figures/ for plots.

Conservative engineering estimates; block intervals are not rigorous confidence bounds or independent proof of the asymptotic range.
