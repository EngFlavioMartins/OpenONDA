# Hourly cylinder campaign monitoring

Automation: `monitor-cylinder-grid-study`, hourly. An initial coarse-build launch
was deliberately interrupted to satisfy the pinned formatter before committing.
Its logs and incomplete campaign were preserved in
`artifacts/reference-flow-laptop-20260906/launch-before-format*` at repository
root. That interruption is not a numerical failure. The final launch uses the
fresh default `study_laptop/` directory and committed source snapshot.

Read `study_laptop/status.json` first. It records the current case, phase, PID,
last update, checkpoint time where available, and final failure/verdict. Only
read a short tail of `study-launch.log`, `study_laptop/mesh-<case>.log`, or
`study_laptop/flow-<case>.log` if the status warrants investigation. A checkpoint
is written every 5 physical time units; lack of a fresh checkpoint by itself
does not demonstrate a stalled solve. Check the log and live process identity.

Default command, from this directory:

```bash
PYTHON=/opt/anaconda3/envs/OpenONDA/bin/python ./allrun.sh
```

Do not run this again while a process owns the campaign lock. Compatible runs
resume from published checkpoints; changed source/configuration requires a new
`--output` directory, preserving the old campaign. Point monitoring at that new
directory explicitly if a justified repair requires one. Do not automatically
delete or replace the failed campaign to evade provenance checks.

Every case exports `solution/<case>/mesh.vtu` and `mesh.npz` under its campaign
directory before FVM admission. `samples/<case>/` holds its force history.
Rejected generated meshes also have backups, with an unqualified marker.
The accepted mesh manifests are separately under `meshes/<grid>/`.

Known pre-launch blocker: the 266,429-cell enlarged-domain diagnostic mesh
passes independent checkMesh at 1.74 GiB peak meshing RSS, but OpenONDA rejects
LSQ 11.67 against production limit 9. Do not loosen this gate or repeatedly
regenerate unchanged rejected meshes. Follow the operator-qualification/local
geometry repair gate in the root `REFERENCE_FLOW_QUALIFICATION_PLAN.md`.
The heaviest production FVM memory/time has not yet been measured.

Stay quiet for healthy progress or unchanged known blockers. Diagnose only a
new actionable failure; small repairs require focused tests. Never change
physics, accuracy tolerances, quality gates or resource budgets merely to get
a passing result. A substantive numerical defect needs an explicit evidence
report, not repeated blind retries. On completion inspect `REPORT.md`, the
Cd/Cl/St statistics and plots; then pause the hourly automation. An inconclusive
study is not a grid-independent result.

The user requested a project-wide commit. Existing VPM edits and plan deletions
are included in that snapshot; ignored binary meshes/checkpoints remain on disk.
