# Continue or extend a simulation

From a tutorial directory:

```bash
python -m openonda.tutorial_runner . setup  # Run or resume the default case.
./allcontinue.sh      # Run or resume the tutorial's case set.
./allrun.sh           # Run the case launcher; most clean previous output.
```

Tutorials set `START_FROM = "latest"` in `setup.py`. If a numerical backup exists, the solver restores it; otherwise it starts from the configured initial conditions. A completed run adds no steps. To extend it, increase the FVM/coupled end time or the VPM **total** `RunPlan.steps`.

Set `START_FROM = "initial"` to start again with the configured initial conditions. Previous solver output is moved into `restart-branches/`. Most `allrun.sh` launchers call `allclean.sh` and delete generated output; the coupled cylinder, cube, and their FVM references preserve results and resume. Variant arguments resume that variant's own case; see the [tutorial index](tutorials.md).

## Preserve the physical case

Keep the mesh, geometry, boundary conditions, viscosity, particle distribution, and numerical model compatible with the saved state. Use a separate case directory when changing these inputs for a comparison. An incompatible or corrupt backup raises an error.

| Solver | Numerical backup |
| --- | --- |
| FVM | `BackupConfig.path`, normally `solution/backup`; MPI backups require the original rank count. |
| VPM/VLM | Latest accepted `solution/vpm/vpm_*.h5`; includes particle and attached lifting-surface state. |
| FVM–VPM | `solution/backups/manifest.json` and both component states; keep the bundle together. |

ParaView files alone cannot restore a simulation. Backup cadence determines how much work is repeated after an interruption. Continuation preserves the model; it does not resolve a numerical instability.

Restarts require the current checkpoint schemas: FVM serial version 10, FVM partitioned version 8, and coupled version 12 with boundary-state schema 4. Checkpoints are admitted without schema migration, and numerical configuration changes require explicit permission.

## After resuming

The solver trims histories and visualization frames to the restored time before advancing again. Regenerate plots with `./allplot.sh`, where provided, after extending a run. See [saved fields](solution_layout.md) and [physical comparisons](visualization.md).

For a custom script, call `solver.run(start_from="latest")` on a newly configured solver. Without `start_from`, VPM's programmatic `run()` advances the requested number of additional steps instead of treating the step count as a total destination.
