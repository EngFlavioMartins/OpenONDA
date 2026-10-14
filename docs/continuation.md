# Continue or extend a simulation

Every tutorial starts its solver from the latest backup (`start_from="latest"`, set in `setup.py` directly or through a `START_FROM` constant): if a numerical backup exists the run resumes from it, otherwise it starts from the initial conditions. From a case directory:

```bash
./allrun.sh        # delete solution/, samples/ and figures/, then run from t = 0
./allcontinue.sh   # resume from the latest backup
```

A finished run adds nothing when resumed. To extend it, raise the end time in `setup.py` (the value passed to `RunPlan(end_time=...)`), then run `./allcontinue.sh`. Outputs written after the restored time are trimmed first, so the histories stay continuous. Regenerate the figures afterwards with `./allplot.sh`.

In your own script, pass `start_from` to `run()`:

| `start_from` | Behaviour |
| --- | --- |
| `"latest"` | Resume the backup with the highest step, or start fresh |
| `"initial"` | Start a new history from the initial conditions, replacing earlier output |
| a path | Resume from that backup |

## What a backup contains

| Solver | Backup |
| --- | --- |
| FVM | `solution/fvm/fvm_{step:06d}.npz`: velocity, pressure and fluxes. An MPI run writes a directory of that name and restarts with the same core count. |
| VPM/VLM | `solution/vpm/vpm_{step:06d}.h5`: particles and lifting-surface state. |
| Coupled | `solution/coupled/coupled_{step:06d}/`: both solver states and the coupling state of one exchange; keep each folder together. |

`Backup(schedule=...)` sets how often a backup is written, and so how much work an interruption costs; `Backup(write_at_end=True)` (default) writes one at the end of the run. ParaView files alone cannot restore a run.

## Keep the case compatible

A backup restores the numerical state of one physical problem. Keep the mesh, geometry, boundary conditions, viscosity and numerical model unchanged when resuming; the solver refuses a backup whose configuration differs. To compare a variant, copy the case to a new directory. Backups written by a different version of the solver may need that version to restart.
