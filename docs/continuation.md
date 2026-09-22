# Continuing simulations

Every tutorial uses `START_FROM = "latest"` in `setup.py`:

```bash
./allrun.sh       # ./allclean.sh, then the tutorial's setup commands
./allcontinue.sh  # the same setup commands, with no cleanup
python setup.py   # start or continue just the default case
```

Variant arguments select that variant's own solution and sample directories.
The reference-flow and rotor-study subdirectories have the same launchers.
The cylinder campaign remains available through `assets/run_pipeline.py`;
the ordinary cylinder launchers run `setup.py` directly.

An empty solution starts from its initial conditions. An existing solution
restores the latest committed numerical backup and advances to the configured
destination. FVM and coupled cases use the configured end time; VPM uses the
**total** `RunPlan.steps`, including restored steps. Repeating a completed run
does not add steps. Extend that destination to run longer.

## What is restored

FVM restores its velocity, pressure, fluxes, multistep history, accepted clock,
adaptive timestep and acceptance state. The configured `BackupConfig.path`
(normally `solution/backup`) is an atomic rolling file in serial execution or
a directory with a committed manifest in partitioned MPI execution. All ranks
participate in restoration; the partitioned format requires the original MPI
rank count.

VPM selects the greatest accepted step among `solution/vpm/vpm_*.h5` (or the
case variant's solution directory). HDF5 backups include particle state,
diffusion/stabilization state and random-walk state. When VLM is attached,
the same backup also carries the moving surfaces, circulation and wake history.
Temporary HDF5 files are ignored. The older flat VPM directory is also readable.

Coupled FVM/VPM runs restore `solution/backups/manifest.json` and its authenticated
FVM, VPM and boundary-history artifacts together. Independent VPM visualization
frames never determine the coupled restart point. The manifest is committed
last, so an interrupted new save leaves the previous committed bundle usable.

Tutorials save periodically and at their destination, including an initial
restart point. Work after the latest successful backup is replayed. Backup
cadence controls how much work an abrupt interruption can lose. A stopped
resource-limited run can continue when resources are available; restoring a
backup does not fix a numerical instability or an unchanged physical limit.

## Samples, diagnostics and logs

Restoration reconciles native CSV/JSONL histories and ParaView collections to
the accepted backup time. Rows after that time, including incomplete appended
tails, are removed from the active history before replay. Superseded histories,
discarded VTK frames and previous invocation metadata are retained under
`restart-branches/`.
Discarded native VPM/VLM backup frames are archived outside the active series
so they cannot become a later automatic restart candidate. Missing scheduled
VPM samples at the committed checkpoint are regenerated. Logs append across
invocations and report the restored clock.

Tutorials with custom FVM histories use `solver.reconcile_history(filename)`
before appending their own rows. Unrecognized files in shared sample directories
are left alone. Regenerate plots after continuation to reflect the active data.

## Compatibility and explicit control

The mesh and numerical configuration must remain compatible with the saved
state. Changing the physical model, timestep, geometry or discretization may
require a new case. Corrupt/incompatible committed backups are errors: automatic
continuation never falls back to an older state or silently starts over.
Visualization-only results from older runs cannot restore numerical history;
restore a numerical backup or use `./allrun.sh` for a clean simulation.

The solver APIs support `solver.run(start_from="latest")` and an explicit backup
path. `start_from="initial"` requires a clean solution directory. FVM and VPM
also expose `solver.start_from(...)` for custom loops, returning whether a backup
was restored. Set custom initial fields before this call; write initial output
only when it returns false. FVM's `BackupConfig` and VPM's `Backup` still control
periodic cadence. Tutorials declare those policies explicitly.

Calling `run()` without `start_from` retains the programmatic in-memory API;
in particular, VPM then runs the requested number of additional steps. Explicit
`load_state`, `load_backup`, and controlled changed-timestep research APIs remain
available. Tutorial continuation no longer needs a `--restart-from` argument or
hard-coded checkpoint filenames.
