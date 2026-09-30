# Continuing simulations

Every tutorial uses `START_FROM = "latest"` in `setup.py`:

```bash
./allrun.sh       # ./allclean.sh, then the tutorial's setup commands
./allcontinue.sh  # the same setup commands, with no cleanup
python setup.py   # start or continue just the default case
```

Set `START_FROM = "initial"` in `setup.py` to ignore existing backups and start
from the configured initial conditions. This does not require a manual cleanup.
Prior solver-owned checkpoints, frames and histories are retired from the active
series into `restart-branches/`; unrelated files and open logs are preserved.
Consequently, a later `"latest"` run selects the new series even when the previous
run had reached a larger step. Invalid old checkpoints are not read in initial
mode. `allrun.sh` instead removes old output using the tutorial's `allclean.sh`
before invoking the same setup commands as `allcontinue.sh`.

Variant arguments select that variant's own solution and sample directories.
The reference-flow and rotor-study subdirectories have the same launchers.
The cylinder launchers run the matched reference/coupled campaign through
`assets/run_pipeline.py`; `python setup.py` runs its single default coupled case.

Without a committed numerical backup, the solver starts from its initial
conditions and retires any previous owned output histories. With a backup, it
restores the latest committed state and advances to the configured
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
Without a committed bundle, rerunning starts from the initial conditions.
An orphan component backup is never treated as a coupled checkpoint.

Tutorials save periodically and at their destination, including an initial
restart point. Work after the latest successful backup is replayed. Backup
cadence controls how much work an abrupt interruption can lose. A stopped run can continue when resources are available; restoring a
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

## Backup preparation and performance

The optimized VPM writer keeps the existing HDF5 schema (10.1), full restart
precision, VPM VTU files, VLM VTP files and PVD collections. Directory names,
frame numbers, output arrays, cadence and atomic commit ordering are unchanged;
existing checkpoints require no conversion. FVM serialization and the coupled
manifest contract are unchanged. Coupled saves use the same VPM field preparation.

Within a scheduled or final output event, VPM can reuse the accepted velocity
already evaluated for health/diagnostics. This is a single-use preparation:
particle revision, clock, background velocity and provider identity must match.
The preparation expires when the event ends, including on failure. Independent
manual saves and coupled coordinator saves evaluate their current transport
field, so changes in external boundary data cannot reuse a previous event's
velocity.

For large float32 Gaussian clouds, backup vorticity uses the source hierarchy to
skip contributions beyond 12 pair-mean core radii, where the Gaussian density
already underflows to zero. All remaining contributions, including self terms,
use the existing Gaussian kernel and individual source/target core sizes.
Summation order can change float32 rounding. Other kernels, float64 and planar
induction keep their existing calculations. Fourier diagnostics retain the same
grid, expansion order and double precision, but reduce quadratic integrals in
bounded blocks and retain penultimate-order scalars instead of duplicate spectra.

Logs distinguish evolution time, accepted-state health, diagnostics, backups and
samplers, and total standalone step time. Each successful backup reports field
preparation and file-writing time separately. The total step includes scheduled
output; evolution time alone does not measure the time between steps.

## Compatibility and explicit control

The mesh and numerical configuration must remain compatible with the saved
state. Changing the physical model, timestep, geometry or discretization may
require a new case. Corrupt/incompatible committed backups are errors: automatic
continuation never falls back to an older state or silently starts over.
Visualization-only results cannot restore numerical history. If no numerical
backup exists, `"latest"` starts from the initial conditions and retires those
old results before writing the new series.

The solver APIs support `solver.run(start_from="latest")` and an explicit backup
path. `start_from="initial"` uses a newly constructed solver's initial state and
starts fresh output even when backups exist. FVM and VPM
also expose `solver.start_from(...)` for custom loops, returning whether a backup
was restored. Set custom initial fields before this call; write initial output
only when it returns false. FVM's `BackupConfig` and VPM's `Backup` still control
periodic cadence. Tutorials declare those policies explicitly.

Calling `run()` without `start_from` retains the programmatic in-memory API;
in particular, VPM then runs the requested number of additional steps. Explicit
`load_state`, `load_backup`, and controlled changed-timestep research APIs remain
available. Tutorial continuation no longer needs a `--restart-from` argument or
hard-coded checkpoint filenames.
