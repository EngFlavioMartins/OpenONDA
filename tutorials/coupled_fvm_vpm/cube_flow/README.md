# Native FVM–VPM cube flow

This Re=1000 cube case uses OpenONDA's adaptive Cartesian FVM mesher and the
FVM–VPM coupler. It has no external mesher or repository-path requirement.

After installing OpenONDA, run the configured cube case with:

```sh
./allrun.sh
./allplot.sh
```

`allrun.sh` has no run modes or command-line flags. It runs the `t=20` case
defined in `cube_flow_setup.py`, checks numerical integrity for the complete
run, and applies the drag/profile physics gate through `t=2.0`. By default it
uses the checked-in full-horizon archive in `reference_flow/`; the archive must
contain forces and both line profiles through `t=20`, and `allrun.sh` never
regenerates or modifies it. `OPENONDA_CUBE_REFERENCE` may point to another
complete reference directory when an explicit override is needed.
The early gate requires every sampled drag-coefficient error and every
authority-stitched line profile's spatial mean velocity error to stay within
`7%`; pointwise maxima remain diagnostic because the accepted historical run
contains large, localized errors at the moving near-body wake feature.
The run removes the previous generated outputs before starting.
The preserved baseline samples remain in `samples_archive/`; the archived
full-horizon reference samples are in `reference_flow/samples/`. `allplot.sh`
compares every coincident archived time through `t=20`. Run `./allplot.sh pdf`
when PDF output is required; PNG is the default.

The pre-change short-run physics and performance benchmark is recorded in
[`baselines/2026-08-25_common_m4_gbd_t1p00`](baselines/2026-08-25_common_m4_gbd_t1p00/README.md).

The case uses four partitioned FVM ranks. Its coupled MPI initialization
includes the body-wall geometry gathers used during transfer setup.

The inputs use the wall-commensurate particle spacing `h=0.03125`, an
FVM time-step size of `0.010 s`, and the historically validated VPM/coupling
time-step size of `0.050 s`. The coupler therefore advances five FVM substeps
per VPM step. All force, line, and surface samplers use the single
`SAMPLING_INTERVAL_TIME` defined in
`cube_flow_setup.py`; FVM and VPM backups are written every `0.5 s`. The
fully meshed reference uses the same `0.010 s` FVM time step and `0.050 s`
sampling interval. FVM visualization and the atomic coupled restart checkpoint
are written every `0.5 s`; the native VPM checkpoint writer is disabled because
it would capture the pre-replacement state. Coupled VPM checkpoints use `f32`
storage and omit the derived velocity-gradient tensor; a restart recomputes that
field. Matching the transient time discretization is required for the reference
comparison. The coupled and reference FVMs use the same pressure corrector
counts. This production transfer uses GBD.

The FVM-to-VPM hand-off is the recovered buffered whole-belt M4' renewal used
by the historically stable long run. After VPM advection/stretching/GBD and the
FVM substeps reach the same new time, the coupler samples the synchronized FVM
velocity trace on one fixed regular VPM lattice. It remeshes every post-GBD
particle inside the buffered belt onto that lattice, blends represented VPM
and FVM states with an inward six-spacing authority ramp, prunes weak lattice
nodes, recovers circulation and linear impulse, and atomically replaces the
belt. Particles beyond the buffer remain untouched as the persistent outer
wake. The buffer contains complete M4' support plus the distance needed to
advect between replacements, preventing particles from being trapped and
deleted at the FVM authority face. Boundary-only panel strengths are refreshed
from the updated particle state before their harmonic velocity is used for the
next FVM boundary interval.

The production lattice is half-grid phased: there is no particle plane at the
downstream face `x=1.25`. With `h=0.03125`, the last FVM-authoritative plane is
`x=1.234375`, the first VPM-only release plane is `x=1.265625`, the last regular
input plane inside the physical renewal belt is `x=1.359375`, and the first
persistent plane is `x=1.390625`. M4' support can populate target planes through
`x=1.421875`; the allocated guard endpoint is `x=1.453125`. If support from an
already persistent regular node, the two co-located strengths are combined so
the seam cannot create duplicate particles.

For short or restartable diagnostics, use
`assets/run_trial.py --coupling-steps N`. This is an execution-only limit: it keeps the configured
20-second horizon unchanged, writes an atomic checkpoint at the clean stop,
and therefore preserves strict checkpoint hashes for continuation.
