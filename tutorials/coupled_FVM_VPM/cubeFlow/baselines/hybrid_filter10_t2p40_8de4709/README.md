# Hybrid cube-flow baseline: face filter x10

Completed successfully on 2026-08-16 from commit `8de4709` with the
production-resolution configuration hardcoded in `allrun.sh`:

- end time: 2.4 s
- FVM time step: 0.01 s
- VPM time step: 0.05 s (RK2)
- particle and FVM cell spacing: 0.04 m
- face-aware overlap-shell prune multiplier: 10
- final retained VPM population: 490,000 particles

The `samples/` directory contains the force history and sampled velocity
profiles/fields.  The `solution/` directory contains lightweight logs,
metadata, and coupling diagnostics; large restart and volume files are not
archived here.

For the 13 Cd samples from 0.6 through 2.4 s, relative to `referenceFlow`:

- mean absolute percentage error: 3.139%
- RMS relative error: 3.714%
- mean relative bias: +1.117%
- final relative error at 2.4 s: +6.527%

This baseline reduced Cd error and particle count relative to the unfiltered
`hybrid_t5p40_cbb76df` run, but it does not meet the 1% validation target.
