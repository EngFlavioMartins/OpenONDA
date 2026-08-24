# Focused cube-flow calibration

## Objective

1. Keep every sampled drag and velocity-profile error at or below 5% against
   a fresh `dt=0.005` fully meshed FVM reference during `t > 2 s`.
2. Among passing configurations, minimize median coupled step time and peak
   particle count.

The accepted replacement baseline is fixed: hard absolute FVM-state
replacement (`eta=0`), `vorticity_mixed`, `h=0.03125`, and core-radius ratio
`1.0`. No remeshing pass is added by the coupler.

## Matrix

| ID | One change from baseline | Interval | Purpose |
|---|---|---:|---|
| B0 | none; absolute GBD threshold `0.02 h^3` | `0–2.5 s` | stability, checkpoint seed, baseline accuracy/cost |
| P1 | panel scope `full` | restart `2–2.5 s` | test whether the body potential must advect overlap particles |
| T2 | GBD threshold `0.04 h^3` | restart `2–2.5 s` | prune more weak particles |
| T4 | GBD threshold `0.08 h^3` | restart `2–2.5 s` | aggressive cost bound |

B0 runs first and a checkpoint at `t=2 s` seeds the three controlled
continuations, so their unsteady states are identical before the single tested
change. The reference and variants run sequentially: concurrent solver work
would make their timing measurements scientifically incomparable.

## Selection

- Reject any non-finite field, failed solve, incomplete run, or metric above 5%.
- Highlight the lowest-error configuration and the fastest passing
  configuration separately.
- Adopt a changed setting only if it remains inside the accuracy gate and
  materially reduces cost or error. Otherwise restore the baseline value.
- Record every result and disposition in `COUPLER_INVESTIGATION_LOG.md`.
