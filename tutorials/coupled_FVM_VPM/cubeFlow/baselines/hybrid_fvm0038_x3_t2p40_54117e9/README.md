# Rejected downstream-extended FVM-box run

This production-resolution run extends only the downstream FVM boundary from
`x = 1.5` to `x = 3.0`.  All other recommended settings match commit
`54117e9`: FVM cell size `0.038`, particle spacing `0.04`, `dt_fvm = 0.01`,
`dt_vpm = 0.05`, RK2, and overlap-shell pruning multiplier 10.

The run completed through 2.4 s with 926,600 FVM cells.  Moving the interface
outside the strong mean wake deficit improved the stitched velocity profiles,
but worsened pressure drag and total runtime:

- Cd for `t >= 0.3`: 4.489% MAPE, 5.430% RMS, +4.489% bias, and 1/15 samples
  within 1% (compact-box baseline: 2.732%, 3.972%, +2.611%, and 8/15).
- Final Cd error: +9.463%; pressure force was 0.459856 versus 0.417003 in the
  fully meshed reference, while the viscous contribution was negligible.
- Final stitched centerline velocity MAE/RMS: 0.752%/1.476% (compact box:
  0.990%/1.693%).
- Final stitched off-axis velocity MAE/RMS: 1.990%/7.023% (compact box:
  2.657%/9.617%).
- Summed coupling-step time: 2710.9 s, 8.9% slower than the compact-box run.
  VPM time fell from 1476.0 s to 1123.8 s, but FVM time rose from 806.9 s to
  1326.2 s.  Final injected particle count was 425,733.

Conclusion: the wake is not inaccurate merely because the compact interface
cuts the recovery region.  Velocity improves when the FVM region is enlarged,
while the elliptic pressure error grows.  The next experiment should retain
the compact box and change only the downstream-face velocity/pressure closure.

