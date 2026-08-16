# Hybrid cube-flow baseline: 0.038 FVM core

Completed successfully on 2026-08-16 from commit `a311d9f`.  Relative to the
0.04 filtered baseline, only the hybrid FVM maximum cell size changed to
0.038; the VPM particle spacing remained 0.04.

The 574,994-cell mesh improved the stitched velocity field and early drag:

- Cd stayed within 1% of `referenceFlow` at all eight samples from 0.30
  through 1.35 s.
- Cd MAPE over 0.30--2.40 s was 2.732% (0.04 baseline: 3.004%).
- Cd error at 2.40 s was +7.605%; the late error is pressure-drag dominated.
- centerline Ux MAE was 0.489%, 0.669%, 0.872%, and 0.991% at the four
  0.6-s-spaced frames.
- off-axis Ux MAE was 0.295%, 0.472%, 1.325%, and 2.205%.

The final VPM integrator population was 473,432 particles and its 48-step wall
time was 1,327.9 s, versus 490,000 particles and 1,502.0 s for the 0.04
baseline.  The improved early particle count more than offset the modestly
larger FVM mesh inside the VPM integrator.

The run does not meet the global 1% target, but is retained because it gives
the best field accuracy and the strongest evidence that late Cd error comes
from near-body pressure recovery rather than transfer discontinuity.
