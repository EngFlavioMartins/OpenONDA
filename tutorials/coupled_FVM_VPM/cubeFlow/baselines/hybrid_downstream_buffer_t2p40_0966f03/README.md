# Grid-aligned downstream-overlap full run

Completed production run with separate handoff and numerical outflow planes.
The FVM handoff remains at x=1.5 m while eight exact compact-grid cells extend
the numerical boundary to x=1.803797468 m.

- Source checkpoint: `0966f03`
- Mesh cells: 624,922 (compact baseline: 574,994, +8.7%)
- Fitted cube bounds preserved exactly at ±0.503164556962 m
- VPM velocity plus momentum-equation pressure-gradient donor
- Cd error from 0.30 through 2.40 s: MAPE 2.190%, RMS 3.276%
- Cd points within 1% from 0.30 through 2.40 s: 9/15
- Cd at 1.50 s: 1.091110 (reference 1.081964, +0.845%)
- Cd at 2.40 s: 0.896217 (reference 0.839868, +6.710%)
- Final particle count: 393,026 (previous compact run: 506,187, -22.4%)
- Total measured runtime: 2,411 s (previous compact run: 2,490 s, -3.2%)

The outer VPM profiles remain accurate: their RMS velocity error is below
0.53% at every scheduled sample. The late discrepancy is localized to the FVM
near-body wake; at t=2.40 s its off-axis RMS error is 17.28%, while the stitched
profile is 4.57%. The impulsive-start Cd point at 0.15 s remains -5.25%.

No extra plots were generated. The force history, diagnostics, and scheduled
0.60/1.20/1.80/2.40 field samples are stored here.
