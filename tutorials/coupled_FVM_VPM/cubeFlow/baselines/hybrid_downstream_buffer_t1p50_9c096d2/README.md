# Grid-aligned downstream-overlap gate

Successful production gate with separate handoff and numerical outflow planes.
The FVM handoff remains at x=1.5 m while eight exact compact-grid cells extend
the numerical boundary to x=1.803797468 m.

- Source checkpoint: `9c096d2`
- Mesh cells: 624,922 (compact baseline: 574,994, +8.7%)
- Fitted cube bounds preserved exactly at ±0.503164556962 m
- VPM velocity plus momentum-equation pressure-gradient VPM BC
- Cd error from 0.30 through 1.50 s: max 0.846%
- Cd at 1.50 s: 1.091121 (reference 1.081964, +0.846%)
- Final particle count: 287,708

The impulsive-start point at 0.15 s remains -5.25%. No extra plots were
generated; the force history and scheduled 0.60/1.20 field samples are stored.
