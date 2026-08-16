# Rejected grid-aligned all-face overlap gate

Short gate with the handoff box eight exact compact-grid cells inside the FVM
boundary on all six faces. Although the integer-cell expansion preserved the
fitted cube bounds exactly, the candidate degraded the early drag response and
was too expensive to promote.

- Source checkpoint: `69e9731`
- Mesh cells: 939,330
- Fitted cube bounds preserved exactly at ±0.503164556962 m
- Cd at 0.15 s: 1.806780 (reference 1.952801, -7.478%)
- Cd at 0.30 s: 1.538550 (reference 1.610456, -4.465%)
- Downstream-only Cd at 0.30 s: 1.609380 (-0.067%)
- Gate runtime: 280 s
- Final particle count: 62,687

The lateral/upstream shell is therefore not the missing coupling ingredient.
No diagnostic fields or extra plots were generated for this rejected gate.
