# Rejected coarse all-face FVM buffer

This gate kept the handoff at ±1.5 m and added a 0.30 m coarse FVM shell on
all faces before the numerical boundary.

- Source checkpoint: `6bec632`
- Mesh cells: 717,192 (compact baseline: 574,994)
- Actual fitted cube bounds: ±0.50625 m (compact baseline: ±0.503165 m)
- Cd at 0.15 s: 1.833198 (reference 1.952801, -6.12%)
- Cd at 0.30 s: 1.538237 (reference 1.610456, -4.48%)

The dyadic coarse-to-fine transition changed the fitted cube geometry and was
rejected before a long run. The next gate preserves uniform compact resolution
and adds a downstream-only overlap buffer.
