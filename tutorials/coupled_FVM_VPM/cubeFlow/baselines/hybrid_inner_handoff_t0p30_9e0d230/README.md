# Rejected inward-handoff gate

This gate separated the vorticity handoff from the numerical boundary by
moving the handoff from ±1.5 m to ±1.2 m while leaving the FVM boundary fixed.

- Source checkpoint: `9e0d230`
- Particle count at the third VPM-BC evaluation: 38,850 (35% below the coincident-box run)
- Cd at 0.15 s: 1.831370 (reference 1.952801, -6.22%)
- Cd at 0.30 s: 1.534548 (reference 1.610456, -4.71%)

The placement was rejected because the reconstruction surface was too close to
the cube. The general separate-handoff capability is retained; the next gate
keeps the proven ±1.5 m handoff and moves the numerical boundary outward in a
coarse FVM-only buffer.
