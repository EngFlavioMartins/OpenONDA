# VPM pressure-gradient coupling gate

Short production gate for the compact cubeFlow configuration using the VPM
momentum-equation pressure gradient on the FVM numerical boundary.

- Source checkpoint: `dd2ce9d`
- End time: `0.30 s`
- FVM cell size: `0.038 m`
- VPM spacing and time step: `0.04 m`, `0.05 s`
- VPM integration: RK2
- FVM extent: `x = [-1.5, 1.5] m`
- Particle count after the first handoff: 55,905
- Cd at 0.15 s: 1.854895 (reference 1.952801, -5.01%)
- Cd at 0.30 s: 1.610171 (reference 1.610456, -0.018%)

The pressure donor excludes panel-body induction and the viscous Laplacian at
the outer interface. Panel blockage remains included in the velocity donor.
