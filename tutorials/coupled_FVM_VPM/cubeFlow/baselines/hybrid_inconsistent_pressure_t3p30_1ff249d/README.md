# Failing FVM-physics baseline through t = 3.30

This run uses the compact downstream-buffer cube configuration with bounded
local transfer gain 1.8. It was stopped after the fully written 3.30 s force
sample once the separated-shear-layer discrepancy had been reproduced.

- Source checkpoint before the local fixes: `1ff249d`
- FVM-to-VPM handoff gain: 1.8
- VPM-BC resynchronization after handoff: disabled
- Pressure-gradient VPM-BC body contribution: disabled
- Off-axis reference minimum at 2.40 s: -0.07109 at x/D = 0.32
- Off-axis hybrid FVM minimum at 2.40 s: 0.48398 at x/D = 0.273
- Cd at 2.40 s: 0.91709 (reference 0.83987, +9.20%)
- Cd at 3.00 s: 0.94768 (reference 0.88333, +7.29%)

The run is retained as the acceptance baseline for fixing the physics inside
the coupled FVM subdomain. Raw line/surface samples and coupler diagnostics are
stored here; the large FVM volume files are intentionally omitted.
