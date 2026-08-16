# Rejected coarse-RK3 cadence gate

Cadence experiment using RK3 with a 0.10 s VPM step and ten interpolated FVM
substeps per coupling update. The run was stopped after three consecutive Cd
checkpoints missed the 1% acceptance threshold.

- Source checkpoint: `daed609`
- Completed through t=0.80 s
- Cd error at 0.15 s: -4.318% (RK2 baseline: -5.250%)
- Cd error at 0.30 s: -0.254% (RK2 baseline: -0.067%)
- Cd error at 0.45 s: +1.153% (RK2 baseline: +0.467%)
- Cd error at 0.60 s: +1.235% (RK2 baseline: +0.187%)
- Cd error at 0.75 s: +1.158% (RK2 baseline: -0.370%)
- Measured runtime through t=0.80 s: 457 s (RK2 baseline: 644 s, -29.0%)
- VPM time through t=0.80 s: 111 s (RK2 baseline: 164 s, -32.2%)
- Handoff time through t=0.80 s: 31 s (RK2 baseline: 64 s, -51.0%)
- Final particle count: 219,607 (RK2 baseline: 207,578, +5.8%)

Coarse RK3 is useful when statistical accuracy and throughput dominate, but it
does not preserve the phase-accurate validation trajectory as well as 0.05 s
RK2. No extra plots were generated.
