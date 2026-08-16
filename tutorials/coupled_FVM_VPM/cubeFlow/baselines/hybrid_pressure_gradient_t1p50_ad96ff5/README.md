# VPM pressure-gradient production baseline

Stopped production run of the hardcoded compact `allrun.sh` configuration after
the first late-time Cd miss established that pressure-gradient transfer alone
does not remove the hybrid/reference drift.

- Source checkpoint: `ad96ff5`
- Last completed time: `1.50 s`
- FVM cell size: `0.038 m`
- VPM spacing and time step: `0.04 m`, `0.05 s`
- VPM integration: RK2
- FVM extent: `x = [-1.5, 1.5] m`
- Donor condition: Dirichlet VPM velocity plus VPM pressure gradient
- Cd error: -5.014% at 0.15 s, within 0.71% from 0.30 through 1.35 s,
  then +1.800% at 1.50 s

The run was interrupted immediately after the 1.50 s handoff. The archived
force history and 0.60/1.20 velocity samples are complete; no extra plotting
was generated.
