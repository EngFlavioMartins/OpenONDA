# Moving-mesh open issues (2026-10-15)

State after commit `2283db79` (rigid FVM mesh motion and coupling on translating
meshes). Verified so far: free-stream preservation, Galilean invariance
(FVM alone and coupled with a body, ~1e-11 and ~1e-6), Wang (1968) added mass
and drag of the oscillating cylinder (Ca 1.12 vs 1.13, Cd within 1.8 %), MPI
partitioned = serial, bit-exact FVM restart, coupled restart.

## 1. Coupled heaving cylinder not yet validated to the end

`tutorials/coupled_fvm_vpm/03_oscillating_cylinder` was compared with its
`reference_flow` only up to t ≈ 18 s: mean Cd within 0.1–0.9 %, Cl range within
0.8–2.3 %. Statistics are meant to be taken from 25 s (locked-in wake).

- Run both to 50 s and compare mean Cd, Cl amplitude, lift frequency / heave
  frequency and lift phase (printed by `./allplot.sh`).
- Fill the missing **Results** numbers in both READMEs and commit the figures.
- Check lock-in: f = 0.18 Hz is near the natural shedding frequency
  (St ≈ 0.18–0.19 at Re = 150); the lift frequency ratio should be 1.000.

## 2. Rotating meshes cannot be coupled

`MeshFrame.validate` raises `NotImplementedError` for any rotating or pitching
motion, because the VPM has no rotating frame (no Coriolis/centrifugal terms,
no rotating domain, diffusion grid or samplers). Options:

- solve the particles in the inertial frame and move the FVM box geometry
  (coupler regions, transfer region, wall triangles) every exchange, or
- add a non-inertial frame to the VPM.

The first is likely simpler and also removes issue 3.

## 3. VPM quantities are in the translating frame

Particle positions, `domain_bounds`, VPM output and VPM samplers are relative to
the box at t = 0. Users must add `mesh_translation`
(`solution/coupler_diagnostics.jsonl`) to get inertial positions. Post-processing
and ParaView states do not do this yet. Large mean translations (a body towing
through still fluid) also need a VPM domain that moves with the box, which the
frame gives for free, but the outflow fringe then sits relative to the body.

## 4. Freestream far field on O-grids is unstable

The circular `freestream` far field becomes unstable once the wake reaches it
(also on static meshes). The tutorials avoid it by splitting the far field into
inlet and outlet halves. A separate task was spawned: "Fix freestream far-field
instability on O-grids". Related: freestream inflow/outflow switching uses the
relative flux; under strong flow reversal at the far field (large heave
velocity) the switching may chatter.

## 5. Time-accuracy of the PISO/PIMPLE loop on moving meshes

- With one outer corrector, the face flux lags the mesh flux by one step;
  for fast changes of the relative flow (high-frequency heave, impulsive
  starts) the error is first order. Recommend ≥ 2 outer correctors in docs or
  measure the error vs. outer correctors on the Wang case.
- The `ddtCorr` limiter is evaluated with the relative flux; under rotation it
  is not frame invariant (only translation tested).
- Galilean tests require a projected, divergence-free start; an impulsive start
  violates discrete continuity and the conservative convection term then depends
  on the frame. Consider projecting the initial field when mesh motion is set.

## 6. Wall pressure on accelerating walls

Walls keep `zeroGradient` pressure. For an accelerating wall the correct
condition is ∂p/∂n = −a·n (or `fixedFluxPressure`), so the near-wall pressure is
first order in the acceleration. Wang's added mass agrees to 0.8 %, but check at
larger KC/β or switch moving walls to `fixedFluxPressure`.

## 7. Adaptive time step stalls with `every_time` samplers

Pre-existing: with adaptive dt and `RunSchedule(every_time=...)` the step is
clipped to the next output time and can collapse. The moving-mesh tutorials use
a fixed dt to avoid it.

## 8. Missing motion features

- No deforming meshes (only whole-mesh rigid motion).
- No multiple independent bodies / overset zones.
- No motion of a sub-region (e.g. rotating zone with sliding interface).
- No prescribed motion from a table (time series file) — only analytic kinematics.

## 9. Mesher cost for the coupled heaving case

Meshing `03_oscillating_cylinder` took > 30 min in mesh optimisation versus
~145 s for the same box in `01_cylinder_shedding_flow`. Investigate (same STL
and box; possibly the slab bounds or the cached-mesh hash). The reference case
also reported a cached-mesh hash mismatch on rerun — check that the moving-mesh
setup does not enter the mesh hash unnecessarily (`fvm/io/backup.py` uses
reference positions).

## 10. Restart reproducibility

FVM restarts are bit-exact when backups fall on aligned steps; otherwise
differences of ~1e-8 to 1e-9 come from the AMG preconditioner history, which is
not saved. Coupled restart is compared with `atol=1e-6/1e-7`.

## 11. Coupled cost

The coupled heaving case costs ~2.7–5 s per exchange (6 interface iterations ×
5 FVM steps + FMM slab induction). Moving-mesh overhead itself is small (geometry
rewrite + boundary stencil cache rebuild only on rotation), but profile once the
validation is done.
