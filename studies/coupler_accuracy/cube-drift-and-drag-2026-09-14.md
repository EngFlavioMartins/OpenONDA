# Cube drift and reference drag diagnosis — 14 September 2026

The reference drag spikes have a reproduced numerical trigger: clipping an
adaptive step to a scheduled output leaves a very small remainder. The time
controller has been corrected. The reason for the growing hybrid/reference
field difference is less settled; internal agreement does not establish
agreement with the independently computed reference.

## Evidence from the saved fine reference

Source files: `tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/solution/fine/diagnostics.jsonl`
and `reference_flow/samples/fine/forces_history.csv` (completed at t=30 s).
The registered requested mesh target is 0.06 m.
The force history and compact diagnostic/field-difference tables are preserved
under [results/cube-drift-and-drag-2026-09-14](results/cube-drift-and-drag-2026-09-14/),
so this evidence survives a fresh reference run.

| State | Time [s] | Accepted dt [s] | Courant max | Pressure min | Pressure max |
|---|---:|---:|---:|---:|---:|
| Before spike | 8.3999870373 | 0.01033396442 | 0.433268 | -0.439475 | 0.507614 |
| Spike | 8.4 | 0.00001296270018 | 0.000543351 | -46.337239 | 16.732002 |

Cd at the spike is 9.557579. Its pressure-force contribution is 4.778355 N,
while its viscous contribution is only 0.000434 N. The pressure span grows
66.6 times as dt falls 797 times. Velocity magnitude remains approximately
1.283 m/s. These are raw solver outputs, not interpolation or plotting errors.
Other large Cd spikes occur at 1.20, 6.70, 11.70, 18.05 and 21.30 s, also
with extremely short steps at scheduled force-output times.

The previous controller used `min(CFL_step, next_event_time - time)`.
At t=8.389653072879867 this produces consecutive steps of
0.010333964419952 and 0.000012962700183 s. The second step exists solely to
reach the 0.05 s sampling boundary. The subsequent 0.05 s interval takes 36
steps because permitted timestep growth is capped at 20% per step.

The pressure projection contains inverse-timestep transient terms. A sharp
reduction exposes finite cell/face and splitting discrepancies as a pressure
impulse. A 64-cell **3D** channel with the same backward/Gauss/linearUpwind
schemes and PIMPLE corrector counts reproduces this with direct linear solves
and 1e-10 tolerances: pressure span rises from 0.0326 to 2.0797 when the tiny
step is imposed. Splitting the same interval into two 0.00517346356 s steps
keeps the span at 0.0343. This isolates the numerical trigger without advancing
either cube simulation or attributing it to a failed iterative linear solve.

The corrected controller partitions the remaining interval into an integer
number of steps below the current CFL ceiling, recalculating that ceiling
after each accepted state. It retains scheduled times, the 0.9 Courant target,
the 0.04 s maximum timestep, the mesh and the numerical schemes. This is the
same scheduling principle used by [OpenFOAM's event alignment](https://cpp.openfoam.org/v13/Time_8C_source.html#l00070),
with no deliberate increase beyond the supplied CFL ceiling.

Regression coverage is in `tests/fvm/test_time_step_control.py`, including
the exact saved time-controller sequence, overlapping output schedules and
the 3D pressure-impulse test. The corrected full fine cube run has **not**
been performed; its force history must confirm the result at production scale.

## Why both coupled fields can drift together

The VPM field inside the overlap is renewed from the coupled FVM field after
each coupling interval, and VPM supplies its next boundary data. Their errors
are therefore correlated. Converging this exchange proves consistency of that
coupled calculation, not agreement with the reference.

Common-grid, area-weighted, three-component velocity differences at z=0:

| Time [s] | Coupled FVM vs VPM [% Uinf] | Reference FVM vs VPM [% Uinf] | Reference FVM vs Coupled FVM [% Uinf] |
|---|---:|---:|---:|
| 1 | 2.23 | 2.47 | 1.78 |
| 5 | 2.78 | 4.56 | 4.12 |
| 8 | 4.40 | 9.07 | 10.39 |
| 12 | 3.71 | 16.44 | 17.13 |
| 14 | 9.97 | 23.64 | 22.94 |

The saved reference remains almost reflection-symmetric about y=0. At t=14,
the RMS difference from the correctly reflected velocity vector is 0.036%
Uinf for the reference, 26.73% for coupled FVM and 22.75% for VPM. Reflection
reverses the y component, rather than comparing speed alone. Decomposing the
coupled/reference difference by this symmetry gives 13.36% antisymmetric and
18.65% symmetric RMS components. Thus differing asymmetry/onset accounts for
about 34% of the squared difference on this slice, and does not explain it all.

There is also an unresolved handoff discrepancy. At t=14 the mean full-vector
velocity mismatch on the downstream coupling face is 12.08% Uinf and its
maximum is 185.93%, despite normal-velocity agreement near roundoff. The
`vorticity_mixed` condition prescribes normal velocity and tangential normal
gradient; it does not prescribe the tangential velocity itself. All 309 saved
interface exchanges through t=15.45 converge their iteration criterion, so
iteration convergence is not a bound on that full-vector mismatch.

These observations make feedback/representation error and different onset of
asymmetry plausible contributors. They do not establish which contribution is
dominant, nor prove that the pressure spikes explain all velocity drift.
Recompute the reference with the confirmed scheduling correction first. A
subsequent transfer-only test using reference snapshots as donors can measure
the reconstructed VPM velocity before time integration or boundary feedback;
that is a better basis for choosing a coupling change than matching the two
already-correlated overlap fields more tightly.

## Subsequent coupled stop at t=15.5 s

The run stopped at VPM step 310 after the particle replacement and interface
iteration. The accepted-state health check refreshes the combined particle
velocity gradient before evaluating `dt * max_p ||S_p||_infinity`, where
`S = (grad(u) + grad(u).T) / 2`. The reported value was 1.05 at dt=0.05 s,
above the unchanged limit of 1. Interface iteration convergence is a separate
criterion and cannot establish this bound.

The reported strain corresponds to approximately 21 /s, requiring dt below
about 0.0476 s at that frozen state. The tutorial now uses dt=0.01 s for VPM
and coupling, giving 0.21 at the reported strain. This is the largest uniform
step below 0.05 s that is an integer multiple of the existing 0.01 s FVM step
and still hits every 0.05 s sample. Backup and detailed transfer diagnostic
intervals remain 0.5 s. FVM mesh, timestep, schemes and LES settings are
unchanged. VPM advances and coupling exchanges per simulated second increase
fivefold; this is a stability correction, not a speedup claim.

An analytical trace-free gradient with infinity strain norm 21 /s reproduces
the health rejection at dt=0.05 s and passes with 0.21 at the new dt. This is
a check of the guard and configuration, not a replay of the failed particle
state. The failed state has no saved VPM field; the last saved VPM field is
at t=15 s. The available output cannot establish whether transfer or the VPM
predictor caused the final crossing. No cube simulation was advanced in this
check, and full-run stability and the earlier reference discrepancy remain
to be evaluated on new results.

## Rerun and plotting

A follow-up check on 15 September confirms that the current fine force history
is byte-for-byte identical to the preserved pre-fix result: 601 samples through
t=30 s, SHA-256 `885b2963b6caef3567ea55d996ad3a3ae1562fb0ffd4d7202c7b3c44ac45ae09`.
Replotting these samples necessarily retains their original pressure spikes.
The raw diagnostic records still show the 797-fold timestep reduction and
66.6-fold pressure-span increase at t=8.4 s.

Fresh isolated imports now resolve the solver and corrected time controller to
this checkout without `PYTHONPATH`; MPI and thread management are internal.
All 15 tests in `tests/fvm/test_time_step_control.py` pass, including the exact
saved controller sequence and the 64-cell 3D pressure-impulse regression.
The current controller divides the problematic interval into two steps of
0.00517346356 s. A full corrected fine run remains necessary to verify its drag.

Preserve the existing `reference_flow/solution/fine/` and
`reference_flow/samples/fine/` directories before starting a fresh run from
t=0. Other grid results need no changes. For only the fine case, from
`reference_flow/`, use:

```bash
python setup.py --name fine --dx 0.06
```

This investigation has not removed or restarted any simulation.
The plane figures now use exactly Coupled FVM vs VPM, Reference FVM vs VPM,
and Reference FVM vs Coupled FVM. The reference label is "Reference FVM".
