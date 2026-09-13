# Velocity projection and conservative face flux in 3D

The reference and hybrid both conserve their accepted FVM face flux to
roundoff. Interpolating their cell velocities to faces gives a different flux,
with substantial discrete divergence near the cube. This is a property of
the stored discrete fields in both solvers, not evidence of compressible flow
or a coupling-specific pressure-solver failure. Correcting only the face-normal
velocity recovers the conserved flux without changing native Gauss circulation.
Matching circulation therefore does not, by itself, match these flux data.

A separate decomposition of the preceding continuous reconstruction rules out
another tempting explanation: removing its Biot–Savart velocity projection
would worsen near-body error in all four tested mesh/input combinations.
Neither diagnostic establishes the cause of the advancing drag difference or
qualifies a new production transfer. The requested small-domain force and
velocity-profile agreement remains unachieved.

## Exact replay of the reference

The [reference replay script](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_reference_flux_replay_3d.py)
repeats the matched medium laminar cube reference on its 53,752-cell native
mesh. It retains the original 50 warmup steps, initial-state reset, and 100
subsequent steps with `dt = 0.01`. All saved cell velocities and pressures
reproduce the earlier reference **bit for bit** at all three observations:

| Observation | Physical time | Solver step | Flux state |
| --- | ---: | ---: | --- |
| Warmup | 0.5 | 50 | Accepted pressure-corrected flux |
| Reset initial | 0.5 | 0 | Flux reinitialized from cell velocity |
| Final | 1.5 | 100 | Accepted pressure-corrected flux |

The warmup and reset have identical velocity and pressure arrays but different
fluxes. This deliberately reproduces the original oracle's `set_initial_state`
call. It does not assert that resetting those two fields is an exact continuation
of the accepted warmup state. Canonical backups now retain the face flux and
its histories as well as velocity and pressure.

The [replay record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-reference-flux-replay-medium-laminar/reference-flux-replay-3d.json)
contains each backup hash, the exact field comparisons, settings and archived
source hashes. This run recovers additional reference observations; it does
not change the hybrid's initialization or its previously measured drag.

## Compare the two discrete fluxes on identical cells

The [mass-flux audit](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/native_mass_flux_audit_3d.py)
uses the same 16,936 small-domain cells for every observation, including the
corresponding native cells inside the full reference. Centroids and volumes
are checked against the cell map. All cases use particle spacing `0.0625` and
the small FVM box of approximately `[-1.5, 1.5]^3` around the unit cube.

For native area vectors `S_f`, it compares the stored conservative flux `φ_f`
with the flux of the solver's linear interpolation of cell velocity:

```text
φ_U,f = S_f · u_linear,f
div_cell = (sum of outward-oriented face fluxes) / V_cell.
```

The counterfactual replaces only interior face fluxes with `φ_U`. Stored
physical-boundary fluxes remain fixed. Divergence is computed independently
using cell accumulation, checked against the FVM diagnostic, and checked for
global interior-face cancellation. The three hybrid velocity arrays replay
their earlier saved comparison fields exactly.

| Snapshot | Stored-flux divergence RMS [U∞/D] | Interior-replaced divergence RMS [U∞/D] | Interior normal-velocity difference RMS / U∞ |
| --- | ---: | ---: | ---: |
| Hybrid control, step 100 | 1.21052e-15 | 0.209520 | 0.00577585 |
| Hybrid native derivative, step 100 | 1.16140e-15 | 0.209533 | 0.00577638 |
| Hybrid native velocity and derivative, step 100 | 3.38053e-15 | 0.209389 | 0.00576927 |
| Reference accepted warmup | 1.18898e-15 | 0.258038 | 0.00702539 |
| Reference reset initial | 0.258038 | 0.258038 | 0 |
| Reference accepted final | 1.16225e-15 | 0.207797 | 0.00554302 |

Divergence RMS uses the shared cell volumes. Normal-velocity difference RMS
uses actual areas of interior faces whose two cells both belong to that
shared region. The reset-initial row is an initialization observation, not
an accepted incompressible time step.

For the near-body region `max(abs(cell_centre)) < 0.8`, the interior-replaced
divergence RMS is `0.688325` in the final hybrid control and `0.688998` in the
final reference. The comparable values demonstrate why reconstruction
divergence alone must not be interpreted as a coupling error. The outer
region does differ: its values are `0.0383906` and `0.00515527`, respectively.
That difference is an observation, not an established source of force error.

An independent face-normal adjustment enforces the stored flux:

```text
u_corrected,f = u_linear,f + ((φ_f - φ_U,f) / |S_f|) n_f.
```

It leaves each face contribution `S_f × u_f` unchanged analytically. Across
all six snapshots, the largest measured cell-vorticity change is `1.43e-14`
in units of `U∞/D`. Thus the native circulation and the conservative flux
contain distinct information. This face adjustment does not specify a unique
continuous velocity field or provide a particle-transfer algorithm.

The [combined six-snapshot result](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-native-mass-flux-qualified/native-mass-flux-audit-3d.json)
archives the source files and per-cell/per-face observations. All 23 input/source
hashes and six output-field hashes were checked after completion.

## Recheck the frozen boundary against accepted flux

The [accepted-flux diagnostic](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/accepted_flux_boundary_3d.py)
now evaluates all eight completed medium shared-trace reconstructions against
the warmup's conservative normal velocity. It verifies identical donor cell
velocities, cut-face geometry, orientation and area, and replays every earlier
linear-reference error. This adds a second boundary observation without changing
any reconstructed velocity or source coefficient.

At the 3,456 coupling faces, accepted and linearly reconstructed reference
normal velocities differ by `0.000110019 U∞` RMS and `0.000537174 U∞` maximum.
The accepted cut flux is `-1.11e-16 U∞D²`; the reset-initial linear cut flux is
`-0.00176378 U∞D²`. The candidates retain their existing mass-flux correction.

| Frozen source | Native normal error versus linear velocity / U∞ | Native normal error versus accepted flux / U∞ |
| --- | ---: | ---: |
| Constant native volume | 0.00208091 | 0.00212296 |
| Linear-face point moment control | 0.00163293 | 0.00166381 |
| Point native-face moments | 0.00107833 | 0.00107015 |
| Point cell-weighted quadratic | 0.00115292 | 0.00115448 |
| Point shared-trace quadratic | 0.00117544 | 0.00117688 |
| Mean native-face moments | 0.00107824 | 0.00107005 |
| Mean cell-weighted quadratic | 0.00108833 | 0.00106986 |
| Mean shared-trace quadratic | 0.00112503 | 0.00110500 |

The shared-trace update still worsens normal-velocity error relative to its
native-face-moment baseline in both input families. Both have positive error
inner products with the update direction under the accepted-flux target.
Their derivative target and derivative errors are unchanged. Consequently,
positive damping still cannot improve both native boundary quantities along
either shared-trace update direction. The mean cell-weighted update slightly
changes its ordering in the normal metric, but its derivative still worsens;
this does not qualify that candidate either.

The [complete accepted-flux result](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-accepted-flux-shared-boundary-medium-laminar/accepted-flux-boundary-3d.json)
contains both point and native observations, six-side reference differences,
the exact quadratic response checks, source archives and the saved boundary
arrays. This additional observation does not demonstrate an advancing solver
improvement.
All 14 recorded source/input hashes were checked, and 32 saved normal-error
measurements were independently recomputed with zero difference.

The completed medium continuous-curl comparison adds four more observations
with the same accepted reference flux:

| Common-trace source | Normal error versus linear velocity / U∞ | Normal error versus accepted flux / U∞ |
| --- | ---: | ---: |
| Point, affine moments | 0.001198124 | 0.001210915 |
| Point, continuous curl | 0.001204893 | 0.001218589 |
| Mean, affine moments | 0.001080053 | 0.001071616 |
| Mean, continuous curl | 0.001079626 | 0.001071805 |

The tiny normal-error improvement within the mean affine/curl pair changes
sign under the conservative-flux observation. All four directions worsen
accepted normal-flux error and native derivative error relative to their
native-face-moment baselines, including at every positive damping scale.
The [expanded result](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-accepted-flux-continuous-boundary-medium-laminar/accepted-flux-boundary-3d.json)
retains all twelve source observations. Its 16 source/input hashes were checked,
and 48 normal-error norms were independently recomputed with maximum difference
`4.34e-19`.

## Isolate the velocity projection

The preceding [continuous-curl reconstruction](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/continuous-curl-reconstruction-3d.md)
constructs a compact, continuous piecewise linear velocity correction `v_h`.
Its curl is a locally solenoidal vorticity source, but `v_h` itself need not be
incompressible. The [decomposition script](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/continuous_curl_hodge_3d.py)
independently checks tetrahedral divergence through face integrals and
evaluates both terms of the free-space identity:

```text
v_h = B(curl v_h) - grad N(div v_h),     N = (-Δ)⁻¹.
```

The identity closes to at most `3.66e-14` at 256 near-body and 24 exterior
observations in both physical meshes. The following errors use the same
near-body reference velocity as the paired reconstruction study. The source
column includes the Biot–Savart correction; the final column also includes
the newly solved body-potential response.

| Mesh/input | Baseline velocity error | Add unprojected correction | Add projected source correction | Include body response |
| --- | ---: | ---: | ---: | ---: |
| Coarse, point | 0.128033 | 0.167073 | 0.128120 | 0.127444 |
| Coarse, cell mean | 0.128020 | 0.146551 | 0.122651 | 0.122342 |
| Medium laminar, point | 0.0874466 | 0.134456 | 0.0937200 | 0.0930789 |
| Medium laminar, cell mean | 0.0874479 | 0.119452 | 0.0893521 | 0.0891941 |

All values are velocity RMS divided by `U∞`. The unprojected addition is a
diagnostic counterfactual, not an admissible incompressible transfer proposal.
Projection lowers its error in all four cases. These local errors are not an
orthogonal global error budget and do not predict an advancing force change.
The completed coarse fields replay the paired induction result bitwise.
The full medium induction/body comparison is complete and independently
verified in the [paired reconstruction study](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/continuous-curl-reconstruction-3d.md).

Records: [coarse decomposition](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-continuous-curl-hodge-coarse/continuous-curl-hodge-3d.json),
[medium decomposition](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-continuous-curl-hodge-medium-laminar/continuous-curl-hodge-3d.json).

## Advancing endpoint before and after replacement

The [endpoint audit](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/coupled_boundary_stage_audit_3d.py)
now compares the accepted FVM trace with the VPM boundary history in all three
matched medium trials at physical time `1.5`. The driver advances FVM using
the VPM prediction, replaces particles from that FVM solution, then recomputes
VPM boundary history. It does not re-solve FVM at that endpoint. The two
backups consequently describe different boundary data at the same time.

The first audit attempt required exact equality with the earlier reference
replay and detected a maximum shared-velocity difference of `7.77e-16`.
The earlier reference ends at solver time `1.0`; the live comparison configures
an end time of `20.0` and observes step 100 at `1.0000000000000007`. A
[replay with the live stepping horizon](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_live_reference_flux_replay_3d.py)
now reproduces every saved shared-cell velocity and the reference drag
coefficient **bit for bit**. The audit retains its exact-equality guard and
uses that replay's flux. The original failed attempt is not a qualified result.

Normal velocity is recovered from the FVM's stored cut-face flux. Its
face-valued ghost velocity reproduces that normal trace within `4.45e-16`.
The tangential mixed increment is recovered from the face and owner velocities
using the native owner-to-face normal distance. The boundary-history archive
supplies the post-replacement VPM normal velocity and tangential derivative.
Reference targets are conservative normal flux and the full reference's
native tangential derivative.

| Advancing trial | FVM normal error / U∞ | Post-replacement normal error / U∞ | Replacement normal jump / U∞ |
| --- | ---: | ---: | ---: |
| Control | 0.00486557 | 0.00492771 | 0.000714325 |
| Native derivative | 0.00486907 | 0.00493162 | 0.000714240 |
| Native velocity and derivative | 0.00453269 | 0.00457029 | 0.000713909 |

| Advancing trial | FVM mixed derivative error [U∞/D] | Post-replacement derivative error [U∞/D] | Replacement derivative jump [U∞/D] |
| --- | ---: | ---: | ---: |
| Control | 0.0132317 | 0.0121164 | 0.00435080 |
| Native derivative | 0.0118543 | 0.0114457 | 0.00197427 |
| Native velocity and derivative | 0.0118681 | 0.0114667 | 0.00196820 |

The replacement improves the measured derivative error but slightly worsens
the normal error in all three final snapshots. All six endpoint cut fluxes
remain below `1.17e-15 U∞D²` in absolute value. This is an interface-state
difference, not a mass leak, extra physical time advance or corrupt restart.
Its effect on force must be tested through an advancing correction.

The [qualified endpoint result](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-coupled-boundary-stage-audit-medium-laminar-qualified/coupled-boundary-stage-audit-3d.json)
stores both traces and their reference errors. All 33 source/input hashes
were checked; 18 error and jump norms recompute independently with zero
difference. The [live reference replay record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-live-reference-flux-replay-medium-laminar/reference-flux-replay-3d.json)
records the exact shared-field and drag checks and backup hash.

The next [interface-iteration experiment](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_interface_iteration_3d.py)
repeats each FVM interval from a saved accepted starting state. Every sweep
replaces the same advected VPM predictor using the newly computed FVM endpoint.
It must not advance VPM again or start renewal from a previous sweep's corrected
particles. The first map evaluation is replayed and required to match numerical
fields and clocks bitwise before testing additional sweeps. The
[qualified first interval](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/interface-iteration-3d.md)
converges after nine sweeps, reducing whole-domain and near-body FVM velocity
errors by 9.06% and 11.28%, while worsening drag error from −0.966% to −1.481%.
The completed 20-interval comparison gives a different force response: RMS
relative drag error falls from 2.082% to 0.555%, final drag error changes from
−2.096% to +0.756%, and final near-body velocity error falls 2.50%. Eight
intervals reach the sweep cap. The final replacement jump is nearly removed,
but normal and derivative reference errors remain approximately `0.00493 U∞`
and `0.01218 U∞/D`. The requested force/profile agreement remains unachieved.
