# Cube transfer: a confirmed failure to preserve matching velocity

The spatial vorticity blend can manufacture velocity when FVM and VPM already
describe the same velocity. The defect is reproducible in three dimensions
without a body, boundary iteration, recirculation, time integration, viscosity,
or pruning. It is also measurable on the actual saved cube state.

This establishes a defect and its location. It does not attribute the entire
time-dependent reference discrepancy to this operation. The particle evolution
has a separate representation sensitivity, and finite-core dynamics and the
viscous closures retain their own accuracy limits.

The shared source baseline is `fda69c0f`. Saved cube states are those from the
completed centred-lattice trial, interpreted by their recorded source revision.
Original simulations and reference results are read-only inputs. No mesh,
FVM scheme, time step, or health threshold is changed in these studies.

## Mechanism

Write the reconstructed Gaussian vorticity as `w`, the induced velocity curl
as `q=curl(u)`, and their longitudinal difference as `ell=w-q=grad(psi)` in a
fixed free-space Helmholtz representation. Biot–Savart removes `ell`.

If the FVM donor describes the **same velocity**, it supplies `q`, whereas the
current represented-state blend targets

```text
w_new = (1-eta) w + eta q = w - eta ell.
curl(w_new-w) = -grad(eta) cross ell.
```

A constant `eta` preserves the invisibility of a gradient. A spatially varying
`eta` generally does not. The current coefficient blend and one-step Gaussian
correction introduce an additional finite-representation change.

The independent [derivation](longitudinal-transport-analysis-2026-09-15.md)
separates this transfer identity from the particle evolution and finite-core
transport terms. In particular, a nodal `Gamma/V` substitution is not an exact
Helmholtz correction.

## Decisive manufactured control

Both solvers describe zero velocity. The VPM density includes the gradient of
a compactly truncated 3D Gaussian. Its core radius remains `0.066 D`; only the
independent integration quadrature is refined. The test calls the current
production blend with the cube's authority ramp and amplification setting.

| Quadrature spacing / D | Initial velocity RMS / U | After cube authority blend | After constant-authority control |
| --- | ---: | ---: | ---: |
| 0.06 | 7.1445e-8 | 0.00637562 | 3.5722e-8 |
| 0.03 | 4.4390e-15 | 0.00637627 | 2.2195e-15 |

The false velocity persists as the input and constant-weight control approach
roundoff. [Driver](cube_transfer_nullspace.py),
[measurements](results/cube-wake-cause-2026-09-15/transfer-nullspace/probe.json).

## Actual cube control

At `t*=8`, an oracle donor supplies the curl of the saved VPM velocity on the
transfer lattice, including preserved outer particles. Its Jacobian is checked
against an independent direct sum at 48 targets (absolute curl RMS difference
`0.0004751 U/D`). There is no FVM advancement or reconstruction in this oracle.

The artificial `w` donor below is a diagnostic control, not physical FVM
vorticity. The coefficient-preserving control retains the same body masks,
remapping, pruning and moment restoration.

| Transfer control | Velocity change RMS in authority region / U | Change in downstream region / U |
| --- | ---: | ---: |
| Current blend, physically matching `q` donor | 0.0141577 | 0.00222217 |
| Current blend, artificially matching `w` donor | 0.0134248 | 0.00145415 |
| Preserve coefficients through the same cleanup | 0.000147059 | 0.000147098 |

The downstream probe region is the same 196-point 3D set with
`1.25 <= x/D <= 1.62` used by the previous operator audit. Its reflection
asymmetry changes from `0.117229` to `0.117847` with the physically matching
donor, versus `0.117214` with the coefficient-preserving control.
[Driver](cube_velocity_matched_transfer.py),
[measurements](results/cube-wake-cause-2026-09-15/velocity-matched-spacing/audit.json).

Capturing an ordinary renewal with the actual FVM donor locates additional
divergence in the coefficient blend and its representation correction:
downstream divergence RMS changes `9.41736 -> 9.45296 -> 9.49430 U/D^2`.
Remapping changes that measurement only at roundoff, and pruning slightly
reduces it to `9.49302`. These are increments of one frozen operation, not
an additive budget of the full trajectory error.
[Capture driver](cube_wake_transfer_stage_probe.py),
[stage measurements](results/cube-wake-cause-2026-09-15/transfer-stages/stages.json).

## Candidate and isolated replay

The candidate changes the transferred quantity to an incremental curl:

```text
delta_w = curl[eta m (u_F-u_V)].
```

Authority and body/support weights belong inside this curl. Matched velocity
samples then produce zero correction. The prototype uses compatible centered
lattice derivatives and applies the correction after ordinary cleanup;
post-curl clipping, pruning or restoration to the old impulse would change
the tested operator. Continuous Gaussian divergence and finite-core filtering
remain separate convergence checks.

The [pure prototype](cube_curl_residual_transfer.py) passes its manufactured
matching-velocity and derivative-convergence checks. The
[bounded replay driver](cube_curl_transfer_replay.py) then restarts the saved
`t*=6` VPM state with prescribed fine-reference donors, so FVM boundary feedback
is absent. Native and candidate controls have the same checkpoint hash and
identical initial probe velocities. Both use linear interpolation between the
native reference states at 6 and 7; intermediate-time errors are therefore
relative to this interpolated reference, not independently saved solutions.

| Time | Native downstream velocity RMS error / U | Curl candidate | Native outer-wake error / U | Curl candidate | Native particles | Candidate particles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6.0 | 0.0820169 | 0.0820169 | 0.0793250 | 0.0793250 | 43,479 | 43,479 |
| 6.1 | 0.0913384 | 0.0911516 | 0.0756013 | 0.0756551 | 44,214 | 101,688 |
| 6.2 | 0.105909 | 0.104525 | 0.0811249 | 0.0815563 | 45,064 | 102,415 |

The full-support candidate is **rejected for production**. Its small early
improvement downstream does not offset a slightly worse outer-wake error and
more than twice the particle count. Keeping every nonzero curl increment adds
many weak particles that subsequent GBD cleanup removes and the next correction
reintroduces. At 20 steps its elapsed time was 271.7 s, versus 100.4 s in the
previous native control. These timings include diagnostics and were collected
under different shared-machine loads; they are evidence of an expensive trial,
not a controlled production speed benchmark.

The trial was deliberately stopped with SIGINT after 22 accepted steps, during
step 23. Original health gates passed those 22 accepted steps. There is no
completed 100-step result or final candidate checkpoint. Its small short-term
effect on an already perturbed state does **not** rule out the transfer defect
as an earlier seed, or establish the result of a corrected clean-start run.
[Candidate measurements](results/cube-wake-cause-2026-09-15/curl-replay/replay.json),
[native control](results/cube-wake-drift-2026-09-15/reference-driven-controls/native/replay.json).

## Actual particle evolution: instantaneous cause budget

The [represented evolution budget](cube_represented_evolution_budget.py) uses
all 43,479 particles and the actual saved stage velocities and strength rates
at `t*=6`. It differentiates the moving Gaussian sum, including center motion.
No new trajectory or change to the mesh, particle cores, time step, or viscosity
is needed. The diagnostic isolates the inviscid stage; the previous native
operator audit separately measures diffusion.

For the cloud's self-induced velocity plus freestream, the decomposition is

```text
u_dot_native = P(u cross q) + 2 K(J.T ell) + K R_core + native_minus_self.
```

Here `K` is Biot–Savart, `P=K curl`, `J_ij=du_i/dx_j`, and `R_core` is computed
independently as the exact represented particle derivative minus the resolved
positive-transpose transport operator. It includes finite-core and quadrature
effects, rather than identifying those effects with longitudinal leakage.
The final term explicitly retains the actual native body/induction contribution
and the discrepancy between periodic self induction and free-space induction.
The body potential is not silently extended across the solid and inserted into
an invalid all-space identity.

For the larger periodic box (`9.6 x 4.8 x 4.8 D`), the signed contributions to
the downstream reflection-asymmetry energy derivative are:

| Contribution | Energy derivative, in `U^3/D` |
| --- | ---: |
| Actual native stage, independent free-space direct evaluation | +0.00986705 |
| Resolved Euler term | +0.00414282 |
| Longitudinal leakage | +0.00626473 |
| Finite-core/transport remainder | -0.000622560 |
| Native minus periodic self contribution | +0.0000813977 |

Thus the longitudinal term supplies about **63% of the instantaneous downstream
growth** in this saved state; finite-core transport slightly opposes it. This is
not a percentage of accumulated reference error. Nor is the remainder an
independent transverse-only effect: it depends on the full particle field.
Farther downstream, longitudinal growth and the finite-core contribution nearly
cancel, and propagation of the existing velocity disturbance dominates.

This conclusion is checked against both quadrature and domain changes. Refining
quadrature from `0.03 D` to `0.02 D` changes the downstream longitudinal energy
contribution by `3.2e-8 U^3/D`; the independent product-rule closure reaches
`6.7e-9 U^2/D` RMS over all probes. Enlarging the base periodic box from
`7.2 x 3.6 x 3.6 D` to the box above changes the downstream longitudinal and
finite-core contributions by approximately 0.6% and 0.8%. The actual periodic
velocity-rate asymmetry derivative differs from the independent free-space
measurement by `6.3e-7 U^3/D` on the larger box.
A final enlargement to `10.8 x 5.4 x 5.4 D` changes the longitudinal and
finite-core contributions by another 0.093% and 0.12%, respectively. The
longitudinal contribution is then `0.00627055 U^3/D`, or 63.55% of native growth.

Periodic self velocity itself still differs appreciably from free space
(`0.028 U` RMS on the larger box). The explicit native-minus-self term and
domain checks matter; an accurate local energy budget does not license claiming
that the periodic velocity approximates the complete free-space field equally
well. Even the last box retains a `0.0202 U` velocity difference.
[Raw budgets and validation](results/cube-wake-cause-2026-09-15/represented-evolution-budget/budget.json).

The production coupling method remains unchanged pending an affordable
correction tested against this mechanism. The transfer fixed-point defect and
the actual longitudinal amplification now have separate, reproducible controls.

## Covector-rate intervention: benefit and failed support qualification

The next intervention changes only the RK strength rate on the existing
particles:

```text
Gamma_dot_p = J_p.T Gamma_p - 2 V_p J_p.T (omega_G(X_p) - curl(u)(X_p)).
```

Every stage reconstructs its own Gaussian density and uses its complete native
Jacobian, including the body contribution. This nodal source is a filtered,
finite-support approximation to the resolved correction above; it is not an
exact Helmholtz method. Original transfer, GBD, core radius, spacing and health
limits are retained. No alignment or new moment repair is added.

Ten steps reduce the added downstream asymmetry energy by 71.3% and improve
velocity error in both measured regions. A fresh native ten-step control
reproduces the previous native velocities to approximately `5e-9 U` in the
reported RMS error. The subsequent 100-step correction trial completes all
original health gates and reaches the actual saved reference endpoint at 7:

| Measurement at `t*=7` | Native control | Nodal correction |
| --- | ---: | ---: |
| Downstream velocity RMS error / U | 0.129684 | 0.114856 |
| Outer-wake velocity RMS error / U | 0.102587 | 0.0865063 |
| Downstream reflection asymmetry / U | 0.0728906 | 0.0499573 |
| Outer-wake reflection asymmetry / U | 0.0576964 | 0.0403325 |
| Particle count | 51,313 | 51,300 |

The error improvements are 11.4% and 15.7%. This is useful causal evidence, but
does not establish a complete coupled solution, force accuracy or long-time
agreement. The correction trial takes 808.1 s including its diagnostics,
versus 496.7 s in the historical native control. Different shared-machine loads
prevent a strict speed benchmark. The measured source implementation itself
uses 360.5 s; it is an expensive host diagnostic, not a production implementation.

Two cost corrections preserve the ten-step outcome within `6.4e-9 U` in its
reported downstream RMS error. Scalar inner-loop accumulators remove temporary
arrays from each Gaussian neighbour interaction, giving identical density values
in the measured cloud. Skipping unused strength-rate work during accepted-field
refreshes halves source calls: the longer trial records exactly 200 calls for
100 Heun steps, and its first recorded stages are 0 and 1 with distinct temporary
states. [Density check](results/cube-wake-cause-2026-09-15/density-allocation-check.json).

The [100-step ledger](results/cube-wake-cause-2026-09-15/covector-control-ledger/replay.json)
records exact particle invariants around RK, diffusion and transfer, plus
tableau-weighted source rates. The maximum norm of the RK net-strength closure
is `1.18e-8 m^3/s`. The integrated added source is
`(0.00220819, 0.00577868, 0.00127855) m^3/s`; its integrated impulse source is
`(0.00223835, -0.0108015, 0.00611430) m^4/s`. A closed bookkeeping identity does
not establish the physical correctness of those added sources.

The [independent support check](cube_covector_support_budget.py) identifies a
material limitation. For the actual cloud's resolved periodic self flow,
integrating `C=-2 J.T ell` over the whole periodic box gives zero to roundoff.
After excluding the cube, the larger-box fluid integral is approximately
`(-0.001962, 0.0000395, -0.0000501) m^3/s^2`. Sampling the same source only at
occupied particle nodes instead gives
`(0.021190, -0.010782, -0.004401) m^3/s^2`. Refining quadrature from 0.03 to
0.02 barely changes the discrepancy. The cube's excluded volume is kept exactly
one cubic metre through volume fractions; this is not a changing-geometry
artifact. [Support measurements](results/cube-wake-cause-2026-09-15/covector-support-budget/support.json).

This check isolates self-flow support error; it does not silently substitute a
periodic body-free field for the complete native body-bounded problem. It is
already enough to reject the claim that the nodal source preserves the intended
moment balance merely because it reduces velocity error. The
[acceptance review](covector-acceptance-review-2026-09-15.md) derives the relevant
body-boundary and impulse terms. A global moment repair would conceal this
discretization issue. The candidate therefore remains **unqualified for
production**, despite its measured velocity benefit.

## Conservative frozen operator and stopping point

The [compact flux control](cube_covector_flux_control.py) fixes the net-source
defect by localizing the potential before differentiation:

```text
phi = chi psi,
C_chi = -2 div(phi J.T)
      = chi C - 2 psi J.T grad(chi),  for div(u)=0.
Gamma_dot_i = -2 sum_faces A_face phi_face (J_face.T n_i).
```

Each internal face has one shared flux, applied with opposite signs to its
two cells. The taper goes to zero at the occupied-support boundary, including
holes and the cube. Its gradient contribution is part of this selected compact
gradient correction. This does not transport the entire longitudinal field:
`grad[(1-chi)psi]` remains outside the correction. A zero flux imposed on an
untapered potential would define a different, sharp surface source.

The free-space scalar potential is computed with the same Gaussian induction
factor as velocity, replacing the cross product by a dot product. Linear FFT
convolution evaluates the finite saved lattice without periodic images. An
independent direct sum at 48 actual particles agrees to `1.43e-8 m/s` RMS.
The potential gauge is fixed by its decay at infinity, since adding a constant
before localization would change the source.

The [three-grid analytic qualification](qualify_covector_flux.py) uses an affine
three-dimensional incompressible strain/rotation field and a compact polynomial
potential with a known nonzero impulse. Net-source norms stay below `4e-17`;
the finest impulse error is `4.06e-9 m^4/s^2`. The measured convergence orders
are 1.972 for the source density and 1.981 for its Gaussian-induced velocity
rate, keeping the Gaussian core fixed. The independent reference integration
is substantially finer than the tested operator.
[Analytic measurements](results/cube-wake-cause-2026-09-15/covector-flux-qualification/qualification.json).

On the actual frozen cube lattice, with a fixed `0.18 m` taper and all 43,479
existing particles, the source has net rate approximately `1.6e-16 m^3/s^2`.
It reduces the instantaneous downstream asymmetry-energy derivative from
`0.00986705` to `0.00862090 U^3/D` (12.6%), and the outer value from
`0.00122671` to `0.000805310` (34.4%). These are **frozen rates**, not a second
completed trajectory or a percentage reduction in total reference error.
Its discrete impulse source is close to the independent midpoint estimate of
`-integral(phi curl(u))`, with a vector difference about `3.1e-5 m^4/s^2`.
[Actual-state measurements](results/cube-wake-cause-2026-09-15/covector-flux-checked/control.json).

The support geometry limits this correction: only 8.0% of downstream particle
strength lies beyond the full `0.18 m` taper distance; no downstream particle
has a full `0.24 m` interior plateau. The needed smooth localization therefore
cannot cheaply act on all the problematic content in this active cloud.
[Geometry measurements](results/cube-wake-cause-2026-09-15/active-support-geometry.json).

This is the end of the bounded investigation, not production acceptance.
The static operator is conservative and independently consistent on the
manufactured grids, but the moving RK cloud does not provide its Cartesian
cell geometry. An integration on a well-defined remeshed state would need its
own time-splitting and coupled-flow qualification. Actual body/support treatment,
the transfer fixed-point defect, and force accuracy also remain to be closed.
No further long replay or production method change is made here. The numerical
mechanism, intervention benefit, failed shortcut, conservative operator, and
remaining implementation requirements are now explicit and reproducible.
