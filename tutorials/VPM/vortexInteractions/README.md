# Vortex-ring interactions: six-case VPM comparison

This tutorial runs leapfrogging and head-on colliding rings with three methods,
for a total of six simulations. Stability does not mean damping the particle
field: the stabilized method is accepted only while the unmodified VPM
evolution satisfies the declared energy and invariant checks.

## Comparison

`--method baseline` uses molecular-viscosity DNS, the fractional
position/strength update, Gaussian kernel, and Barnes--Hut interactions.

`--method les` keeps that numerical core and adds the static Smagorinsky LES
model. This isolates the SGS-model effect from the stabilization methodology.

`--method les_stabilized` keeps the same LES model and uses:

- `dt <= 20 h^2 / Gamma0`;
- a spacing fine enough that the anti-diffused initial core is not aliased
  (the paper's stricter `h/a0 <= 0.2` is still checked, and the shipped
  `h = 0.030` runs with `--allow-underresolved` at `h/a0 = 0.30`);
- the same Gaussian kernel as the two controls, so the comparison does not
  change the represented initial vorticity field;
- `C_nu = 4` for Gaussian core spreading and its matching initial
  anti-diffusion shift;
- exact direct interactions;
- the **pairwise-conservative** stretching exchange (`CONSERVATIVE`);
- **coupled implicit-midpoint** stages for `(x, Gamma)`;
- `C_s = 0.20`, the documented coarse-grid Smagorinsky setting (`les` keeps
  the lower-dissipation `C_s = 0.16` control);
- Strang-split core spreading;
- automatic subcycling with
  `dt_sub ||S||_2 <= 0.08` and `|u| dt_sub / h <= 0.25`;
- no post-step modification of circulation, core size, or particle topology.

### Why those last two are the ones that matter

The invariants are exact by construction, not by tuning:

- The stretching exchange is antisymmetric in `(i, j)`, so `sum(Gamma)` is an
  invariant of the semi-discrete system. Its tangential part additionally
  satisfies `r_ij x dGamma_ij = -(u_ij x Gamma_i + u_ji x Gamma_j)`, which
  cancels the advective half of `dI/dt` exactly — so the **linear impulse**
  `I = 1/2 sum(x x Gamma)` is an invariant too. `TRANSPOSED` gets the first
  identity but not the second, and leaks impulse at first order in `dt`.
- `sum(Gamma)` is *linear* in the state, so any integrator preserves it. `I` is
  *quadratic*, and a Runge--Kutta method preserves a quadratic invariant only if
  `b_i a_ij + b_j a_ji = b_i b_j`. The Gauss methods satisfy it and no explicit
  method does, so the one-stage Gauss method — implicit midpoint — is what
  carries `I` to round-off instead of to `O(dt^p)`.

Measured on the leapfrog case at `t = 3.0 s` (`t Gamma0/R0^2 = 9.4`), the two
changes are independent and multiply — all four rows are the same physics, the
same `h`, and the same `dt`:

| stretching mode | scheme | `abs(dI)/Gamma` | `div(w)` error | survives to |
|---|---|---|---|---|
| `TRANSPOSED` | RK2 | 1.9e-2 | 0.013 | `t > 5.5` |
| `TRANSPOSED` | `MIDPOINT` | 1.9e-2 | 0.013 | `t > 5.5` |
| `CONSERVATIVE` | RK2 | 5.4e-5 | 0.101 | `t ~ 4.0` |
| `CONSERVATIVE` | `MIDPOINT` | **3.9e-7** | 0.099 | `t ~ 4.0` |

`MIDPOINT` buys nothing with `TRANSPOSED`, which is the point: that mode's leak
is `O(dt)` in the *semi-discrete* system, so no integrator can remove it. Only
once the mode supplies the invariant does the Gauss method have something to
preserve.

### The cost, and why the run stops early

The same `Gamma_i x Gamma_j` sign that the impulse identity forces is the one
that excites the divergent part of the discrete vorticity field.
`CONSERVATIVE` grows `||div w||/||grad w||` about eight times faster than
`TRANSPOSED` and goes unstable near `t = 4`. **Refining does not cure it** —
at `h = 0.0318` the divergence error follows the same trajectory (0.094 at
`t = 2.8` versus 0.099 at `t = 3.0` for `h = 0.045`) and the blow-up arrives
*sooner*, so this is an intrinsic property of the scheme rather than an
under-resolution artifact.

Within this kernel family the two properties are therefore mutually exclusive:
the impulse identity fixes the rotational term's sign, and that sign is
destabilizing. The stabilized method takes exact conservation and lets the
contract bound the interval, which is why its curves end before the controls'.
The unexplored escape route is the *radial* term: the identity
`r_ij x dGamma_ij = -(u_ij x Gamma_i + u_ji x Gamma_j)` constrains only the
rotational part, leaving `dGamma_ij = -C Gamma_i x Gamma_j + lambda r_ij` free
in `lambda`. A `lambda` chosen to damp the divergence mode would keep both
invariants and the stability.

If you would rather have the full-duration run than the exact impulse, swap
`StretchingConfig.conservative` for `.transposed` in `build_solver_config`: that
restores `TRANSPOSED`/`MIDPOINT` from the table above — stable throughout, with
`sum(Gamma)` still exact and the impulse drifting at `2e-2`.

The strict guard applies only to `les_stabilized`. It checks that kinetic
energy is non-increasing, its decay agrees with the viscous enstrophy sink, and
vector circulation, linear impulse, and kernel-corrected angular impulse remain
within tolerance. Baseline and LES are controls: the same quantities are
recorded and plotted without rejecting the control merely for demonstrating
the error under study.

A second, independent guard checks **resolution**, because a
structure-preserving scheme conserves its invariants whether or not the particle
field still resolves the flow — conservation alone would happily pass a wrong
answer. The solver records particle overlap `h_nn/sigma`, the vorticity
divergence error `||div w||/||grad w||`, and the angle between `Gamma_p` and
`w(x_p)` with every diagnostics row (`rings_resolution.png`), and the stabilized
method rejects a run that exceeds `--max-overlap-ratio`,
`--max-divergence-error` or `--max-misalignment-deg`.

The conserved circulation diagnostic is the vector sum `sum(Gamma_i)`.
`sum(|Gamma_i|)` is total strength variation and is not conserved by
three-dimensional stretching. In the head-on family its growth is physical line
stretching: the rings expand radially and `sum(|Gamma_i|) / (2 pi R Gamma)`
stays constant to about 1%.

Angular impulse is *cubic* in the state and has no discrete conservation
structure here, so unlike the other two it is a truncation error that converges
under refinement (16x smaller going from `h=0.045` to `h=0.0318`). It therefore
has its own `--angular-drift-tolerance`, looser than the round-off bound
`--invariant-drift-tolerance` that applies to the two exact invariants; holding
a truncation quantity to a conservation tolerance would just report the
particle spacing.

### What the study measured at the previous, coarser spacing

At `h=0.045` (axisymmetric, no Widnall perturbation) both stabilized cases
passed every check until the energy budget closed to worse than 30%, then
stopped and recorded why:

| | `abs(dSumGamma)/Gamma` | `abs(dI)/(Gamma R0^2)` | steps with `dE/dt>0` | admissible to |
|---|---|---|---|---|
| leapfrog, stabilized | 4.6e-7 | **6.1e-6** | 0 / 29 | `t = 2.90` |
| leapfrog, DNS control | 7.5e-7 | 5.8e-2 | 3 / 72 | (not gated) |
| collide, stabilized | 1.1e-6 | **1.0e-6** | 0 / 22 | `t = 2.20` |
| collide, DNS control | 6.8e-6 | 1.8e-1 | 5 / 60 | (not gated) |

## Run and plot

Run both interaction families and all three methods:

```bash
./allrun.sh
```

One armed configuration, no profiles: `h=0.030`, `dt = 20 h^2/Gamma0`, the
`epsilon_W=0.025` 24-mode Widnall perturbation, and `t_end = 12` (leapfrog) /
`10` (collide) so the rings have time to destabilize and break down. That
spacing carries about 33k particles — 11.5x the previous study — and is the
smallest that clears initial aliasing of the anti-diffused core (see the table
at the top of `allrun.sh`) while keeping the `O(N^2)` direct/conservative
stabilized method tractable. Expect several hours for the full matrix.

The contract is enforced for `les_stabilized`, always: past the point where it
fails, the structure-preserving scheme does not produce a less accurate answer,
it produces a conservative wrong one, so the run stops and is kept with a
`rejected_physical_contract` status. The controls are never gated.

Restrict the matrix when needed:

```bash
METHODS="les_stabilized" RUN_FAMILIES="leapfrog" ./allrun.sh
```

Every knob is an environment override — `PARTICLE_SPACING`, `DT`, `LF_STEPS`,
`COLLIDE_STEPS`, `EPSILON_W`, `GUARD_FREQUENCY` — so a cheaper smoke run is:

```bash
PARTICLE_SPACING=0.045 DT=0.010 LF_STEPS=720 COLLIDE_STEPS=600 EPSILON_W=0 ./allrun.sh
```

`allrun.sh` is transactional. It writes each rerun below `solution/.running`
and replaces an existing case only after the new run records a terminal
status. It does not erase the other cases or figures. To explicitly discard
everything first, use `CLEAN_ALL=1 ./allrun.sh` or `./allclean.sh --all`.
All three terminal statuses are promoted. `rejected_physical_contract` is the
stabilized method's own result, and `terminated_nonphysical` is the controls'
-- they are *expected* to blow up once the rings break down, which is the error
under study. Keeping an older run instead would leave one case at a different
spacing from the rest of the matrix, and a figure built from mixed resolutions
looks fine while being wrong; a missing case is caught loudly by
`assets/validate_plot_inputs.py`.

Regenerate the publication figures from an existing solution root:

```bash
./allplot.sh --solution-dir solution --figures-dir figures
```

Plotting validates the complete six-case matrix before overwriting figures.
For an intentional subset, add `--allow-partial`.

No finite-resolution method can guarantee every invariant after the spatial
field becomes unresolved. The stabilized LES case therefore provides a
bounded contract: accepted steps preserve the stated physics; an inadmissible
run is retained with a rejected status and asks for finer spacing or a smaller
macro step.
