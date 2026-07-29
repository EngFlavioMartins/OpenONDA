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

`--method les_stabilized` is numerically **identical** to `les` -- same
fractional RK3 core, Gaussian kernel, treecode and core spreading -- plus two
things: the coarse-grid Smagorinsky constant `C_s = 0.20`, and the enstrophy
envelope. The comparison therefore isolates the envelope instead of confounding
it with a pile of numerical differences.

### What the envelope is

A control-barrier safety layer on a *vorticity-sensitive* norm. Conserving
circulation and impulse bounds nothing about `|Gamma_p|/sigma^3` or
`||grad u||`, so a scheme can hold its invariants to round-off and still blow up
-- which is what the previous pairwise-conservative stabilization did here
(it crashed at `t = 0.14`, where the ungoverned controls reach `t = 5.87`).

It does **not** cap enstrophy magnitude. Vortex stretching legitimately produces
enstrophy: the head-on collision grows it `1826 -> 2890` while
`sum|Gamma|/(2 pi R Gamma)` stays constant to 1%, i.e. pure line stretching. A
fixed cap near `Z(0)` would suppress the mechanism the solver exists to resolve.

It caps *anomalous scale-localized* production instead. With `Z_D` the
particle-scale enstrophy and `Z_2D` the same quadratic form at a widened kernel
width, physical coherent stretching raises both, while particle-scale pile-up
raises mainly `Z_D`. The admissible set is

    Z_D <= rho_max * B_L + Z_floor,    Bdot_L <= a_L + b_L B_L,   B_L >= Z_2D

so the cap moves with credible coarse-scale growth. Gronwall bounds `B_L` at
every finite time, hence `Z_D`; with `sigma_p >= sigma_min` and finite total
volume, Cauchy-Schwarz then bounds `sum|Gamma|`, `u` and `grad u`, so the
particle ODE cannot blow up. **The guarantee is for this modified discrete
model, not for Navier-Stokes.**

The correction is the minimum-norm strength change restoring the bound, subject
to nine hard equality rows -- total vorticity, linear impulse, angular impulse
(with the kernel second moment) -- plus a no-energy-injection inequality. A plain
`-lambda Gamma_p` sink would also restore the bound, but it shrinks the impulse;
the constrained solve is what keeps the conservation properties. Measured on a
deliberately violating state: bound restored exactly, invariants held to `1e-16`.

Two hard ceilings sit above all of that (`--omega-hard`, max `h_nn/sigma`) and
are deliberately **fatal**: past them the requested state is outside what the
discretization can represent, and dissipating hard enough to survive would
produce a bounded wrong answer that looks like a result.

### Calibrate before trusting it

`rho_max` and `b_L` are measurements, not defaults. Run the ungoverned controls,
then:

```bash
python assets/calibrate_envelope.py --solution-dir solution
```

It reads `rho_Z` and the coarse-scale growth rate from `baseline` and `les`,
truncates each at the point where the resolution diagnostics go bad (so the fit
never legalises the blow-up), and takes the 99.9th percentile plus the spread
between resolutions. The shipped `--rho-max 2.0` is a placeholder.

### Reading the result

`envelope_active_fraction` and `envelope_chi` (safety dissipation over the
physical plus SGS sink) are in `flow_integrals.csv`. **A run that survives only
because the envelope is active most of the time is a bounded wrong answer, not a
success** -- it means the resolution or the primary closure is inadequate, and
the diagnostics are there to say so.

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

Every case runs to the end unless the solution actually falls apart, and then it
says so plainly and saves the state at the crash. Conservation and resolution are
recorded every logging step for the figures; they are diagnostics, never gates.

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
Both terminal statuses are promoted, `completed` and `crashed`. A control that
blows up once the rings break down *is* the result -- that is the error under
study, and `rings_stability.png` exists to show where each variant dies. Keeping
an older run instead would leave one case at a different spacing from the rest of
the matrix, and a figure built from mixed resolutions looks fine while being
wrong; a missing case is caught loudly by `assets/validate_plot_inputs.py`.

Regenerate the publication figures from an existing solution root:

```bash
./allplot.sh --solution-dir solution --figures-dir figures
```

Plotting validates the complete six-case matrix before overwriting figures.
For an intentional subset, add `--allow-partial`.

No finite-resolution method can guarantee every invariant once the spatial field
becomes unresolved. What the envelope guarantees is narrower and precise: the
discrete state cannot blow up. Whether the bounded answer is also *right* is a
separate question, and `envelope_chi` is the number that answers it.
