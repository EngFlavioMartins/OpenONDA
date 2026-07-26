# Vortex-ring interactions: baseline versus stabilized VPM

This tutorial compares the previous numerical core with a physically
stabilized discretization. Stability here does not mean damping the particle
field: a step is accepted only while the unmodified VPM evolution satisfies
the declared energy and invariant checks.

## Comparison

`--method baseline` uses the fractional position/strength update, Gaussian
kernel, static Smagorinsky model, and Barnes--Hut interactions.

`--method stabilized` uses:

- `h/a0 <= 0.2`;
- `dt <= 20 h^2 / Gamma0`;
- the Winckelmans--Leonard algebraic kernel;
- `C_nu = 256/45` for core spreading and the matching initial anti-diffusion
  shift;
- exact direct interactions;
- coupled RK2 stages for `(x, Gamma)`;
- antisymmetric pairwise conservative stretching;
- Strang-split core spreading;
- automatic subcycling with
  `dt_sub ||S||_2 <= 0.08` and `|u| dt_sub / h <= 0.25`;
- no post-step modification of circulation, core size, or particle topology.

The guard checks that kinetic energy is non-increasing, its decay agrees with
the viscous enstrophy sink, and vector circulation, linear impulse, and
kernel-corrected angular impulse remain within tolerance. It stops at the first
failed step instead of silently changing the solution.

The conserved circulation diagnostic is the vector sum `sum(Gamma_i)`.
`sum(|Gamma_i|)` is total strength variation and is not conserved by
three-dimensional stretching.

## Run and plot

Run both interaction families and both methods:

```bash
./allrun.sh
```

Restrict the matrix when needed:

```bash
METHODS="stabilized" RUN_FAMILIES="leapfrog" ./allrun.sh
```

Regenerate the publication figures from an existing solution root:

```bash
./allplot.sh --solution-dir solution --figures-dir figures
```

No finite-resolution method can guarantee every invariant after the spatial
field becomes unresolved. The stabilized case therefore provides a bounded
contract: accepted steps preserve the stated physics; an inadmissible run
fails and asks for finer spacing or a smaller macro step.
