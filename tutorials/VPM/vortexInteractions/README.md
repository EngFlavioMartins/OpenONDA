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

The strict guard applies only to `les_stabilized`. It checks that kinetic
energy is non-increasing, its decay agrees with the viscous enstrophy sink, and
vector circulation, linear impulse, and kernel-corrected angular impulse remain
within tolerance. Baseline and LES are controls: the same quantities are
recorded and plotted without rejecting the control merely for demonstrating
the error under study.

The conserved circulation diagnostic is the vector sum `sum(Gamma_i)`.
`sum(|Gamma_i|)` is total strength variation and is not conserved by
three-dimensional stretching.

## Run and plot

Run both interaction families and all three methods:

```bash
./allrun.sh
```

Restrict the matrix when needed:

```bash
METHODS="les_stabilized" RUN_FAMILIES="leapfrog" ./allrun.sh
```

`allrun.sh` is transactional. It writes each rerun below `solution/.running`
and replaces an existing case only after the new run records a terminal
status. It does not erase the other cases or figures. To explicitly discard
everything first, use `CLEAN_ALL=1 ./allrun.sh` or `./allclean.sh --all`.

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
