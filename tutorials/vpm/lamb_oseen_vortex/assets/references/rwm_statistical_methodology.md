# Statistical treatment of Random Walk Method output

## Scope

The Random Walk Method (RWM) advances every particle with an independent
Brownian increment

\[
\Delta \boldsymbol{x}_p
= \sqrt{2\nu\Delta t}\,\boldsymbol{\eta}_p,
\qquad \boldsymbol{\eta}_p\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I}).
\]

Consequently, one computed particle state and one sampled Eulerian plane are
Monte Carlo realizations. They are not definitions of the mean flow. The
statistical error of classical random-vortex calculations decreases only as a
square root of particle/realization count; increasing temporal output cadence
does not create independent realizations. See Chorin (1973), Milinazzo &
Saffman (1977), Roberts (1985), and Goodman (1987).

This benchmark estimates the deterministic Navier-Stokes solution represented
by the expectation of the RWM particle measure. It does not interpret the
numerical Brownian scatter as physical turbulence.

## Primary field estimand

The benchmark is two-dimensional and statistically homogeneous along its
finite vortex column. For ensemble member \(m\), the primary cross-sectional
vorticity estimator is therefore the column projection

\[
\widehat{\omega}^{(m)}_z(x,y,t)
= \frac{1}{L}\int \omega^{(m)}_z(x,y,z,t)\,\mathrm{d}z.
\]

The integral is evaluated directly from the Gaussian particle backups.
The corresponding in-plane velocity is recovered with the free-space 2-D
Biot-Savart operator. This uses all particles in the homogeneous direction and
does not privilege one noisy \(z\)-plane.

At each fixed physical time, independent seeded realizations are averaged
component by component:

\[
\overline{\boldsymbol{u}} = M^{-1}\sum_m\widehat{\boldsymbol{u}}^{(m)},
\qquad
\overline{\omega}_z = M^{-1}\sum_m\widehat{\omega}^{(m)}_z.
\]

Velocity magnitude, vorticity magnitude, gradients, peaks, and vortex
features are computed after this signed/vector average. In particular,
\(\lVert\mathbb{E}[\boldsymbol{u}]\rVert\) is the reported mean-flow speed;
\(\mathbb{E}[\lVert\boldsymbol{u}\rVert]\) is a different, positively biased
quantity and is not substituted for it.

Temporal smoothing or averaging is not used. The vortex pair evolves and
merges, so a moving time window would change the fixed-time estimand, attenuate
peaks, and smear the merger time.

## Flow-feature definitions

The primary quantities follow Cerretelli & Williamson (2003), whose reference
curves are used by this tutorial.

- **Vortex centre** \(\boldsymbol{x}_{c,i}\): geometric centre of the connected
  area enclosed by the 80%-of-local-peak signed-vorticity contour. This avoids
  assigning the centre to one noisy grid maximum.
- **Separation** \(b\): before peak coalescence,
  \(b=\lVert\boldsymbol{x}_{c,1}-\boldsymbol{x}_{c,2}\rVert\). Once only one
  vorticity maximum remains, \(b=0\), exactly matching the experimental
  definition; it is not treated as missing data.
- **Structure orientation** \(\theta\): before merger, the undirected line
  joining the two centres; after merger, the major axis of the positive-
  vorticity quadrupole on the connected 5%-of-peak support. Both axes have
  period \(\pi\), so unwrapping is performed on \(2\theta\) and divided by two.
- **Velocity core radius** \(a_c\): radius at which the azimuthally averaged
  tangential velocity is maximal. Before merger, the average is taken on the
  outward semicircle of each vortex, excluding the region directly between
  the vortices, and the reported pair value is the mean of both radii. After
  merger it is the full-circle radius of the single structure.
- **Resolved pair**: two distinct 80%-contour regions exist, the centres are at
  least two output-grid spacings apart, and the smaller peak exceeds the
  intervening saddle by more than the 95% ensemble uncertainty of that
  contrast. Once this condition fails, pair loss is absorbing: later numerical
  peaks cannot resurrect two centres. The post-merger values are instead the
  well-defined \(b=0\), ellipse orientation, and full-circle core radius.

The merger comparison uses \(\nu t/a_{c,0}^2\). The primary source uses
\(\tau=\nu t/b_0^2\), so its traceable curves are transformed by
\(\tau/(a_{c,0}/b_0)^2\). With the documented experiment and simulation value
\(a_{c,0}/b_0=0.125\), the literature endpoint \(\tau\simeq0.0478\) becomes
\(\nu t/a_{c,0}^2\simeq3.06\). Simulation samples are retained through the
first output at or beyond 3.0.

The setup also stores the Gaussian 1/e vorticity radius in metadata. It is not
called \(a_c\) in the comparison figures: for a Lamb-Oseen vortex the
velocity-peak radius is about 1.1209 times the Gaussian radius. Keeping these
two radii distinct prevents a systematic definition error.

## Uncertainty and convergence

At every point and fixed time, the sample standard deviation across the \(M\)
independent seeds gives the standard error of the mean. Two-sided 95% Student-t
intervals are stored because ensemble sizes are finite.

Centres, peaks, radii, separation, and orientation are nonlinear functionals of
the ensemble-mean field. Their uncertainty is therefore recomputed with a
delete-one-member jackknife of the complete feature-extraction pipeline; it is
not inferred from the scatter of noisy single-member peak locations.

The following are separately reported:

1. Monte Carlo uncertainty of the ensemble mean;
2. comparison error against the exact Lamb-Oseen solution for the isolated
   vortex, which also contains particle, kernel, time-step, finite-column, and
   field-grid bias;
3. circulation captured by the projection grid;
4. difference between the first half of the ensemble and the full ensemble.

The result checks require at least four unique seeds, no identical
nonzero-time trajectories, at least 99.5% projected absolute circulation, and
relative field standard error at most 7.5%. Ten members form the initial pilot.
If a case fails that precision gate, the launcher adds an independent batch
estimated from the observed inverse-square-root scaling, with a 10% sample-count
margin, and recomputes every fixed-time estimator. It retains all seeds and
never selects trajectories based on their outcome. Case-specific sample sizes
are recorded in metadata; no cross-case paired statistic is reported.

The default cap is 80 members. Failure at the cap is explicit and requires a
larger `--maximum-realizations`; the 7.5% gate is unchanged. The Student-t and
jackknife intervals are nominal fixed-ensemble intervals, not confidence
sequences with an optional-stopping guarantee. For confirmatory inference,
use the pilot to predeclare a fixed production size or validate conclusions
on an independent fixed ensemble.

## Energy-rate convention

The finite-difference `dE/dt` compares consecutive unbounded kinetic-energy
measurements. Small clouds use direct Gaussian pair integrals. Uniform-core,
uniform-viscosity large clouds use zero-padded linear correlations with the
unbounded transverse Gaussian Green tensor. Padding prevents pair wraparound;
resizing the FFT box does not remove the open column's far-field energy.

Viscous power uses the matching transverse (divergence-free) projection of the
Gaussian vorticity. Using the full vorticity norm instead would include a
longitudinal component that induces no velocity in a finite open column.

DVH accumulates 29 accepted steps before applying a resolved heat transfer.
Its energy output interval is rounded up to cover at least one transfer, so
an advection-only plateau is not tested as a viscous energy interval. The
field-output cadence is unchanged. Raw finite differences remain signed; no
positive rate is clipped, smoothed, or replaced by a prescribed negative rate.

## Reproducibility and files

Raw, independently seeded member backups and flow integrals are retained as
distinct flat cases `{physics}_rwm_<nnn>` under `solution/` and `samples/`.
Ensemble means are written under `samples/{vortex,dipole,merging}_rwm/`,
alongside:

- `run_metadata.json`: seeds, estimator, confidence level, and definitions;
- `rwm_convergence.csv`: time-resolved field uncertainty and projection QA;
- `field_diagnostics.csv`: mean-flow features and jackknife intervals;
- `flow_integrals.csv`: ensemble means and Student-t intervals;
- VTS files: mean velocity/vorticity and their pointwise standard errors.

RWM uses a deterministic counter-based generator keyed by seed, accepted step,
and particle index. It supports Metal as well as CPU, CUDA, and Vulkan; the
nonzero-time distinctness check remains required on every backend.

## Primary sources

- A. J. Chorin, “Numerical study of slightly viscous flow,” *Journal of Fluid
  Mechanics* 57 (1973), 785-796.
  <https://doi.org/10.1017/S0022112073002016>
- F. Milinazzo and P. G. Saffman, “The calculation of large Reynolds number
  two-dimensional flow using discrete vortices with random walk,” *Journal of
  Computational Physics* 23 (1977), 380-392.
  <https://doi.org/10.1016/0021-9991(77)90069-9>
- S. Roberts, “Accuracy of the random vortex method for a problem with
  non-smooth initial conditions,” *Journal of Computational Physics* 58
  (1985), 29-43. <https://doi.org/10.1016/0021-9991(85)90154-8>
- J. Goodman, “Convergence of the random vortex method,” *Communications on
  Pure and Applied Mathematics* 40 (1987), 189-220.
  <https://doi.org/10.1002/cpa.3160400204>
- C. Cerretelli and C. H. K. Williamson, “The physical mechanism for vortex
  merging,” *Journal of Fluid Mechanics* 475 (2003), 41-77.
  <https://doi.org/10.1017/S0022112002002847>
