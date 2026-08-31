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

The certification gates require at least four unique seeds, no identical
nonzero-time trajectories, at least 99.5% projected absolute circulation, and
relative field standard error below 7.5%. Eight members are the default pilot.
If a physics case fails that precision gate, add independent seeds only to
that case and recompute its fixed-time estimator; case-specific sample sizes
are valid because no cross-case paired statistic is reported.  Never relax a
predeclared gate after inspecting the result.  For publication-quality use,
also extend the ensemble when the confidence interval of the scientific
conclusion is still material; do not stop solely because a fixed member count
has been reached.

For this benchmark, the eight-member pilot passed for the single vortex and
merger but the dipole did not.  The reproducible production defaults are
therefore 8/12/8 members for vortex/dipole/merger. These sample sizes are
written explicitly in `allrun.sh`; deliberate convergence studies should
change those three arguments explicitly.

## Energy-rate convention

The finite-difference \(dE/dt\) is reported only while consecutive samples use
the direct, unbounded kinetic-energy integral. Above the direct-evaluation
particle limit, the solver uses a finite Fourier box for instantaneous audits.
That box follows the particle support, so consecutive energies do not share
one integration domain and their difference is not a defined time derivative.
Such rates are stored as unavailable rather than smoothed.

The enstrophy-based viscous power \(-2\nu Z\), with
\(Z=\tfrac12\int |\boldsymbol{\omega}|^2\,dV\), remains well defined and is
retained over the complete history. This distinction is especially important
for DVH and GBD, whose regenerated grids can exceed the direct-integral limit.

## Reproducibility and files

Raw, independently seeded member backups and flow integrals are retained
under `solution/rwm_ensemble/` and `samples/rwm_ensemble/`. Canonical ensemble
means are written under `samples/{vortex,dipole,merging}_rwm/`, alongside:

- `run_metadata.json`: seeds, estimator, confidence level, and definitions;
- `rwm_convergence.csv`: time-resolved field uncertainty and projection QA;
- `field_diagnostics.csv`: mean-flow features and jackknife intervals;
- `flow_integrals.csv`: ensemble means and Student-t intervals;
- VTS files: mean velocity/vorticity and their pointwise standard errors.

Seeded RWM ensembles must use CPU, CUDA, or Vulkan with the current Taichi
runtime. Metal is rejected because that backend does not accept the requested
random seed, so independent reproducible members cannot be certified.

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
