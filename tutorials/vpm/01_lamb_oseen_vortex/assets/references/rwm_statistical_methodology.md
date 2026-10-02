# Random-walk mean flow and uncertainty

The [Lamb–Oseen tutorial](../../README.md) compares random-walk diffusion with deterministic methods. Each seeded RWM run adds Brownian displacement

$$
\Delta\boldsymbol{x}_p=\sqrt{2\nu\Delta t}\,\boldsymbol{\eta}_p,
\qquad \boldsymbol{\eta}_p\sim\mathcal N(0,I).
$$

One run is a Monte Carlo realization. Its scatter is numerical sampling noise; changing output cadence does not add independent realizations.

## Mean field

At each physical time, the finite column of length $L$ is projected onto a common plane:

$$
\widehat\omega_z^{(m)}(x,y,t)=\frac1L\int\omega_z^{(m)}(x,y,z,t)\,dz.
$$

Projection uses the Gaussian particle backups; the in-plane velocity follows from free-space two-dimensional Biot–Savart induction. Average signed vorticity and vector velocity over independent seeds before calculating speed, peaks or flow features. Thus the plotted speed is $|\overline{\boldsymbol u}|$, rather than the mean of individual speeds. No temporal smoothing is used.

## Vortex features

- **Centre:** geometric centre inside the local 80%-of-peak signed-vorticity contour.
- **Separation $b$:** distance between the two centres; zero after their peaks coalesce.
- **Orientation $\theta$:** undirected centre-to-centre axis before merger; major axis of the positive-vorticity quadrupole on 5%-of-peak support afterward. Unwrap $2\theta$ and divide by two.
- **Velocity-core radius $a_c$:** radius of peak azimuthally averaged tangential velocity. Before merger use each vortex's outward semicircle; afterward use the merged structure's full circle.

Two cores are resolved only while their 80% contour regions are distinct, their centres are at least two sampling-grid spacings apart, and the weaker peak exceeds the saddle by more than the contrast's 95% ensemble uncertainty. Once lost, pair identity is not restored by later noisy peaks.

Times use $\nu t/a_{c,0}^2$. The [experimental data](README.md) use $\tau=\nu t/b_0^2$, converted with the actual initialized velocity-peak radius. The Gaussian vorticity radius differs: $a_{c,0}\approx1.12a_0$.

## Uncertainty and stopping

Field and linear-integral intervals are two-sided 95% Student-$t$ intervals across independent seeds. Nonlinear centres, radii, separation and orientation use a delete-one-seed jackknife of the complete mean-field/feature extraction. Shading measures finite-ensemble uncertainty, not physical fluctuations or discretization error.

The campaign starts with ten seeds and adds batches until velocity and vorticity relative standard errors are at most 7.5%, with a default cap of 80. Checks require unique seeds, distinct nonzero-time trajectories and at least 99.5% projected absolute circulation. Results also compare the first half of the ensemble with the full ensemble and report isolated-vortex analytical errors.

The intervals are nominal fixed-ensemble intervals. Adaptive stopping does not provide an optional-stopping confidence guarantee; confirmatory conclusions require a predeclared fixed ensemble or independent validation. Bias from spacing, time step, finite-column geometry and sampling must be assessed separately.

## Output

Seeded cases occupy `solution/<physics>_rwm_<nnn>/` and matching `samples/` directories. Mean-field outputs are in `samples/<physics>_rwm/`: `rwm_convergence.csv`, `field_diagnostics.csv`, `flow_integrals.csv` and VTS fields with pointwise standard errors. Metadata retain each seed and numerical inputs.

References: [Chorin (1973)](https://doi.org/10.1017/S0022112073002016), [Milinazzo & Saffman (1977)](https://doi.org/10.1016/0021-9991(77)90069-9), [Roberts (1985)](https://doi.org/10.1016/0021-9991(85)90154-8), [Goodman (1987)](https://doi.org/10.1002/cpa.3160400204).
