# Leapfrogging reference

`leapfrogging_lbm_trajectory.csv` contains digitized vortex-core trajectories from Fig. 5(b) of Cheng, M., Lou, J. & Lim, T. T. (2015), [Leapfrogging of multiple coaxial viscous vortex rings](https://doi.org/10.1063/1.4915890), *Physics of Fluids* 27, 031702. See the [case comparison](../../readme.md).

The paper's axial coordinate $Z$ is the simulation's $x$. CSV coordinates are `x_over_R0` and `r_over_R0`; subtracting the reference midpoint 2.5 aligns its initial centres at $Z/R_0=2,3$ with $x/R_0=-0.5,0.5$. Only the origin changes.

Fig. 5 uses rings with $\omega_\theta=\Gamma_0e^{-(r/\sigma_0)^2}/(\pi\sigma_0^2)$, $\sigma_0/R_0=0.1$, separation $h_0/R_0=1$ and $Re_\Gamma=3000$. The authors exclude flow instabilities from these runs; the mode-8 axial perturbation belongs to their Fig. 3 at $Re_\Gamma=3415$. At $Re_\Gamma=3000$ the rings leapfrog twice and then merge: the contracting ring's core deforms strongly, loses circulation and is engulfed by the leading ring (their Fig. 4).

The LBM uses periodic boundaries in a $7R_0\times7R_0\times20R_0$ box with spacing $R_0/200$. The VPM tutorial has unbounded induction and spacing $R_0/20$.

The plot follows two dominant sampled vorticity maxima and stops identity assignment if a bridge or competing peak becomes comparable. Its cutoff is a diagnostic choice, not a physical merger criterion. Group centroids represent each initial ring's vorticity contribution; material-core shapes would require separate passive tracers.

Numerical-model sources: the sigma model of Nicoud, Toda, Cabrit, Bose & Lee (2011), [Using singular values to build a subgrid-scale model for large eddy simulations](https://doi.org/10.1063/1.3623274), *Physics of Fluids* 23, 085106; particle strength exchange after Degond & Mas-Gallic (1989), *Mathematics of Computation* 53, 485–526; strength relaxation after Pedrizzetti (1992), *Fluid Dynamics Research* 10, 101–115, in the form of [Winckelmans (1995), Eq. 11](https://ntrs.nasa.gov/citations/19960022324).
