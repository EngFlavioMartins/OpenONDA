# Leapfrogging reference

`leapfrogging_lbm_trajectory.csv` contains digitized vortex-core trajectories from Fig. 5(b) of Cheng, M., Lou, J. & Lim, T. T. (2015), [Leapfrogging of multiple coaxial viscous vortex rings](https://doi.org/10.1063/1.4915890), *Physics of Fluids* 27, 031702. See the [case comparison](../../readme.md).

The paper's axial coordinate $Z$ is the simulation's $x$. CSV coordinates are `x_over_R0` and `r_over_R0`; subtracting the reference midpoint 2.5 aligns its initial centres at $Z/R_0=2,3$ with $x/R_0=-0.5,0.5$. Only the origin changes.

Fig. 5 uses unperturbed rings at $Re_\Gamma=3000$, physical core $a_0/R_0=0.1$ and separation $h_0/R_0=1$. The mode-8, amplitude-0.05 axial perturbation belongs to the separate Fig. 3 case at $Re_\Gamma=3415$ and is not used here.

The LBM reference has periodic boundaries in a $20R_0\times7R_0\times7R_0$ box and spacing $0.005R_0$. The VPM tutorial has unbounded induction and spacing $0.05R_0$. This is a kinematic comparison; boundary and resolution differences prevent a matched validation claim.

The plot follows two dominant sampled vorticity maxima and stops identity assignment if a bridge or competing peak becomes comparable. Its cutoff is a diagnostic choice, not a physical merger criterion. Group centroids represent each initial ring's vorticity contribution; material-core shapes would require separate passive tracers.

Numerical-model sources: [Winckelmans' 1989 thesis, p. 89](https://thesis.caltech.edu/697/5/winckelmans-gs_1989.pdf) for fixed-core splitting at offsets $h/4$, and [Winckelmans (1995), Eq. 26](https://ntrs.nasa.gov/citations/19960022324) for positive-production selective viscosity. The implementation uses $h=V_p^{1/3}$ and coefficient $C=2C_w^2$; optional feedback is disabled in this case.
