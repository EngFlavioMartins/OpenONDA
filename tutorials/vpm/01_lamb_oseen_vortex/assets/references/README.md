# Cerretelli-Williamson merger reference data

The CSV files in this directory are digitized curves from C. Cerretelli and
C. H. K. Williamson, *The physical mechanism for vortex merging*, Journal of
Fluid Mechanics 475 (2003), 41-77, DOI
[`10.1017/S0022112002002847`](https://doi.org/10.1017/S0022112002002847).

Each file has two headerless columns:

- `theta_vs_tau.csv`: `nu*t/b0^2`, orientation angle in degrees.
- `a2_over_b02.csv`: `nu*t/b0^2`, `a_c^2/b0^2`.
- `b_over_b0_time.csv`: the published figure 4 measurements of `b/b0` against
  dimensional time for the `Re=530` experiment.

Figure 5(a,b) explicitly uses the same experiment and timescale on
`tau=nu*t/b0^2`. The common final acquisition is digitized at `t=33.60 s` in
figure 4 and `tau=0.04744` in figure 5(b), so `postprocess.py` applies the
linear conversion `tau/t=0.04744/33.60 s^-1` directly to every original
figure 4 sample. No rank-resampling or interpolation of `b/b0` is used.

`postprocess.py` then converts all three reference histories to
`nu*t/velocity_peak_radius_0^2` using the actual initialized velocity-peak
radius from each run's metadata.
The digitization uncertainty is not available, so the solid literature curves
must not be interpreted as uncertainty-free measurements.
