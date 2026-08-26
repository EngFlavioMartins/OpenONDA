# Cerretelli-Williamson merger reference data

The CSV files in this directory are digitized curves from C. Cerretelli and
C. H. K. Williamson, *The physical mechanism for vortex merging*, Journal of
Fluid Mechanics 475 (2003), 41-77, DOI
[`10.1017/S0022112002002847`](https://doi.org/10.1017/S0022112002002847).

Each file has two headerless columns:

- `theta_vs_tau.csv`: `nu*t/b0^2`, orientation angle in degrees.
- `a2_over_b02.csv`: `nu*t/b0^2`, `a_c^2/b0^2`.
- `b_over_b0_tau.csv`: a historical rank-resampling of the dimensional
  vortex_separation data onto the core-radius coordinates. This is retained only for
  auditability and is deliberately **not plotted** because rank-resampling is
  not a physical time conversion.
- `b_over_b0_time.csv`: original digitized dimensional-time coordinate and
  `b/b0`; retained as source data. It remains unplotted until the experiment's
  `nu/b0^2` conversion is established from primary-source parameters; matching
  the span of another digitized curve is not a physical time conversion.

The plotting script converts the first coordinate to `nu*t/core_radius_0^2` using the
actual initialized velocity-peak core radius from each run's metadata.
The digitization uncertainty is not available, so the solid literature curves
must not be interpreted as uncertainty-free measurements.
