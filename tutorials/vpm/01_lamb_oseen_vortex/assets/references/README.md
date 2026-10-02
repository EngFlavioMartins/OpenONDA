# Vortex-merger reference data

Digitized curves from Cerretelli, C. & Williamson, C. H. K. (2003), [The physical mechanism for vortex merging](https://doi.org/10.1017/S0022112002002847), *Journal of Fluid Mechanics* 475, 41–77. See the [Lamb–Oseen comparison](../../README.md).

Each CSV has two headerless columns:

| File | Horizontal coordinate | Measured quantity |
| --- | --- | --- |
| `theta_vs_tau.csv` | $\tau=\nu t/b_0^2$ | Orientation angle, degrees. |
| `a2_over_b02.csv` | $\tau$ | Squared velocity-core radius $a_c^2/b_0^2$. |
| `b_over_b0_time.csv` | Time, s | Separation $b/b_0$ for $Re_\Gamma=530$. |

The common final acquisition is digitized at $t=33.60$ s and $\tau=0.04744$. Postprocessing converts dimensional times with $\tau/t=0.04744/33.60$ s⁻¹, then converts all histories to $\nu t/a_{c,0}^2$ using the run's initialized velocity-peak radius. Original separation samples are retained without resampling. Digitization uncertainty is unavailable.
