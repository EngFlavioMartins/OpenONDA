# Methods and scope

This package preserves two accepted standalone studies. The main result is the
fixed-core Gaussian experiment; the earlier linear/synthetic experiment is a
supporting comparator. Neither script imports or runs the native production
solver.

## Main fixed-core study

The state is `Y=(X,Gamma)`: particle positions and vector strengths. Velocity
and its gradient are recomputed from the complete stage state; velocity is not
a separately integrated state variable. The isolated NumPy probe freezes the
evaluated conventions used in the original review:

- direct pair summation with `sigma_ij=(sigma_i+sigma_j)/2`;
- the then-production Abramowitz--Stegun approximation with its
  `rho<0.2` series branch;
- transposed strength evolution `(grad u)^T Gamma`;
- fixed particle count, volumes, and positive core sizes during integration.

The experiment excludes viscosity, core evolution, LES, relaxation,
redistribution/remeshing, VLM forcing, particle insertion/deletion, and
tree/FMM approximation. Later corrections to repository kernel source files
are not inputs to the frozen probe; the evaluated formula is preserved in
[`kernel_snapshot.json`](provenance/kernel_snapshot.json) and
[`fixed_core_probe.py`](provenance/frozen_scripts/fixed_core_probe.py).

The temporal/geometry study uses `N=27`: a regular cloud, two seeded 18%
jitter clouds, and a deliberately clustered/voided cloud. The volume-based
particle spacing is `h=V_p^(1/3)=1`, and each cloud is evaluated at
`sigma/h` in `{0.6, 1.0, 1.5}`. Every time-integration method advances to
`T=0.4` with `dt` in `{0.05, 0.025, 0.0125, 0.00625}`. A coupled RK4 solution
with 512 steps is the common reference. The methods are:

- **Coupled SSPRK3:** one SSPRK3 tableau applied to the full `A+B` vector
  field, three full right-hand-side evaluations per step.
- **Sequential RK3:** the historical full-step `A` (position) update followed
  by a full-step `B` (strength) update, with SSPRK3 applied to both; six defined
  evaluations per step.
- **Symmetric split:** `A/2`, `B`, `A/2`, each advanced with SSPRK3; nine
  defined evaluations per step.

These are evaluator counts in the standalone probe, not production timings.
The reported temporal error is the RMS of the complete scaled Euclidean state
difference: positions are scaled by `h=1`, strengths by their initial RMS,
and the concatenated norm is divided by `sqrt(6N)`.

The `N=8` tangent experiment takes six steps at `dt=0.05`. It estimates each
step Jacobian by centered differences in the same full scaled Euclidean
coordinates, multiplies the step maps, and compares the largest singular value
with a refined RK4 flow-map estimate. Five nominal-trajectory samples provide
a trapezoidal logarithmic-norm diagnostic. These are numerical estimates, not
certified continuous bounds.

The geometry diagnostics are particle-site finite-cloud measurements. The
fill-distance value in the raw table is the maximum over a `9x9x9` sampling
grid, not the continuum supremum. Moment values include boundary truncation.

## Supporting comparator

The earlier comparator contains exact two-way-coupled linear oscillator and
growing systems plus a smooth two-particle **Gaussian-like synthetic field**.
It does not use the production Gaussian/Winckelmans kernel. The coupled and
sequential methods use SSPRK3; the comparator's symmetric split uses classical
RK4 subflows and therefore has 12 right-hand-side calls per macro step. It is
included to expose order and the oscillator stability counterexample cleanly,
not as physical VPM validation or a cost model.

## Plot construction

[`plot_results.py`](scripts/plot_results.py) reads only the accepted copied
tables. It calls the repository's `openonda.plotting.set_thesis_style()` and
uses its 12.5 cm width, 10.95 pt size, palette, and LaTeX/newpx typography.
Consequently, figure regeneration requires the same LaTeX, `dvipng`,
`newpxtext`, and `newpxmath` dependencies as other thesis plots. Each figure is
exported as 400 dpi PNG plus vector PDF and SVG.

PNG and SVG output is byte-stable in repeated runs in the documented
environment. TeX-generated PDF font-subset prefixes may vary between otherwise
equivalent runs. After intentionally regenerating the shipped figures in a
repository checkout, refresh and verify their recorded hashes with
`python3 scripts/verify.py --write-manifest`. Ordinary integrity checking uses
`python3 scripts/verify.py` and does not require the historical source folders.

## Reproduction boundary

Use [`run_fixed_core.py`](scripts/run_fixed_core.py) or
[`run_comparator.py`](scripts/run_comparator.py) to run an exact frozen script
into a `reproduced/` subdirectory without overwriting the accepted baseline.
Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` to
retain the recorded one-CPU execution boundary. No GPU or package installation
is needed in the documented environment.
