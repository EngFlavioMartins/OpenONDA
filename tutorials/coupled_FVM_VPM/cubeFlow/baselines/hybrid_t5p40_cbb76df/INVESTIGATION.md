# cubeFlow coupling investigation

This note records the controlled comparisons made from checkpoint
`cbb76df776797147d92ef15e53d619f2c1694bb9`. The interrupted production run
was preserved at `t = 5.40` before any solver changes.

## Baseline observations

- Over the first 36 common force samples (`t = 0.15` through `5.40`), the
  hybrid/reference mean absolute relative drag difference is 4.82%.
- Over the last 10 available samples, mean `Cd` is 0.95170 (hybrid) versus
  0.90900 (fine reference), a +4.70% difference.
- At `t = 5.40`, normalized `Ux` RMS differences on the `z = 0` slice are
  1.40% for hybrid FVM versus VPM, 9.27% for hybrid FVM versus the fine
  reference, and 9.65% for VPM versus the fine reference. The two coupled
  representations therefore remain close while the hybrid trajectory becomes
  phase-separated from the reference.
- A 0.15 s lag gives the best drag-history correlation after `t = 1.5`
  (`r = 0.896`, versus `0.838` without a lag), confirming that instantaneous
  LES phase error is part of the later field difference.

## Startup control matrix

All values below are at the first common sample, `t = 0.15`. The current-code
fine reference reproduced the checked-in reference `Cd` to approximately
`1e-10`, so the archived samples remain numerically valid.

| Control | Cells | Near-wake spacing | Beale passes | Cd | Error vs fine reference |
|---|---:|---:|---:|---:|---:|
| Fine full-domain reference | 2,013,232 | 0.02841 | n/a | 1.952801 | 0.00% |
| Resolution-matched full-domain reference | 900,584 | 0.03906 | n/a | 1.802252 | -7.71% |
| Production hybrid | 543,912 | 0.04000 | 1 | 1.813169 | -7.15% |
| Fine-FVM hybrid (`FVM=0.03`, `VPM h=0.04`) | 1,012,160 | 0.03000 | 1 | 1.881948 | -3.63% |
| Fine-FVM hybrid, deconvolution ablation | 1,012,160 | 0.03000 | 4 | 1.996772 | +2.25% |

The production hybrid differs from the resolution-matched full-domain control
by only +0.61% in `Cd`. Its VPM boundary trace also matches the fine reference on the
inner-box interface to 0.43% RMS at this time. The original -7.15% startup gap
is therefore primarily a mesh-resolution comparison, not a failed coupling
boundary.

Refining only the Eulerian mesh closes about half the gap without increasing
the VPM diffusion-grid memory. Refining both methods to `h = 0.03` is not a
one-knob solution: the production VPM box exceeds the configured GPU-grid
memory guard, and a startup-only reduced-box test with `dt_VPM = 0.05`
violated the stretching stability recommendation and produced non-finite VPM-BC
velocities on its second step.

Four Beale/Picard passes reduced the startup mollification residual but
overshot drag, increased particle count, and tightened the stretching margin.
Forcing that residual to zero amplifies circulation that the chosen particle
resolution cannot safely carry; a fixed multi-pass correction is rejected.

## Confirmed diagnostic defects

1. VPM line samplers overwrote their CSV at every sample, so velocity-profile
   plots silently retained only the final VPM time.
2. The fine-reference comparison panel plotted hybrid FVM data but labelled it
   as VPM.
3. The field plot called `100 |Delta Ux| / U_inf` a generic percent error and
   masked locations where the left field was near zero. It is an absolute
   normalized difference and remains meaningful there.
4. `flux_ratio` compared Beale-deconvolved raw particle strengths with the
   physical FVM circulation trace. This produced values near 2.5 while direct
   sampled VPM/FVM outflow vorticity remained near one. The diagnostic must
   mollify particle strengths before forming the L1 ratio.

## Root-cause ranking

1. **Resolution mismatch in the validation comparison.** The production
   hybrid uses 0.04 near-wake cells, while the fine reference uses about
   0.0284. A matched full-domain control removes nearly all startup drag error.
2. **Instantaneous phase comparison of an unsteady LES.** Once shedding
   develops, pointwise fields and instantaneous drag should be supplemented by
   time-averaged fields, mean/RMS forces, and spectra over a statistically
   stationary window.
3. **Diagnostics that hid history or compared incompatible representations.**
   These are corrected in the accompanying changes.
4. **Residual handoff mollification error.** It is real, but the four-pass
   ablation shows that blindly converging the deconvolution is not the missing
   fix.

No sign, gradient-layout, panel-scope, VPM-BC-flux, or blending-complement defect
was found. The panel in `vpm_bc` scope solves against the particle wake and is
added only to FVM VPM-BC and blending-zone evaluations, avoiding body-potential
double-counting in particle advection.
