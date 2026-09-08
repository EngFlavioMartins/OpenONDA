# Vortex interactions

The study aims to reproduce the LBM leapfrogging trajectory and its loss of
coherent ring motion, then determine whether stabilization extends the
physically credible VPM solution. **No viscous winner or stabilized LBM match
has yet been established.**

All current study cases use LES (Smagorinsky Cs=0.20), SSPRK3, transposed
stretching, Re_Gamma=3000 and the same corrected Gaussian initial condition.
The case selects the treecode backend with
`vpm.TreecodeInduction(stretching_scheme="TRANSPOSED")`. The study's
`--stretching direct|mixed|transposed` option changes only the formulation;
qualification campaigns retain transposed stretching.

The Fig. 5 LBM trajectory is unperturbed; an imposed disturbance belongs to a
separate experiment. VPM uses unbounded induction whereas the LBM reference
uses a periodic domain; that discrepancy remains part of the qualification.

From this directory, with the OpenONDA environment active:

```sh
./allrun.sh                         # bounded LES baseline/stabilization study
./allplot.sh                        # plots from recorded VPM sampler outputs
./allrun.sh --campaign qualification  # optional broader viscous screen
```

The default budget campaign uses GBD/Lagrange6 as a working candidate, h=.04,
sigma=.04 and dt=.0075. It runs an unstabilized baseline and a weak
moment-preserving realignment trial to t=6, each with a 50-minute runtime cap,
then a half-timestep baseline check with a 30-minute cap. Including startup,
terminal output and analysis, this is intended to fit within three hours.
The cap is checked between accepted steps; an in-flight step and terminal
output are allowed to finish. Budget-limited runs have status `wall_time_limit`
and retain their terminal sampler output and native backup.

This is a bounded exploratory comparison, not a declaration of a viscous
winner or full convergence. The separate qualification campaign compares CS,
GBD/M4' and GBD/Lagrange6 at dt=.00375 and .001875 through t=.15, h=.03 and
sigma=.04; it can take much longer and is no longer the default.
Explicit `--campaign baseline|stabilized --viscous cs|gbd_m4|gbd_lagrange6`
commands remain available for extended studies.

Run the selected baseline first and inspect its actual loss of accuracy or
health stop. The stabilized campaign includes the identical baseline before
candidate interventions. For CS these are splitting, remeshing and weak
moment-preserving realignment with remeshing. For GBD these are splitting and
weak moment-preserving realignment; GBD already regenerates its particles.
These remain trials, not optimized or validated settings. A baseline is not
required to blow up, and health limits must not be loosened to manufacture
longer survival.

## Automatic output

The case configures the VPM `FlowIntegralsSampler`, `RingDiagnosticsSampler`
and `SurfaceSampler`. The solver supplies its normal health, LES, conservation
and stabilization-event diagnostics. No auxiliary particle snapshots,
reconstruction or continuation scripts are needed.

Under `study_results/<tag>/samples/diagnostics/`:

- `flow_integrals.csv`: conservation, resolution and stabilization diagnostics.
- `ring_diagnostics.csv`: particle-group geometry and impulse proxies.
- `core_section.pvd` and VTS files: velocity and curl-derived vorticity on
  z=0, y>=0, initially, every .15 physical seconds and at termination, at .02
  spacing. In this plane omega_theta equals omega_z.

Plots read these files directly. Field-core locations are maxima on the saved
plane grid, not particle-label centroids or reconstructed azimuthal averages.
The LBM comparison stops assigning core identities when two distinct maxima
cannot be resolved. That diagnostic cutoff is not by itself proof of physical
breakdown; check the fields, sampling resolution and VPM health diagnostics.

`allrun.sh` automatically plots sampler fields and writes the LBM comparison
under `figures/study/les/`. Short qualification runs do not cover the scoring
interval and therefore report the trajectory score as unavailable. Each run
records its exact configuration, source fingerprint, status and termination
reason in `result.json`. Changed configurations archive previous results;
`--resume` only reuses compatible completed runs.

Figures use the repository thesis style. Historical seeded stabilization
campaigns remain accessible via `--campaign strategies|screen|legacy`; they
are not the unperturbed LBM qualification.

See [study status](../../../docs/reviews/2026-09-vpm-core-transport.md),
[original stabilization audit](../../../docs/reviews/2026-09-vpm-stabilization.md)
and [reference provenance](assets/references/README.md).
