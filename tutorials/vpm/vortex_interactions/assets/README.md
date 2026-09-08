# Vortex interaction analysis

The current workflow is documented in the [tutorial README](../readme.md).
It uses LES, SSPRK3 and transposed stretching throughout qualification,
baseline and stabilization comparisons.

`plot_core_sections.py` reads the VPM SurfaceSampler VTS/PVD output and renders
signed curl-derived vorticity with the shared thesis figure style.
`assess_lbm_agreement.py` locates grid-resolved maxima on those same recorded
planes, follows a separated pair and compares its radius-versus-distance
trajectory with the digitized LBM reference. It records source-field hashes,
uses a fixed midpoint origin and does not fit a phase shift or extrapolate.
A third maximum, clipped core or strong bridge terminates identity assignment;
these are diagnostic cutoffs, not physical breakdown criteria. Plane-grid and
output-cadence refinement remain required checks.

`allrun.sh` invokes both automatically. For selected existing sampler outputs, run from the case directory containing `setup.py`:

```sh
python -m openonda.tutorial_runner . assets.plot_core_sections --runs RUN_NAME
python -m openonda.tutorial_runner . assets.assess_lbm_agreement RUN_NAME
```

`plot_rings_*.py` read flow and ring sampler CSV files. Material-group radius
and circulation are proxies; group mixing during remapping prevents their use
as independent proof of coherent core identity or merger.

Older snapshot-based utilities in this directory belong to the historical
stabilization audit. They are not dependencies of the current LES workflow.
The abandoned no-LES transport campaign data and auxiliary continuation
helpers were removed at the user's request; compact useful findings are
retained in `docs/reviews/2026-09-vpm-supporting-controls.json`.

See [reference provenance](references/README.md) for the distinction between
the unperturbed Re=3000 trajectory and the separate seeded instability case.
