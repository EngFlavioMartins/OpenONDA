# Vortex-ring interactions: DNS, LES, and stabilized LES

This tutorial compares two fully three-dimensional vortex-ring interactions:
equal co-propagating rings that leapfrog, and equal counter-propagating rings
that collide. Both rings carry a reproducible broadband Widnall perturbation.

`./allrun.sh` runs the complete six-case matrix:

| Interaction | DNS | LES | LES + stabilization |
|---|---|---|---|
| Leapfrogging | `leapfrog_dns` | `leapfrog_les` | `leapfrog_les_stabilized` |
| Collision | `collide_dns` | `collide_les` | `collide_les_stabilized` |

DNS contains molecular viscosity only. LES adds the Smagorinsky eddy viscosity
with the deliberately visible coefficient `C_s = 0.32`. The third variant adds
stretching-aware residual viscosity,

```text
nu_stab = C_stab Delta^2 max(Gamma . S Gamma / |Gamma|^2, 0),  C_stab = 0.5.
```

It acts only where strain locally amplifies a vortex-line element. It does not
clip or directly change circulation; it enters core spreading through
`nu_eff = nu + nu_t + nu_stab`, so its energy removal is measured by the same
viscous-energy budget as DNS and LES.

Residual viscosity alone cannot repair a particle cloud whose vortex vectors
have become geometrically misaligned. The stabilized variant therefore checks
the reconstructed divergence and circulation-weighted misalignment every 20
steps. Once either exceeds `0.20` or `20 deg`, respectively, it scatters the Gaussian field onto a
well-overlapped grid and discards at most 0.3% of the absolute-circulation tail.
A minimum-norm correction then restores vector circulation, linear impulse, and
finite-core angular impulse. A candidate is accepted directly when both energy
and enstrophy decrease by less than 30% and 15%, respectively. If the raw
candidate would inject either quantity or over-dissipate enstrophy, a correction
in the null space of those nine constraints restores the production Gaussian
enstrophy integral to within `5e-6` while requiring energy to decrease.
If the rebuilt cloud still has divergence error above `0.08`, a constrained
Helmholtz projection reduces it while preserving the same moments and retaining
the non-injecting energy/enstrophy gates.
Once the stabilized cloud reaches its 8,000-particle budget, later health
interventions are projection-only: the solver does not repeatedly remesh an
expanding support or discard an ever larger circulation tail. Before each such
projection, the current cloud is retained unchanged; the projected operation is
accepted only when its net production energy and enstrophy transfers remain
nonpositive.

The regularization is accepted only if kinetic energy decreases. Its exact
energy transfer is exported separately and is bounded to 30% per event; in the
validated setup the transfers are far smaller. Thus this is a conservative LES
filter with an explicit energy ledger, not a hidden remeshing source or a
strength clip. The health trigger also means that a resolved cloud is left
untouched.

All six cases use the same coupled RK2 advection/stretching discretization and
the same minimum-norm stage projection for vector circulation, linear impulse,
angular impulse, and discrete inviscid energy. This shared projection corrects
finite-particle conservation error; it is not the stabilization being compared.
Interactions are evaluated directly, so a tree-approximation change cannot be
mistaken for a stabilization effect.
DNS and plain LES use the fixed cloud. Stabilized LES adds the residual
viscosity and the audited conservative filter described above.

The code reports `Omega = integral(|omega|^2 dV)`. For constant viscosity the
energy identity is therefore `dE/dt = -nu Omega`; with the alternative
`Enstrophy = Omega/2` convention it is `dE/dt = -2 nu Enstrophy`. LES and the
stabilized case have spatially varying viscosity, so the CSV's
`neg_nu_enstrophy` column contains the exact weighted quadratic sink rather
than the generally incorrect approximation `-mean(nu_eff) Omega`.
Across a regularization event the complete discrete balance is

```text
Delta E = integral(neg_nu_enstrophy dt) + Delta E_filter,
```

where `regularization_cumulative_energy_transfer` records the second, always
non-positive term. The automated gate checks this combined balance as well as
strictly decreasing energy and conservation of both impulses.

The deliberately under-modelled DNS and plain-LES baselines may eventually
lose fixed-particle resolution. They stop cleanly when the circulation-weighted
vortex-line misalignment exceeds 45 degrees or the reconstructed normalized
vorticity-divergence error exceeds 0.25, record that final diagnostic state, and
declare `status: resolution_lost` in their manifests. Peak particle strength is
plotted but is not itself treated as failure because material line stretching
can increase it physically. The two stabilized cases are not allowed to stop
early: the automated gate requires both to reach the full requested final time.

Run all cases after installing OpenONDA:

```sh
./allrun.sh
```

Run one or more named cases with, for example:

```sh
./allrun.sh leapfrog_dns leapfrog_les_stabilized
```

The setup selects the CPU backend intentionally, making the same tutorial work
without CUDA on Linux and macOS. Existing results are never overwritten. Remove
one case or the complete generated result set explicitly:

```sh
./allclean.sh collide_dns
./allclean.sh --all
```

Results are written under `solution/CASE/`. Each case contains a manifest,
flow-integral CSV, solver log, periodic backups, and a final HDF5/XDMF state.
After all six runs pass the physics and comparative-stability gate, generate
the PNG and PDF diagnostics with:

```sh
./allplot.sh
```
