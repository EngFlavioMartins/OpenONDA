# Vortex-ring interactions: DNS, LES, and stabilized LES

This tutorial compares two fully three-dimensional vortex-ring interactions:
equal co-propagating rings that leapfrog, and equal counter-propagating rings
that collide. Both rings carry a reproducible broadband Widnall perturbation.

Each ring is initialized with particle spacing `0.04` and Gaussian particle
radius `0.08`. This resolves the physical core diameter with five particle
intervals and seeds 19 complete cross-section orbits per ring (6,688 particles
for the two-ring initial condition). Complete azimuthal loops preserve the
ring symmetries and vector circulation. The time step keeps
`Delta t Gamma/R_0^2 = 0.032`; 1,140 steps retain the previous long physical
horizon, `t Gamma/R_0^2 = 36.48`.

`./allrun.sh` runs the complete six-case matrix:

| Interaction | DNS | LES | LES + stabilization |
|---|---|---|---|
| Leapfrogging | `leapfrog_dns` | `leapfrog_les` | `leapfrog_les_stabilized` |
| Collision | `collide_dns` | `collide_les` | `collide_les_stabilized` |

DNS contains molecular viscosity only. LES adds the Smagorinsky eddy viscosity.
The sustained leapfrogging deformation uses the classical `C_s = 0.16`; the
more violent head-on collision uses `C_s = 0.32`. Each stabilized case uses the
same coefficient as its plain-LES reference. The third variant adds
stretching-aware residual viscosity,

```text
nu_stab = C_stab Delta^2 max(Gamma . S Gamma / |Gamma|^2, 0),  C_stab = 0.5.
```

It acts only where strain locally amplifies a vortex-line element. It does not
clip or directly change circulation; it enters core spreading through
`effective_viscosity = nu + eddy_viscosity + nu_stab`, so its energy removal is measured by the same
viscous-energy budget as DNS and LES.

Residual viscosity alone cannot repair a particle cloud whose vortex vectors
have become geometrically misaligned. The stabilized variant therefore checks
the reconstructed divergence and circulation-weighted misalignment every 20
steps. Once either exceeds `0.20` or `4 deg`, respectively, it scatters the
Gaussian field onto a well-overlapped grid and budgets at most 0.3% of the
absolute-circulation tail before the particle-count limit is applied.
A minimum-norm correction then restores vector circulation, linear impulse, and
finite-core angular impulse. A candidate is accepted directly when energy and
enstrophy both decrease by less than 20% and 15%, respectively. If the raw
candidate would inject either quantity or over-dissipate enstrophy, a correction
in the null space of those nine constraints restores the production Gaussian
enstrophy integral to within `5e-6` while requiring energy to decrease.
If the rebuilt cloud still has divergence error above `0.12`, a constrained
Helmholtz projection reduces it while preserving the same moments and retaining
the non-injecting energy/enstrophy gates. The projection itself may change the
circulation field by at most 5% for leapfrogging and 10% for the stronger
head-on collision.

The regular grid has spacing `0.084`; the regenerated Gaussian core radius is
`0.23` for leapfrogging and `0.195` for collision. Above 70% of the
20,000-particle budget, the solver enters capacity mode:
it uses the capacity health triggers `0.20` and `25 deg`, a `0.13` grid, and a
`0.195` core. Separating grid spacing from blob radius avoids the previous
failure in which obtaining population headroom also broadened every blob and
removed excessive enstrophy. The capacity lattice leaves useful population
headroom while the independently controlled core keeps the transfer within the
declared physics gates. If resetting to the configured core would inject energy
or enstrophy,
the filter broadens that event's regenerated core in 5% increments and accepts
the first non-injecting candidate. It aborts instead of hiding an injection
behind a large circulation-field correction.

The regularization is accepted only if kinetic energy decreases. Its exact
energy transfer is exported separately and is bounded to 20% per event; the
automated gate reports the largest accepted transfer. Thus this is a
conservative LES filter with an explicit energy ledger, not a hidden remeshing
source or a strength clip. The health trigger also means that a resolved cloud
is left untouched.

All six cases use the same coupled RK2 advection/stretching discretization and
the same minimum-norm stage projection for vector circulation, linear impulse,
angular impulse, and discrete inviscid energy. A final minimum-norm correction
removes the finite-step defect of the bilinear impulses after each coupled RK
substep. These shared projections correct finite-particle/time-integration
error; they are not the stabilization being compared.
The cases run in double precision. After every core-spreading half-step, a
second minimum-norm projection restores circulation and both impulses; this is
needed because independently spreading blobs at a spatially varying LES
viscosity otherwise creates a spurious finite-core angular impulse. Its
correction and residual are exported and gated separately.
Interactions are evaluated directly, so a tree-approximation change cannot be
mistaken for a stabilization effect.
DNS and plain LES use the fixed cloud. Stabilized LES adds the residual
viscosity and the audited conservative filter described above.
The 1,140-step endpoint coincides with the 20-step filter cadence, so the final
state is itself health-checked and cannot end between scheduled repairs.

Global flow integrals and the two grouped-ring histories are sampled by the VPM
solver every five steps (`Delta t_sample Gamma/R_0^2 = 0.16`). The former are written to
`samples/flow_integrals.csv`; the built-in `RingDiagnosticsSampler` writes
centroids, major core_radius, tube circulation, impulse, and strength measures to
`samples/ring_diagnostics.csv`. These are the only quantitative inputs used by
the plotting scripts—solver logs are not parsed and plots do not reconstruct
diagnostics from particle files.

Full HDF5/XDMF particle states are stored every ten steps
(`Delta t_frame Gamma/R_0^2 = 0.32`), including step zero,
giving 115 regularly spaced frames for each full-duration case. This cadence is
independent of diagnostic sampling and is intended for smooth ParaView playback.
Open the numbered `vpm_CASE_*.xdmf` sequence in ParaView as a file series. The
final restart-capable state is also written explicitly.

The code reports `Omega = integral(|omega|^2 dV)`. For constant viscosity the
energy identity is therefore `dE/dt = -nu Omega`; with the alternative
`Enstrophy = Omega/2` convention it is `dE/dt = -2 nu Enstrophy`. LES and the
stabilized case have spatially varying viscosity, so the CSV's
`neg_nu_enstrophy` column contains the exact weighted quadratic sink rather
than the generally incorrect approximation `-mean(effective_viscosity) Omega`.
Across a regularization event the complete discrete balance is

```text
Delta E = integral(neg_nu_enstrophy dt) + Delta E_filter,
```

where `regularization_cumulative_total_kinetic_energy_transfer` records the second, always
non-positive term. The automated gate checks this combined balance as well as
strictly decreasing energy and conservation of both impulses. It permits at
most 2% error in the integrated balance and 20% RMS mismatch in the sampled
instantaneous rate, whose backward energy difference is compared with the
trapezoidal viscous sink plus the filter-transfer rate.

The deliberately under-modelled DNS and plain-LES baselines may eventually
lose fixed-particle resolution. They stop cleanly when the circulation-weighted
vortex-line misalignment exceeds 45 degrees or the reconstructed normalized
vorticity-divergence error exceeds the 0.12 warning threshold, record that final
diagnostic state, and
declare `status: resolution_lost` in their manifests. Peak particle strength is
plotted but is not itself treated as failure because material line stretching
can increase it physically. The two stabilized cases are not allowed to stop
early: the automated gate requires both to reach the full requested final time.
The warning threshold leaves sampling headroom below the hard accepted-state
divergence bound of 0.25.

The comparative gate requires plain LES to outlive DNS by at least 5% and to
improve at least two of strength growth, divergence, and vortex-line
misalignment. Stabilized LES must improve two of the same indicators by at
least 10%. Each comparison uses the two models' common physical-time window;
the absolute physics gates still inspect every sample of the longer run.

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

Results are written under `solution/CASE/`. Each case contains a manifest, both
sampler CSVs, a solver log, periodic visualization states, and a final
HDF5/XDMF state.
After all six runs pass the physics and comparative-stability gate, generate
the PNG and PDF diagnostics with:

```sh
./plot_all.sh
```
