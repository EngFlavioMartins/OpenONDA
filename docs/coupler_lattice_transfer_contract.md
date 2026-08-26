# FVM→VPM lattice-transfer contract

This document defines the promise of `source/coupler/lattice_transfer.py` and
`source/coupler/vorticity_transfer.py`. It deliberately separates algebraic
properties of the mapper from physical properties that require a reconstructed
VPM field or a coupled integration test.

## Scope and production status

The cube-flow production candidate is `buffered_m4_renewal` with GBD. It ports
the fixed whole-belt M4′ renewal used by the historical 20-second stable run,
while retaining the current synchronized `t+1` solver and panel ordering. The
older `common_m4_lattice_blend`, projected-renewal code, and the F1 surface-flux
handoff remain available for focused experiments; they are not selected by the
cube configuration.

`evaluate_gaussian_vorticity` is an exact, untruncated float64 reference used
by certification tests. Its direct target-by-particle evaluation is not a
production-scale transfer algorithm.

## State represented

The FVM input is a collection of cell-centred, cell-integrated circulations

\[
  (x_c,V_c,\omega_c),\qquad \Gamma_c=V_c\omega_c.
\]

The VPM input is the current particle state `(x_p, Gamma_p)`. Both inputs are
scattered with the same M4′ tensor-product stencil to one complete Cartesian
lattice with the configured anchor and spacing `h`. The mapper never clips a
donor stencil to the ownership box.

The lattice state is absolute, not additive:

\[
  \Gamma_{ijk}=\eta_{ijk}\Gamma^{F}_{ijk}
      +(1-\eta_{ijk})\Gamma^{V}_{ijk}.
\]

With a positive blend width, `eta` is the product of six C1 cosine face
windows. With zero width it is hard ownership. A persistent particle on an
outer lattice node receives any release-stencil contribution; it is not
overwritten by that contribution.

## Mapper invariants

On a complete M4′ stencil in float64, the pure mapper reproduces, component by
component, total vortex strength and all monomials through degree two:

\[
 \sum_p\Gamma_p,\quad \sum_p x_{p,j}\Gamma_{p,i},\quad
 \sum_p x_{p,j}x_{p,k}\Gamma_{p,i}.
\]

`sum(|Gamma_p|)` is explicitly **not** an invariant: M4′ has legitimate
negative lobes. The invariant is the signed vector sum.

The common-lattice path reports circulation excluded from solid target nodes
explicitly. The buffered production candidate instead builds solid confidence
and taper information into the fixed renewal lattice, applies bounded local
redistribution and pruning, then recovers its signed circulation and impulse
invariants before particle mutation.

## Buffered whole-belt renewal

The buffered method uses a fixed lattice covering the FVM authority box, the
distance a particle can travel in one coupling interval, and the complete
two-cell M4′ guard. At each handoff it:

1. separates the renewable belt from the untouched outer wake;
2. scatters every renewable VPM particle to that one lattice;
3. reconstructs the synchronized FVM target from the current velocity trace;
4. blends represented FVM and VPM states with inward FVM authority that is
   exactly zero at and beyond the ownership boundary;
5. prunes and corrects the lattice state under bounded amplification and
   invariant checks; and
6. atomically replaces the complete managed belt while appending the untouched
   outer wake.

Renewing the complete belt, including its managed release support, is the
anti-accumulation contract. A released particle becomes persistent only after
it has left that belt; it is not independently re-added at every handoff.

## Solver-level invariant definitions

Certification tests must use `ParticleFieldEvaluation`, whose discrete
definitions are:

\[
  \Gamma=\sum_p\Gamma_p,\qquad
  I=\tfrac12\sum_p x_p\times\Gamma_p,
\]

\[
  A=\tfrac13\sum_p x_p\times(x_p\times\Gamma_p)
    -\tfrac{2}{9}C_{\rm kernel}\sum_p\sigma_p^2\Gamma_p.
\]

The last term is important whenever particle core radii vary. Geometric
quadratic-moment preservation alone is therefore not a claim that angular
impulse is preserved if the transfer changes core radii.

## Divergence contract

For a positive blend width, the correction source is the actual residual of
one periodic spectral operator `D_h`:

\[
 r=D_h[\eta\Gamma_F+(1-\eta)\Gamma_V]
   -\eta D_h[\Gamma_F]-(1-\eta)D_h[\Gamma_V].
\]

The same `D_h` forms the FFT Poisson correction and reports the remaining
residual. This is a lattice diagnostic only. It does not certify free-space
boundary behaviour, continuous Gaussian `div(omega)`, or velocity neutrality
of the periodic correction.

## Mutation contract

Capacity, duplicate-node, and geometric validation occurs before mutation. If
`update_particle_vortex_strength`, `remove_particles`, `add_vortex_particles`,
or the post-mutation count check raises, the complete VPM state is restored
through `replace_vortex_particles`: position, velocity, Gamma, core radius,
volume, molecular and eddy viscosity, group/zone IDs, velocity gradient, and
strain rate.

## Current certification status

The unit regressions cover M4′ moment identities, C1 ownership seams,
release-support ownership, mutation rollback, invariant recovery, the matched
discrete divergence operator, the exact Gaussian reference, and one
authoritative f32 post-insertion state download. The following remain
**not certified** and are release gates for the coupled cube continuation:

- continuous Gaussian field accuracy and true `div(omega)`;
- free-space velocity neutrality and domain-padding sensitivity of the FFT
  correction;
- persistent Gaussian-tail transparency;
- repeated f32 transfer drift over a production-scale sequence;
- the one-step and next-handoff cube checkpoint gates with GBD;
- dynamic face/edge/corner handoff and long-time coupled accuracy;
- GBD/DVH anchor assertions, LES-state reconstruction, restart/MPI ordering;
- the offline actual-cube `t=2` audit and transfer performance envelope.

The isolated F1 experiment and its current geometric limitation are documented
in `docs/flux_handoff_experiment.md`; passing F1 unit tests does not certify it
for cube production.
