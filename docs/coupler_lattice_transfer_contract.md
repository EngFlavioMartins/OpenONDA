# Common-lattice FVM→VPM transfer contract

This document defines the promise of `source/coupler/lattice_transfer.py` and
`source/coupler/vorticity_transfer.py`. It deliberately separates algebraic
properties of the mapper from physical properties that require a reconstructed
VPM field or a coupled integration test.

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

When a target lattice node is inside a solid, its strength is redistributed to
nearby fluid nodes with constrained weights reproducing constants, linear
terms, and quadratic terms. It is never simply discarded. The operation
requires at least ten linearly independent fluid nodes; otherwise it fails
before particle mutation. Each successful solve records its matrix condition
number, maximum absolute weight, and weight L1 norm. A candidate is rejected
before mutation unless those values are at most `1e6`, `8`, and `16`,
respectively.

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
solid-target redistribution, hard-release collision ownership, mutation
rollback, the matched discrete divergence operator, and one authoritative f32
post-insertion state download. They also exercise f32 fixed-point replacement
for 1,000 transfers, mutation count-check rollback, and M4′ agreement with the
GBD/DVH grid kernel. The following remain
**not certified** and are release gates for the coupled cube continuation:

- continuous Gaussian field accuracy and true `div(omega)`;
- free-space velocity neutrality and domain-padding sensitivity of the FFT
  correction;
- persistent Gaussian-tail transparency;
- repeated f32 transfer drift over a production-scale sequence;
- dynamic face/edge/corner handoff, flux/Stokes tests, and all viscous schemes;
- GBD/DVH anchor assertions, LES-state reconstruction, restart/MPI ordering;
- the offline actual-cube `t=2` audit and transfer performance envelope.
