# F1 conservative flux-handoff experiment

`source/coupler/flux_handoff.py` is an isolated experimental path.  It is not
called by the production FVM-to-VPM transfer and it does not mutate the VPM
solver state.

The experiment tests whether a release surface inside the FVM domain can give
the two properties the common-lattice replacement lacks: conservative
circulation handoff and safe, `h`-spaced particle birth geometry.  It does not
claim that those properties alone make the FVM/VPM velocity field continuous.

## Flux convention

For incompressible flow with constant kinematic viscosity,

```math
\partial_t \omega_i + \partial_j
\left(u_j\omega_i-u_i\omega_j-\nu\partial_j\omega_i\right)=0.
```

With an outward surface normal, the corresponding vorticity transport flux is

```math
F_\omega=(u\mathbin{\cdot}n)\omega-(\omega\mathbin{\cdot}n)u
          -\nu\partial_n\omega.
```

The viscous term therefore has a **minus** sign.  A plus sign is inconsistent
with the conservative equation above under the stated outward-normal
convention.

## F1 release contract

Each globally identified release slot owns a circulation reservoir.  During a
coupling interval it receives the outward, area-integrated vorticity flux;
duplicate FVM patches for the same slot are summed before the reservoir is
updated.  The reservoir advances through the physical normal transport
distance rather than one particle per coupling event.  It emits a particle at
each exact `h` crossing, using the flux interpolated over that subinterval.

Candidate births are checked against pre-existing and already-created VPM
particles.  A candidate closer than `h` is held in its slot reservoir: its
circulation is neither discarded nor merged.  Inward-facing patches are
recorded but are not released through this one-way handoff.  The accounting
invariant is

```math
\Gamma_{\rm received}=\Gamma_{\rm emitted}+\Gamma_{\rm pending}.
```

This is a birth/release mechanism, not a replacement for the VPM velocity,
diffusion, stretching, remeshing, or FVM boundary treatment.

## Initial manufactured evidence

| Check | Result |
| --- | --- |
| Conservative viscous-flux sign | Pass |
| Duplicate FVM patches per global slot | Pass: accumulated once |
| Unsafe birth geometry | Pass: circulation retained in a held reservoir |
| Cube-scale ratio `U Δt / h = 0.317` | Pass: births at events 4, 7, 10, 13, 16, 19 |
| Timestep sweep `U Δt / h = 0.1, 0.25, 0.317, 0.5, 1, 1.5` | Pass: `h` separation and budget closure to floating-point tolerance |
| Resolved steady vorticity sheet at a check surface `4h` downstream | 0.684% RMS vorticity error |

The final manufactured sheet test uses `h = 0.03125`, `sigma/h = 1`, the
cube-scale ratio `0.317`, a smooth tangential scale of `8h`, a 29 by 29
surface lattice, 25 coupling events, and seven emitted normal layers (5,887
particles).  It compares the induced VPM vorticity to the analytic FVM sheet
reference at the check surface.  It is a useful feasibility signal, not a
general field-continuity certification.

## What remains unproven

This experiment does not yet exercise a real FVM surface, face-gradient
reconstruction, variable viscosity, LES closures, vortex stretching, or a
moving/curved release surface.  It also does not certify velocity continuity,
the no-jump condition, divergence behaviour, topology preservation, or the
full cube.  The one-way contract is deliberately inapplicable where sustained
inward vorticity transport must be represented.

The next meaningful gate is a resolved three-dimensional manufactured vortex
packet with velocity, vorticity, divergence, and no-jump checks on both sides
of the release surface.  The cube must remain blocked until those gates and
the persistent/replaced-state ownership gate are passed.
