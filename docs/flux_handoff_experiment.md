# F1 conservative flux-handoff experiment

`source/coupler/flux_handoff.py` is an isolated experimental path.  It is not
called by the production FVM-to-VPM transfer.  Its explicit injection adapter
appends an emitted batch through the native VPM particle API with zero cached
velocity; the VPM recomputes velocity at the next Runge--Kutta stage and owns
all later motion.

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
The full transport velocity advances an interpolated birth to the end of the
coupling interval.  This matters in oblique flow: using only the normal
component loses tangential phase before VPM takes ownership.

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

Every batch records nearest-neighbour distance divided by `h`, `sigma/h`, and
neighbour counts inside `2 sigma` and `3 sigma`, in addition to the pre-mutation
new/new and new/existing minimum distances.

## Manufactured and native-solver evidence

| Check | Result |
| --- | --- |
| Conservative viscous-flux sign | Pass |
| Nonzero diffusive flux in emitted circulation budget | Pass: `+2` convection and `-1` diffusion emit `+1` componentwise |
| Direct flux-vector mapping for an oblique vortex line | **Fail by identity:** inviscid contracted flux is surface-tangent, so a normal vorticity component is lost |
| Duplicate FVM patches per global slot | Pass: accumulated once |
| Unsafe birth geometry | Pass: circulation retained in a held reservoir |
| Cube-scale ratio `U Δt / h = 0.317` | Pass: births at events 4, 7, 10, 13, 16, 19 |
| Timestep sweep `U Δt / h = 0.1, 0.25, 0.317, 0.5, 1, 1.5` | Pass: `h` separation and budget closure to floating-point tolerance |
| Resolved steady vorticity sheet at a check surface `4h` downstream | 0.684% RMS vorticity error |
| Oblique carrier velocity at 0, 30, 45, 60, and 75 degrees, with surface-tangent vorticity | Pass after retaining full-velocity sub-interval phase |
| Native `VPMSolver` injection | Pass: exact position, vortex strength, volume-derived vorticity, core radius, and group ownership |
| Free VPM advection after the external source stops | Pass: counter-rotating Gaussian pair follows the analytic convection-plus-mutual-induction trajectory within `2e-9` absolute position error |
| Continuous injection into an advancing VPM | Pass: four emitted pairs, eight particles, exact emitted circulation, no pending circulation |
| Independent Gaussian-blob ODE comparison | Relative position error `6.73e-6`, `8.48e-7`, `1.10e-7` under two successive timestep halvings |
| VPM timestep refinement | Pass: error decreases by factors `7.94` and `7.75`, consistent with RK3 advection |
| Check-plane velocity RMS against independent pure-VPM reference | `2.04e-6`, `2.55e-7`, `3.56e-8` |
| Check-plane vorticity RMS against independent pure-VPM reference | `8.23e-6`, `1.02e-6`, `1.16e-7` |
| Check-plane `div(omega)` RMS against independent pure-VPM reference | `8.23e-6`, `1.02e-6`, `1.16e-7` |
| Native `FVMSolver` manufactured release plane | Pass: 64 face velocities, reconstructed vorticities, and fluxes match analytic linear shear |
| Native FVM batch entering native VPM dynamics | Pass: one VPM RK3 step agrees with independent DOP853 Gaussian-blob dynamics within `2e-10` |
| Closed 3-D vortex ring tangent to the release plane | Pass: circulation vector, first vorticity moment, Gaussian angular impulse, direction/topology, radius, and centroid |
| Ring fields across `-3h` to `+6h` | No transfer jump for the finite Gaussian reference; native VPM agrees with independent velocity and vorticity kernels |
| Molecular diffusion after injection | Pass: native core spreading gives `sigma^2=sigma_0^2+4 nu t` and analytic centre vorticity |
| Native 3-D stretching after injection | Pass: injected cloud is bitwise identical to a separately seeded pure-VPM reference while strength evolves physically |
| Evolved nearest-neighbour distribution | Minimum/1st/5th percentile `d_nn/h = 0.97885`; median `1.01967` after four interacting pair releases |

The final manufactured sheet test uses `h = 0.03125`, `sigma/h = 1`, the
cube-scale ratio `0.317`, a smooth tangential scale of `8h`, a 29 by 29
surface lattice, 25 coupling events, and seven emitted normal layers (5,887
particles).  It compares the induced VPM vorticity to the analytic FVM sheet
reference at the check surface.  It is a useful feasibility signal, not a
general field-continuity certification.

The analytic external-solver tests use a manufactured state: a uniform
normal velocity convects a counter-rotating tangential-vorticity pair through
the release surface.  The flux is formed from the conservative transport law,
accumulated by F1, and injected through `VPMSolver.add_vortex_particles`.  No
velocity override is installed and no external position update occurs after a
particle is emitted.  A single pair has an analytic constant translation speed
for the regularized Gaussian Biot--Savart kernel.  The continuous test compares
four successive pair emissions against a separate SciPy DOP853 integration of
the many-particle Gaussian-blob ODE at `rtol=1e-12`, `atol=1e-14`.

At the final downstream check plane, the same DOP853 reference supplies an
independent position state for velocity, vorticity, and vorticity-divergence
reconstruction.  All three errors decrease by approximately a factor of eight
per timestep halving.  The largest pending strength during the three runs is
`0.10`, `0.15`, and `0.175`, respectively, versus a per-particle strength of
`0.20`; it is released every physical distance `h`, not every coupling event.

The native FVM test constructs `u=(U,0,Sx)` on an actual `FVMSolver` coupling
box.  The solver's least-squares curl reconstructs `omega=(0,-S,0)` and its
internal `x=0` face geometry/interpolation provides 64 unique release slots.
F1 converts the face-integrated flux into 64 particles with componentwise
budget closure, then the unmodified native VPM advances the cloud.  This is the
current end-to-end proof that vorticity originating in another solver can be
injected without transferring ownership of particle position or velocity.

The closed ring is a compact finite-support physical-field reference rather
than an infinite sheet.  It checks vector circulation, first moment, angular
impulse, vortex-line direction, velocity, vorticity, divergence, and a dense
normal line before advancing under its own regularized Biot--Savart velocity.
Separate tests then enable native direct stretching and native core-spreading
diffusion.  The transfer is not called again in any of those evolution phases.

## Certification verdict

The restricted **surface-tangent injection and post-injection dynamics
contract is certified** by the manufactured and native-solver tests.  In that
scope, contracted vorticity flux from an actual FVM state can be converted into
native VPM particles without budget loss or unsafe birth spacing.  The
unmodified VPM then owns the particles and advances them with its own
Biot--Savart, stretching, and diffusion operators.  Independent analytic,
DOP853, and pure-VPM references all agree.

This verdict certifies the mechanics of the handoff API and native
Gaussian-blob evolution.  It does **not** certify F1 as a general 3-D physical
transfer, the production FVM/VPM coupled application, or the cube.

## What remains unproven, and why the production candidate is not certified

There is a fundamental geometric blocker.  For the inviscid convective part,

```math
n\mathbin{\cdot}\left[(u\mathbin{\cdot}n)\omega
-(\omega\mathbin{\cdot}n)u\right]=0.
```

The contracted flux vector is therefore always tangent to the release
surface.  A vortex-particle strength is instead a volume-integrated vorticity
vector and must point along the represented vortex line.  In the explicit
45-degree test, direct `Delta Gamma = F_omega Delta S Delta t` mapping emits a
tangential particle and drops the normal component, rotating the vortex-line
direction by 45 degrees, even though its contracted-flux accounting closes to
roundoff.  The passing ring, pair, and stretching-cloud cases all use source
strengths tangent to the release plane and cannot remove this limitation.

The passing check-plane field comparison is against a finite pure-VPM
reference.  The ring no-jump test also uses the same finite Gaussian basis on
the two sides of the authority switch.  Those are valid transfer and dynamics
oracles, but they do **not** demonstrate spatial refinement from a continuum
FVM field into a different finite-core VPM representation.

An exploratory steady-sheet check illustrates the distinction.  Twelve
injected normal layers reconstruct local vorticity to `0.711%` RMS, but their
velocity differs by `55.6%` from the analytic *infinite-sheet* velocity.  A
finite injected slab cannot reproduce an infinite sheet's non-local velocity,
so this is not evidence of unstable VPM dynamics; it means that this setup is
not a defensible cross-continuum velocity oracle.  It is deliberately not
counted as a pass.

Before another acceptance campaign, F1 needs a reformulation that reconstructs
a solenoidal 3-D vortex representation from the surface-flux tensor and vortex
connectivity; the contracted vector cannot simply become one particle's
strength.  That revised method then needs a compact continuum solenoidal packet
with well-posed finite FVM velocity and vorticity references, followed by
spatial `h` refinement.  The experiment also lacks a production
release-surface extractor, LES modeled flux, curved/moving surface tests,
sustained recirculation handling, and the requested F1-versus-P1 cost and
conditioning comparison.  A one-way contract remains invalid wherever
sustained inward vorticity transport is material.

The cube remains blocked.  Its checkpoint must not emit particles until the
continuum refinement gate passes and a read-only release-surface audit shows
predominantly outward transport.  The earlier volumetric persistent/replaced
ownership tests also still contain two strict expected failures and are not
made valid by F1's results.
