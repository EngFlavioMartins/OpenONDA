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
| Native `VPMSolver` injection | Pass: exact position, vortex strength, volume-derived vorticity, core radius, and group ownership |
| Free VPM advection after the external source stops | Pass: counter-rotating Gaussian pair follows the analytic convection-plus-mutual-induction trajectory within `2e-9` absolute position error |
| Continuous injection into an advancing VPM | Pass: four emitted pairs, eight particles, exact emitted circulation, no pending circulation |
| Independent Gaussian-blob ODE comparison | Relative position error `6.73e-6`, `8.48e-7`, `1.10e-7` under two successive timestep halvings |
| VPM timestep refinement | Pass: error decreases by factors `7.94` and `7.75`, consistent with RK3 advection |

The final manufactured sheet test uses `h = 0.03125`, `sigma/h = 1`, the
cube-scale ratio `0.317`, a smooth tangential scale of `8h`, a 29 by 29
surface lattice, 25 coupling events, and seven emitted normal layers (5,887
particles).  It compares the induced VPM vorticity to the analytic FVM sheet
reference at the check surface.  It is a useful feasibility signal, not a
general field-continuity certification.

The real-solver tests use a manufactured external solver state: a uniform
normal velocity convects a counter-rotating tangential-vorticity pair through
the release surface.  The flux is formed from the conservative transport law,
accumulated by F1, and injected through `VPMSolver.add_vortex_particles`.  No
velocity override is installed and no external position update occurs after a
particle is emitted.  A single pair has an analytic constant translation speed
for the regularized Gaussian Biot--Savart kernel.  The continuous test compares
four successive pair emissions against a separate SciPy DOP853 integration of
the many-particle Gaussian-blob ODE at `rtol=1e-12`, `atol=1e-14`.

## Certification verdict

The narrow F1 injection/advection contract is certified by the manufactured
tests: solver-independent conservative vorticity flux can be converted into
native VPM particles without circulation loss or unsafe birth spacing, after
which the unmodified VPM advances the particles according to its own
regularized Biot--Savart dynamics with the expected third-order timestep
convergence.

This verdict certifies the handoff API and inviscid Gaussian-blob evolution. It
does not certify the production FVM/VPM coupled application or the cube.

## What remains unproven

This experiment does not yet exercise a real FVM surface, face-gradient
reconstruction, variable viscosity, LES closures, vortex stretching,
diffusion, or a moving/curved release surface.  It verifies the continuous
Gaussian VPM field after injection, but does not yet certify the cross-solver
velocity no-jump condition, divergence behaviour, topology preservation, or
the full cube.  The one-way contract is deliberately inapplicable where
sustained inward vorticity transport must be represented.

The next meaningful gates are a native FVM release-surface extractor followed
by a resolved three-dimensional manufactured vortex packet with velocity,
vorticity, divergence, stretching, diffusion, and no-jump checks on both sides
of the release surface.  The cube must remain blocked until those gates and the
persistent/replaced-state ownership gate are passed.
