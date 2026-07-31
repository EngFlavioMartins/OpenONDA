# FVM–VPM coupling reference

## Run

```bash
cd tutorials/coupled_FVM_VPM/cubeFlow
./allrun.sh
```

Run `referenceFlow/allrun.sh` before `allplot.sh` when a new matched reference
is required. Both cases use a 0.0125 s FVM step and write every 0.5 s.

## Production method

The supported coupler uses one exchange path:

1. Advance the Gaussian-particle VPM solution on Metal in `f32`.
2. Evaluate the complete particle velocity, including the body harmonic field,
   on the cropped FVM boundary.
3. Project the face velocity to zero net volume flux and impose it as a
   Dirichlet velocity trace. The FVM pressure boundary uses
   `fixedFluxPressure`.
4. Relax the FVM fringe toward the same time-interpolated VPM velocity used by
   the boundary trace.
5. Reconstruct the FVM velocity on the handoff lattice with a cached,
   four-neighbour, gradient-corrected weighted trace.
6. Transfer its curl to Gaussian particles, reuse lattice-aligned particles,
   apply one strength correction, exclude the solid exactly, and prune by
   circulation magnitude while preserving circulation and linear impulse.

The cube benchmark uses a `[-1.8, 1.8]^3` FVM box, 0.05 m particle spacing,
0.0125 s FVM steps, 0.05 s VPM steps, a 300,000-particle cap, four CPU ranks
for FVM, and Metal/f32 for VPM. The explicit Metal request is strict: backend
initialization fails instead of silently switching the particle solve to CPU.

## Validation result

The accepted configuration was selected from matched runs through 3 s. All
variants used the same 448,000-cell cropped mesh, harmonic donor,
`fixedFluxPressure`, timestep, and 1,036,000-cell reference solution.

| Method | Drag error | Mean lift | Velocity relative L2 | Median window |
|---|---:|---:|---:|---:|
| Original baseline | +4.25% | -0.00028 | 2.558% | — |
| Aligned remesh and interface-ranked cap | +5.14% | -0.00010 | 2.419% | 68.2 s |
| Weighted trace and interface-ranked cap | +4.48% | +0.00423 | 2.175% | 68.2 s |
| Weighted trace without fringe | -5.98% | -0.04248 | 3.163% | 75.6 s |
| Production method | **+4.07%** | +0.00383 | **2.158%** | **57.3 s** |

At 3 s, the stitched velocity field has 1.1% mean and 3.4% P95 error relative
to the fully meshed reference. Aligned remeshing was 13 times faster than
unconditionally scattering an already aligned cloud and conserved circulation
to roundoff.

## Rejected methods

These methods were tested and are intentionally unsupported:

- Disabling the fringe caused transverse drift and degraded both velocity and
  force histories.
- Ranking the particle cap by an interface-velocity bound reduced that bound
  but worsened drag.
- Nearest-cell velocity traces increased cell-switching noise.
- Full M4 remeshing of aligned particles added cost without improving the
  solution.
- Cell-vorticity handoff was less consistent than the velocity-gradient trace.
- Live FVM-vorticity Biot–Savart reconstruction omitted the harmonic velocity
  component and introduced a kernel splice at the interface.
- Mixed, characteristic, and scalar Robin boundary variants did not recover
  the reference force history.
- A VPM-derived pressure gradient was incompatible with the prescribed FVM
  flux and became unstable. `fixedFluxPressure` is required.
- Overlap velocity overriding and redundant frozen-donor Picard iterations
  added cost without improving the accepted result.
- Applying the body panel field throughout the particle domain was expensive;
  applying a separate Dirichlet panel correction double-counted blockage.

## Remaining limitation

The remaining approximately 4% drag bias is a pressure/traction-consistency
error. It is larger than `f32` roundoff and is not corrected by more remeshing,
stronger pruning, disabling the fringe, or changing the particle kernel. A
future improvement should exchange a conservative momentum flux or traction
at the interface and demonstrate convergence against a matched monolithic
split test.

## Diagnostics

Every coupling window writes VPM, donor, fringe, FVM, handoff, and total wall
times to `solution/coupler.log` and `solution/coupler_diagnostics.jsonl`. Force
histories are written by the FVM solver and include pressure and viscous wall
contributions. Comparison plots must use coincident physical output times.
