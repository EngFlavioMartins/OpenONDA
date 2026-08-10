# FVM–VPM coupling

## Cube benchmark

```bash
cd tutorials/coupled_FVM_VPM/cubeFlow
./allrun.sh
./allplot.sh
```

Force and field samples are written to `samples/`. Comparisons use coincident
physical times from `referenceFlow/samples/`.

## Production method

Each 0.05 s coupling window:

1. advances Gaussian particles with RK2, stretching, LES, and GBD on the GPU;
2. evaluates the particle and panel velocity on the cropped FVM boundary;
3. removes the integrated donor-flux residual;
4. advances the FVM with Dirichlet velocity and `fixedFluxPressure`;
5. relaxes the FVM fringe toward the same VPM field;
6. transfers the FVM velocity curl to particles with conservative remeshing,
   solid exclusion, invariant recovery, and weak-particle pruning.

The validated cube setup is:

| Setting | Value |
|---|---:|
| FVM box | `[-1.5, 1.5]^3` |
| FVM cells | 120,248 |
| Wall / maximum cell size | `0.0125 D` / `0.1 D` |
| FVM step / ranks | `0.01 s` / 4 |
| VPM step / spacing | `0.05 s` / `0.04 D` |
| Handshake width / dead strip | `0.24 D` / 0 |
| Particle cap | 1,500,000 |
| VPM backend | Metal, f32 |

`AUTO` selects an available Taichi backend. Set `OPENONDA_PROCESSING_UNIT=CPU`
for a portable diagnostic run or select a supported GPU backend explicitly.

## Validation

The accepted setup was run from rest to 3 s against the saved fully meshed
reference.

| Time | Reference Cd | Hybrid Cd | Error |
|---:|---:|---:|---:|
| 0.15 | 1.95280 | 1.96910 | +0.83% |
| 0.60 | 1.38213 | 1.39923 | +1.24% |
| 1.20 | 1.20312 | 1.15007 | -4.41% |
| 1.65 | 1.02008 | 1.01098 | -0.89% |
| 2.10 | 0.87281 | 0.93170 | +6.75% |
| 2.70 | 0.87694 | 0.86942 | -0.86% |
| 3.00 | 0.88333 | 0.84641 | -4.18% |

Over all 20 coincident samples, mean Cd is 1.13599 versus 1.13404 (+0.17%).
The pointwise mean absolute error is 3.30%; the maximum is 8.12%. The remaining
error is primarily a wake-phase/interface-trace error, not a mean-drag or wall
force-integration error.

After JIT warm-up, one window takes 21–22 s: about 12 s VPM, 5.8 s four-rank
FVM, and 3.8 s handoff. The particle population remains at or below 200,000.

## VPM overhaul repair

The post-overhaul failure had three causes:

- Vulkan was requested on macOS and Taichi silently selected the CPU;
- GPU grid diffusion dispatched over the complete retained-domain allocation;
- the active diffusion box could move past a domain face and alternated its
  ping field, causing repeated compilation and invalid extents.

The repaired path selects Metal, verifies the initialized Taichi architecture,
allocates the GPU diffusion workspace once, executes only on a clamped active
box, ignores already-removed out-of-domain particles, and restores a canonical
ping state after every GBD step.

## Rejected variants

- A 0.4 D handshake improved startup drag but increased the error after 1 s.
- A 0.16 D handshake was worse after the wake reached the interface.
- Applying the panel velocity in every particle RK stage was slower and did not
  improve drag.
- A wall-pressure variant reduced Cd to 0.54 at 0.15 s and is not used.
- Matching the full reference refinement inside the cropped box required
  1.02 million cells and removed the cost advantage.
- A global circulation-magnitude particle cap worsened mature-wake drag because
  weak far-wake particles still carry important impulse and induced velocity.
- Exterior-halo overwrite generated 387,000 candidates and forced destructive
  cap pruning.

The supported path remains the single configuration above. Improving the
pointwise transient further requires a conservative momentum/traction trace or
a true FVM handshake shell, not another scalar boundary or pruning coefficient.

## Native cylinder and NACA workflows

`cylinderSheddingFlow` and `naca4412Flow` use solver-native Cartesian meshes
with direct-forcing immersed bodies. Their VPM coupling windows are subcycled
by the FVM so the immersed-boundary solve respects its CFL constraint. Each
case runs `assets/check_run.py` automatically: short runs gate finite fields,
linear convergence, CFL, continuity, no-slip error, donor-flux closure, and
handoff conservation; completed production horizons additionally gate wake or
load statistics.
