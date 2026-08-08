# GBD/DVH on Vulkan: bounded active workspace

**Status:** fixed 2026-08-08. Pre-existing defect (reproduced on `7d7f135`, before the VPM audit).

## Symptom

```
Performing GBD diffusion(h=4.000e-02, ...)
RuntimeError: [runtime.cpp:launch_kernel@572] Dispatch error : RhiResult(not_supported)
```

`tutorials/coupled_FVM_VPM/cubeFlow`, step 2 — the first step carrying VPM particles.

## Root cause

Two independent whole-allocation operations, both driven by the same design conflation.

**1. Active grid = whole domain.** When the domain was pre-configured (the Vulkan path always
is), GBD/DVH set the active grid to the entire `vpm_domain_bounds` every firing, regardless of
where particles were:

| | |
|---|---|
| `VPM_DOMAIN` / `VPM_SPACING` | (-4.5, 11.0, -4.5, 4.5, -4.5, 4.5) / 0.04 |
| active grid | 395 × 232 × 232 = **21,260,480 nodes** |
| cloud actually needs | ~180,000 nodes |

Taichi flattens n-D loops to a 1-D dispatch, so that is ~166k workgroups against
`maxComputeWorkGroupCount[0] = 65,535` (Intel Iris Xe ADL GT2). Taichi reports the rejection
only as `RhiResult(not_supported)`.

**2. Whole-field host transfers.** `_nu_eff_grid.from_numpy`, `_current_grid.from_numpy` and two
`_current_grid.to_numpy()[:nx,:ny,:nz,:]` each launch a copy kernel over the full field — the
slice happens *after* the transfer — hitting the same ceiling and moving ~255 MB per firing.

Isolated confirmation: struct-for or `fill()` over a 19.8M-node field reproduces the error
exactly; `ti.ndrange(60,60,60)` is fine.

## Fix

**Lattice-aligned active box.** `_lattice_aligned_bounds` sizes the active box to the particle
cloud while keeping the origin on the pre-configured lattice — offset from `_fixed_grid_min` by
a whole number of cells — so every node keeps the coordinate it would have had on the full grid.
That preserves the fixed grid phase the full-domain path existed to provide, and with it the
asymmetric flat-end artefact it avoided. Clamped to stay inside the allocation.

Verified: lattice phase error ≤ 3e-6 cells, node coordinates match the full-domain lattice to
≤ 1.0e-7 (f32 eps), box always covers the cloud and fits the allocation.

**Bounded host transfers.** Three chunked kernels (vec3 down, vec3 up, scalar up) map a flat
active index to `(i,j,k)` and move `_GRID_TRANSFER_CHUNK = 65536` nodes per launch through a
fixed-shape staging buffer. Fixed shape matters: varying ndarray shapes accumulate Vulkan
staging allocations on Taichi 1.7.x, which is why `physics/base.py` already uses this pattern.
Transfer and dispatch cost is now O(nx·ny·nz), never O(prod(_grid_shape)).

Grid kernels (`_zero_grid_kernel`, body-mask fill/apply, grid norm) likewise take the active
extents rather than iterating the allocation and filtering.

**Device-pool visibility.** `initialize_taichi_backend` now publishes the pool it actually
chose as `constants.TAICHI_POOL_BYTES`, and the diffusion module warns when the grid crowds it:

```
Diffusion grid 395x232x232 uses 649 MB, 85% of the 768 MB device pool;
particles, treecode and staging buffers share what is left.
```

This is deliberately a **warning, not a rejection**. An earlier revision of this fix made it a
hard `MemoryError` at 45% of the pool; that rejected cubeFlow's 649 MB grid — a configuration
that demonstrably runs — and on 4 MPI ranks it surfaced as a silent hang, because one rank
raised while the others spun in a barrier at 100% CPU. The 1 GiB `_MAX_PREALLOC_BYTES` ceiling
remains the allocation authority. Do not turn the pool share into an enforced limit without
first measuring the real ceiling on the target device.

## Results

| | before | after |
|---|---|---|
| active grid (cubeFlow) | 21,260,480 nodes | **~180,000 nodes** |
| whole-allocation transfers per firing | 4 (~255 MB) | **0** |
| cubeFlow step 2 | `RhiResult(not_supported)` | **passes** |

`tests/vpm/test_grid_diffusion_active_box.py` — 9 tests: lattice phase, cloud coverage,
allocation containment, transfer round-trips (including a multi-chunk case), a guard that no
whole-allocation transfer returns to the module, GBD active-box vs full-domain numerical
equivalence, and a Vulkan large-allocation/small-box dispatch regression.

**Numerical neutrality is pinned**, not assumed: `test_active_box_gbd_matches_full_domain_gbd`
runs the same particles through GBD twice — once with the active box, once with the old
full-domain box forced — and requires identical particle count, and positions, circulations,
radii, group IDs and total circulation equal to f32 tolerance.

51 diffusion/GBD/DVH tests pass.

## Not addressed

- Treecode capacity still allocates to the declared particle ceiling (`max_particles=300_000`
  for cubeFlow) to avoid Taichi field replacement on Vulkan. Deliberately left for a separate
  measurement of the whole device-memory distribution.
- The device-pool warning does not account for the treecode and particle fields in the same
  budget, so it under-reports total pressure. Measuring the real allocation ceiling on the
  target device is the prerequisite for turning it into an enforced limit.
