# GBD/DVH on Vulkan: fixed full-domain grid exceeds the dispatch limit

**Status:** root cause proven, **NOT yet fixed**. Pre-existing (reproduced on `7d7f135`,
before the VPM audit).

## Symptom

```
Performing GBD diffusion(h=4.000e-02, nu=1.000e-03, threshold=3.00e-01, LES nu_eff/nu max=2.02).
RuntimeError: [runtime.cpp:launch_kernel@572] Dispatch error : RhiResult(not_supported)
```

`tutorials/coupled_FVM_VPM/cubeFlow`, step 2 — the first step with VPM particles.

## Root cause

`_gbd_diffusion_impl` (and the DVH equivalent) takes this branch when the domain is
pre-configured, which the Vulkan path always does:

```python
if self._fixed_grid_min is not None and self._max_grid_dims is not None:
    grid_min_np = self._fixed_grid_min.copy()
    nx, ny, nz = self._ensure_grid_capacity(*self._max_grid_dims)
```

So the **active** grid is the entire declared `vpm_domain_bounds`, every firing,
regardless of where the particles actually are. Measured for cubeFlow:

| | |
|---|---|
| `VPM_DOMAIN` | (-4.5, 11.0, -4.5, 4.5, -4.5, 4.5) |
| `VPM_SPACING` | 0.04 |
| active grid | 395 × 232 × 232 = **21,260,480 nodes** |
| particle cloud bbox | ~3.0 × 1.8 × 1.8 m → would be ~170k nodes |

Taichi flattens an n-D loop to a 1-D dispatch, so 21.3M elements at `block_dim=128` is
~166k workgroups against `maxComputeWorkGroupCount[0] = 65,535` on this device
(Intel Iris Xe ADL GT2). Taichi reports the rejection only as `RhiResult(not_supported)`.

Confirmed in isolation:

| dispatch | result |
|---|---|
| struct-for / `fill()` over the full 19.8M–21.3M field | `RhiResult(not_supported)` |
| `ti.ndrange(60, 60, 60)` | OK |

The fixed-grid branch exists for two good reasons: a fixed origin avoids the asymmetric
flat-end artefact, and a fixed allocation avoids Taichi 1.7.x retaining replaced Vulkan
fields. Neither reason requires *dispatching* over the whole domain.

## Applied (correct, but the case still fails)

**Fix A landed.** Both fixed-grid sites (GBD and DVH) now call `_lattice_aligned_bounds`,
which keeps the origin on the pre-configured lattice — offset from `_fixed_grid_min` by a
whole number of cells, so every node keeps the position it would have had on the full
grid — and shrinks the extent to the occupied region, clamped inside the allocation.

Measured on the cubeFlow cloud: active grid **21,260,480 -> 179,712 nodes (118x smaller)**.
39 GBD/DVH/diffusion tests pass.

## Still blocking: full-field host transfers

The case still raises `RhiResult(not_supported)`, now from the remaining transfers that
touch the **whole allocation** rather than the active box:

| line | call |
|---|---|
| 1083 | `self._nu_eff_grid.from_numpy(nu_eff_grid_np)` |
| 1125 | `self._current_grid.to_numpy()[:nx, :ny, :nz, :]` |
| 1373 | `self._current_grid.from_numpy(buf)` |
| 1456 | `self._current_grid.to_numpy()[:nx, :ny, :nz, :]` |

`to_numpy()`/`from_numpy()` launch a copy kernel over the entire field, so they hit the
same 65,535-workgroup ceiling — and move ~255 MB per firing where the active box needs
~2 MB. Note the slicing happens *after* the transfer, so the full grid is moved either way.

**Fix:** copy the active sub-box into a staging field (or ndarray) sized to the box with a
bounded kernel, and transfer only that. Same pattern as `_zero_grid_kernel`. This is the
last piece; it was not attempted here.

## Earlier work (necessary, not sufficient)

Grid kernels no longer iterate the whole allocation and filter; they take the active
extents directly (`ti.ndrange(nx, ny, nz)`), and the three `fill(0.0)` calls became a
bounded `_zero_grid_kernel`. Correct and faster, but it does not help here because the
active box *is* the full domain. 37 GBD/DVH tests pass.

## The two candidate fixes

**A — active box tracks the cloud, allocation stays fixed (preferred).**
Keep the fixed allocation and the fixed lattice, but set `(nx, ny, nz)` from the particle
bbox snapped to that lattice, as `configure_grid_lattice_anchor` already supports. The
grid origin stays on the same lattice, so the flat-end artefact is still avoided; no
reallocation occurs while the box fits the allocation. Dispatches drop from 21.3M to
~170k nodes here — and the diffusion also stops touching ~21M empty nodes per firing.

**B — chunk the dispatch.** Give each grid kernel a `k0` offset and launch it in k-slabs
sized so `nx·ny·kc` stays under the device limit. Backend-agnostic and preserves current
behaviour exactly, but touches ~6 kernel signatures and leaves the wasted work in place.

A is better physics-per-watt and is closer to what the non-fixed branch already does.

## Standing memory constraint

Independently of dispatch: 21.3M nodes × 12 B × 2 ping-pong vec3 grids ≈ **510 MB**, plus
an 85 MB i32 body mask, against the **768 MB** pool `config/backend.py` assigns to
integrated GPUs. Fix A removes this too (the allocation can then follow the cloud);
fix B does not.

If it recurs, the config levers are: shrink `VPM_DOMAIN` to the region the wake occupies,
coarsen `VPM_SPACING`, or run the VPM on CUDA/CPU.

A configuration-time preflight comparing `nx·ny·nz` against the device
`maxComputeWorkGroupCount` and pool size — failing with an actionable message rather than
`RhiResult(not_supported)` — is worth adding either way.
