# GPU diffusion workspace repair

**Validated:** macOS/Metal, 2026-08-09.

## Failure

The cube tutorial requested Vulkan on macOS. Taichi selected the CPU, or GBD
dispatched over its full 21.3-million-node retained-domain allocation. Active
boxes also became invalid at domain faces and alternated Taichi ping fields.

## Repair

- `AUTO` selects Metal for macOS f32 runs without CPU fallback.
- Explicit Vulkan on macOS and Metal off macOS are rejected.
- The initialized Taichi architecture must equal the requested architecture.
- GPU DVH/GBD uses one fixed allocation and a lattice-aligned active extent.
- Active extents clamp independently at all six domain faces.
- Particles outside the retention domain do not enlarge the diffusion box.
- GBD starts and ends on the canonical ping field.

Regression tests cover backend selection, architecture fallback, six-face
clamping, out-of-domain particles, active/full-grid equivalence, and ping-state
stability.

The production cube configuration and measured force/runtime results are in
[`docs/fvm_vpm_coupling.md`](../fvm_vpm_coupling.md).
