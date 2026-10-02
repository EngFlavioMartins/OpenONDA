# Saved simulation fields

A case writes its field time series beneath `solution/`:

| File | Meaning |
| --- | --- |
| `fvm.pvd` | Mesh velocity and pressure; open in ParaView. |
| `vpm.pvd` | Particle positions and fields; open in ParaView. |
| `vlm.pvd` | Lifting-surface geometry and loading, when present. |
| `fvm/mesh.npz`, `fvm/mesh.vtu` | Native FVM mesh for Python and ParaView. |
| `fvm/fvm_*.vtu` or `*.pvtu` | Saved FVM field frames. |
| `vpm/vpm_*.h5` | Particle state and standalone VPM/VLM restart data. |
| `vpm/vpm_*.vtu`, `vlm/vlm_*.vtp` | Particle and surface visualization frames. |
| `fvm_metadata.json`, `vpm_metadata.json`, `run_metadata.json` | Executed configuration and saved run timing; present for the relevant solver. |

Open the root `.pvd` collection to load all saved times. Copy the entire `solution/` directory when moving results so its relative frame paths remain valid. Tutorial plots also read case-specific histories in `samples/` or `solution/`.

Use the solver guides for [FVM field units](fvm.md), [particle strength and vorticity](vpm.md), and [coupled fields](coupling.md). See [visualization](visualization.md) for comparisons and [continuation](continuation.md) for numerical backups.
