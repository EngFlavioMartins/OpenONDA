# Saved results

A case writes its results below its `directory`, next to `setup.py` in the tutorials. With `variant` set, `solution/` and `samples/` gain a sub-directory of that name.

| Location | Contents |
| --- | --- |
| `solution/fvm.pvd` | FVM velocity, kinematic pressure, vorticity and eddy viscosity at saved times |
| `solution/vpm.pvd` | Particle positions, strengths, vorticity and core radii |
| `solution/vlm.pvd` | Lifting-surface geometry, circulation and loads, when present |
| `solution/fvm/`, `solution/vpm/`, `solution/vlm/` | The individual frames referenced by the `.pvd` files, and the numbered backups `fvm_{step:06d}.npz` and `vpm_{step:06d}.h5` |
| `solution/*_metadata.json` | The configuration that ran and its timing |
| `solution/*.log` | Solver logs (`fvm.log`, `vpm.log`, `coupler.log`) |
| `solution/coupler_diagnostics.jsonl` | Coupled cases: per coupling step the coupling-iteration residuals, vorticity-transfer conservation, particle counts and timings |
| `samples/` | Sampler output: CSV histories of forces, line probes and global invariants, VTK frames of plane probes |
| `figures/` | Plots written by `./allplot.sh` |
| `assets/<case_name>_mesh.npz` | The generated FVM mesh (with a `.vtu` copy), reused while its inputs are unchanged |

Open a root `.pvd` file in ParaView to load every saved time. Copy the whole `solution/` folder when moving results, so the relative frame paths stay valid. Backups for restarting are described in [continuation](continuation.md); field units are in the [FVM](fvm.md) and [VPM](vpm.md) guides.
