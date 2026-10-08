# Saved results

A case writes its results next to its `setup.py`:

| Location | Contents |
| --- | --- |
| `solution/fvm.pvd` | FVM velocity, kinematic pressure, vorticity and eddy viscosity at saved times |
| `solution/vpm.pvd` | Particle positions, strengths, vorticity and core radii |
| `solution/vlm.pvd` | Lifting-surface geometry, circulation and loads, when present |
| `solution/fvm/`, `solution/vpm/`, `solution/vlm/` | The individual frames referenced by the `.pvd` files |
| `solution/*_metadata.json` | The configuration that ran and its timing |
| `solution/*.log` | Solver logs |
| `solution/coupler_diagnostics.jsonl` | Coupled cases: per-exchange boundary convergence, particle counts and timings |
| `samples/` | Sampler histories as CSV: forces, line and plane probes, global invariants |
| `figures/` | Plots written by `./allplot.sh` |
| `assets/<case_name>_mesh.npz` | The generated FVM mesh (with a `.vtu` copy), reused while its inputs are unchanged |

Open a root `.pvd` file in ParaView to load every saved time. Copy the whole `solution/` folder when moving results, so the relative frame paths stay valid. Backups for restarting are described in [continuation](continuation.md); field units are in the [FVM](fvm.md) and [VPM](vpm.md) guides.
