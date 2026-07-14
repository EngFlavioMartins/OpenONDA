# cubeFlow — hybrid FVM-VPM, native backend (no OpenFOAM)

Flow past a cube at Re = 1000, coupling the native Python FVM (near field,
immersed-boundary cube) with the VPM particle wake. System-agnostic port of
`tutorials/coupled_OFW_VPM/cubeFlow`: no case tree or cfMesh is required.
The tutorial uses the default serial execution configuration.

## Run

```bash
python cube_setup.py                    # full run (t_end = 7.5 s)
HYBRID_T_END=0.3 python cube_setup.py   # quick smoke (4 coupling steps)
```

## Output (all under `solution/`)

| File | Contents |
|------|----------|
| `coupler.log` | per-step donor BC, flux-residual, fringe and injection diagnostics |
| `ibm_forces_history.csv` | time, Cd, Cl, marker slip for the cube (every FVM step) |
| `fvm.log`, `vpm.log` | captured per-solver output |
| `*.vtu` / `.pvd` | FVM snapshots (U, p, vorticity), written in lock-step with VPM backups |
| `vpm_solution_*.h5` | VPM particle backups |

## Notes

- Default resolution (h = 0.1, 30³ cells) is chosen for the serial Python
  FVM; the OFW tutorial runs h = 0.04. Tighten `FVM_SPACING`/`VPM_SPACING`
  together — the box extent must stay an integer number of cells.
- The cube is a direct-forcing immersed body (`box_surface_markers` +
  `ImmersedBody.from_points`); drag/lift come from the IBM force integral,
  not a wall-patch integration.
- Restart is not yet supported with `backend="fvm"`.
