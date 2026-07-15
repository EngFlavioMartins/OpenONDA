# cubeFlow — hybrid FVM-VPM, native backend (no OpenFOAM)

Flow past a cube at Re = 1000, coupling the native Python FVM (near field)
with the VPM particle wake. System-agnostic port of
`tutorials/coupled_OFW_VPM/cubeFlow`: no case tree, no cfMesh, no MPI — the
mesh is built in memory with the **cube carved out of it**, exactly like the
OpenFOAM topology: no cells exist inside the body, and its exposed faces form
the no-slip `cube` wall patch where the boundary condition is applied.

## Run

```bash
./allrun.sh                       # clean, solve (t_end = 7.5 s), plot
HYBRID_T_END=2.0 ./allrun.sh      # ~4 min preview (wake reaches x ≈ 3D)
./allclean.sh                     # remove solution/ and figures/
./allplot.sh                      # (re)generate figures from an existing run
```

`allrun.sh` resolves the `OpenONDA-VPM` conda env automatically. To drive the
run directly instead: `python cube_setup.py` (a full run) or
`HYBRID_T_END=0.3 python cube_setup.py` (a 4-step smoke).

## Figures (`figures/`)

| Figure | Shows |
|--------|-------|
| `forces_history` | cube `Cd`, `Cl` and the pressure/viscous drag split vs `t U∞/D` |
| `vpm_diagnostics` | far-field kinetic energy, enstrophy, particle count |
| `hybrid_wake` | z=0 slice: sharp FVM near-field vorticity + VPM far-wake particles on one colour scale, plus the centreline velocity deficit |

## Output (`solution/`)

| File | Contents |
|------|----------|
| `coupler.log` | per-step donor BC, flux residual, fringe and injection diagnostics |
| `vpm_solution.log` | captured solver output (FVM + VPM; see note below) |
| `ibm_forces_history.csv` | time, Cd, Cl, marker slip for the cube (every FVM step) |
| `samples/flow_integrals.csv` | VPM global diagnostics (energy, enstrophy, particles) |
| `coupled_*_*.vtu` / `.pvd` | FVM snapshots (U, p, vorticity), in lock-step with VPM backups |
| `vpm_vpm_solution_*.h5` / `.xdmf` | VPM particle backups |

## Notes

- Resolution is `FVM_SPACING = VPM_SPACING = 0.0625`: a 48³ box minus the 16³
  cube hole = 106,496 cells. The spacing must put BOTH the box faces (±1.5)
  and the cube faces (±0.5) exactly on mesh planes (i.e. it must divide 0.5:
  0.1, 0.0625, 0.05, …) — the mesh generator refuses misaligned bodies.
  Refine `FVM_SPACING`/`VPM_SPACING` together; the per-step cost grows with
  cell and particle count (~15 s FVM per coupling step at this resolution).
- The body spec (`wall_patch_name="cube"` + `surface={"cube": ...}`) is the
  same one the OFW/cfMesh case consumes, so setups translate between the two
  backends unchanged.
- The FVM uses `bicgstab` momentum + AMG (pyamg) pressure; install the `fvm`
  extra (`pip install -e '.[fvm]'`) so pyamg is present, or it falls back to a
  slower Jacobi-CG pressure solve.
- `fvm.log`/`vpm.log` stay empty in coupled runs — the VPM's own logging
  captures process stdout, so all solver output lands in `vpm_solution.log`.
- Restart is not yet supported with `backend="fvm"`.
