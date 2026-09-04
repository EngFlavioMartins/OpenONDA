# Cylinder grid-independence study

The reference case uses `openonda.fvm.mesher.CartesianMesher` directly. It
does not require Gmsh or another mesh backend. The cylinder is recovered from
`../assets/cylinder_long.stl`, and its no-slip wall is resolved with
isotropically small, body-fitted Cartesian cut cells. Explicit anisotropic
boundary layers are intentionally disabled: the wall spacing itself is the
grid-study parameter.

Run the complete study from `reference_flow/` with:

```bash
./allrun.sh
```

The four wall spacings are D/12, D/24, D/36, and D/54. The three production
grids use a constant refinement ratio of 1.5. The background, wake, near-body,
and wall sizes all scale with the requested `dx`; consequently, the dyadic
Cartesian octree cannot collapse two requested study grids onto the same
effective resolution.

Each generated solver mesh is stored automatically in two forms:

- `solution/<case>/mesh.vtu` is the ParaView-readable mesh, including cell
  volume, Cartesian size/level, and boundary-layer index arrays.
- `solution/<case>/mesh.npz` is the lossless OpenONDA-native copy that can be
  passed back to `create_fvm_solver(mesh=...)` without regenerating the grid.

Flow fields remain in the configured VTK time series.

After all four cases finish, `assets/postprocess.py` writes
`solution/grid_study.json` with the common-window force statistics,
Richardson extrapolation, and fine-grid GCI. `assets/plot_grid_study.py` writes
the comparison figures under `figures/`. Use `./allplot.sh` to rebuild only
the report and figures from existing samples.
