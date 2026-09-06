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

The preflight wall spacing is D/12. The three production grids request D/40,
D/80, and D/160. Their resolved background and wall sizes are dyadic and use a
constant refinement ratio of 2. The background, wake, near-body, and wall sizes
all scale with the requested `dx`; the postprocessor records the realized
sizes and rejects duplicate effective grids before computing GCI.

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
