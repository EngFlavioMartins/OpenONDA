# cfMesh differential-parity oracle

This development-only tool compares OpenONDA's native Cartesian output against
the locally installed `cartesianMesh` executable. It is never imported by the
production mesher and does not make OpenONDA depend on OpenFOAM.

Run one explicit case in a fresh artefact directory:

```bash
python -m tools.mesh_parity.parity_report \
  tests/mesh_parity/cases/cube_aligned.json \
  --output /private/tmp/openonda-cube-parity
```

Set `CFMESH_CARTESIAN_MESH` or pass `--cfmesh-executable /absolute/path/to/cartesianMesh`
when the executable is not on `PATH`. Native OpenFOAM.app builds on macOS also
need their environment launcher; set `CFMESH_LAUNCHER` or pass its versioned
path with `--cfmesh-launcher`. The report records hashes for both files.

The oracle runs cfMesh with `OMP_NUM_THREADS=1` by default because parallel
surface projection is schedule-dependent at exact face/edge ties. The chosen
value is recorded in the report. Set `CFMESH_PARITY_OMP_NUM_THREADS` only when
deliberately investigating parallel behavior.

For example, with OpenFOAM-v2412.app:

```bash
python -m tools.mesh_parity.parity_report \
  tests/mesh_parity/cases/cube_aligned.json \
  --output /private/tmp/openonda-cube-parity \
  --cfmesh-executable "$HOME/OpenFOAM/$USER-v2412/platforms/darwin64ClangDPInt32Opt/bin/cartesianMesh" \
  --cfmesh-launcher /Applications/OpenFOAM-v2412.app/Contents/Resources/etc/openfoam
```

The runner writes:

- the generated cfMesh case and complete `cartesianMesh.log`;
- ASCII `constant/polyMesh` output for both implementations;
- `parity_report.json`, including executable hash/version help, source STL
  hashes, effective configuration, and OpenONDA Git commit;
- `parity_summary.txt` with the first failed gate.

The comparison first requires exact global topology invariants: cells, faces,
points, patch names/types/counts, face/cell valence histograms, cell-neighbour
histograms, and connected components. Only then does it construct a bounded
centroid/volume/topology-constrained cell mapping, verify every adjacency and
face incidence, and report geometry errors. It does not compare raw numbering
or use a whole-mesh graph-isomorphism search.

The geometry profile is selected by checkpoint. The current audited profiles
are:

| Checkpoint | Relative centroid | Relative volume | Boundary-normal angle |
| --- | ---: | ---: | ---: |
| template generation / surface topology | `1e-8` | `1e-8` | `1e-5` degrees |
| surface projection / patch assignment | `1e-3` | `1e-2` | `1.01` degrees |
| edge extraction | `5e-3` | `5e-2` | `0.65` degrees |
| boundary-layer generation | `5e-4` | `7e-2` | `0.65` degrees |
| mesh optimisation / boundary-layer refinement | `2e-5` | `2.5e-4` | `0.032` degrees |

cfMesh's iterative `surfaceOptimizer` stops at an objective tolerance of
`1e-3`; machine-epsilon octree perturbations can consequently choose equivalent
symmetric minima. The later profiles are measured envelopes for the checked-in
coarse curved-cylinder oracle, not permission to relax topology: global
invariants, adjacency, face incidence, and patch incidence remain exact at
every stage. Every report records the selected profile and all numeric
thresholds. A run without `--stop-after` uses the strict profile.

For supported cfMesh checkpoints, use:

```bash
python -m tools.mesh_parity.parity_report \
  tests/mesh_parity/cases/cube_aligned.json \
  --output /private/tmp/openonda-cube-ladder \
  --checkpoint-ladder
```

OpenONDA exposes each checkpoint in the ladder. Unsupported controls, including
patch refinement while that production feature remains unimplemented, yield an
explicit partial report instead of comparing an unrelated mesh or silently
dropping a control.

At the current clean-oracle checkpoint, the aligned cube passes through final
optimisation. The oblique cube retains exact final topology and passes through
boundary-layer generation, but its final maximum boundary-normal error is
`0.0323615` degrees against the `0.032`-degree gate. The coarse curved cylinder
also has exact final global invariants and passes through boundary-layer
generation; its final tight cell mapping matches 1,494 of 2,356 cells. These
are open parity items. Do not describe the Cartesian mesher as fully
cfMesh-parity complete until they and the remaining refinement, scale,
unseen-geometry, integrity, and flow gates pass.
