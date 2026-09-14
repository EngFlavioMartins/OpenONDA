# Native Cartesian mesher performance

Measured on 2026-09-14 with the unchanged `FVM_MESH` from
`tutorials/coupled_fvm_vpm/02_cube_flow/setup.py`: 303,264 cells, 924,588 faces,
and 318,192 points. Environment: macOS arm64, Python 3.11.15, NumPy 2.4.6,
Numba 0.66.0.

| Run | Numba threads | Full build time |
| --- | ---: | ---: |
| Original source, first compilation in an isolated source copy | 4 | 569.01 s |
| Updated source, fresh compilation cache | 4 | 136.65 s |
| Updated source, compiled kernels already cached | 1 | 116.61 s |

The first-compilation comparison is 4.16× faster.
The one-thread setting matches the cube tutorial's MPI-root meshing runtime.
Every row constructs the entire mesh and performs final validation; none loads
a cached mesh. Timings exclude imports, solver startup, and mesh-file export.
These are individual local measurements, not a statistical benchmark.

| Stage | Original (s) | Updated, first compilation (s) |
| --- | ---: | ---: |
| Template generation | 50.97 | 25.33 |
| Patch assignment | 48.45 | 10.77 |
| Patch point remapping | 43.07 | 5.48 |
| Surface optimisation | 133.30 | 16.60 |
| Wrapper layer generation | 19.07 | 8.13 |
| Mesh optimisation | 183.51 | 23.49 |
| Wall constraint | 52.65 | 30.65 |
| Final mesh validation | 17.42 | 2.82 |

## What changed

- Surface stencils are packed once and reused across all five smoothing passes.
  Scalar Newton arithmetic avoids allocating small arrays for each triangle.
- Triangle/box overlap, hanging-node expansion, interior averaging, wrapper
  reference geometry, and cell-edge validation run in cached Numba kernels.
- VTK intersection validation reuses connectivity while updating and checking
  each trial's coordinates. Startup reports stages through the configured logger.

The mesher retains its refinement requests, cell ordering, floating-point
operation order, five-pass smoothers, wall constraints, and quality checks.
All three runs have identical SHA-256 hashes for vertices, faces, owners, and
neighbours, and exactly equal wall-constraint diagnostics and quality reports.
Raw measurements are in [mesher_performance.json](mesher_performance.json).

## Reproduce

From the repository root, in the OpenONDA Python environment:

```sh
NUMBA_CACHE_DIR=/tmp/openonda-mesher-jit-fresh python scripts/benchmark_fvm_mesher.py --threads 4 --output /tmp/openonda-mesher-cold
NUMBA_CACHE_DIR=/tmp/openonda-mesher-jit-fresh python scripts/benchmark_fvm_mesher.py --threads 1 --output /tmp/openonda-mesher-warm
```

Use a previously nonexistent compilation-cache directory for the first command.
The benchmark bypasses `CachedMesh` and leaves tutorial solution/cache files
alone. `--save-mesh` optionally writes the validated native mesh in its output
directory. Ordinary tutorial launches continue using their existing cache.

## Verification

83 tests passed across native cfMesh numerical fixtures, octree contracts,
geometry and topology checks, VTK coordinate updates, startup logging, and mesh
caching. The acceptance cases for a rotated box, ellipsoid, torus, and finite
wing pass. The two-disjoint-body acceptance case fails with
`Box wall face has no unambiguous Cartesian surface plane`; the same failure
was reproduced against the untouched original source in an isolated directory.
It remains a pre-existing geometry limitation.

The triangle/box predicate also matched the original implementation for
100,000 randomized and degenerate triangles. OpenONDA builds as a
`py3-none-any` wheel, and importing and running compiled topology validation
from the extracted wheel succeeded. No dependency or installation toolchain
was added; runtime compilation uses the existing Numba dependency.
