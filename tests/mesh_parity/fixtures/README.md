# Native cfMesh regression fixtures

These compressed NumPy arrays contain native outputs, not fitted targets.
Load with `allow_pickle=False`. All captures use the provenance below.

| File | Coverage | SHA256 |
| --- | --- | --- |
| `cfmesh_template_vertices.npz` | All 1,188 raw template, projected, and assigned cylinder vertices, before output renumbering | `860e686096834949e4f15f746b1a23161d6f09d812a797cc86adb39f26dace18` |
| `cfmesh_wrapper_vertices.npz` | All 2,684 wrapper vertices from the independent native cylinder run | `5ed9d63747d358ac0b3f0808c5dad1fcef30adeafd8276244b5faa945fa5994b` |
| `cfmesh_surface_partition_passes.npz` | Five partition-point Laplacian/objective passes from identical input | `e1a967e45daa24b59d653b79f6eedc55817c5e0822493cc9064312cb15dfd889` |
| `cfmesh_volume_first_pass.npz` | First internal Knupp/untangler/volume pass | `486c7f809a2c2b06502d2332ab435c28e752261f33cd19c68104cf32ee3fd484` |
| `cfmesh_volume_second_pass.npz` | Next internal pass, including refreshed auxiliary centres | `53ecb564dff8fc91d8a8ff9d237ce1d2b3050e85d86fe74957247ab9cb3544e9` |
| `cfmesh_boundary_volume_pass.npz` | One unconstrained boundary-volume pass | `924678ede899a182a4b60b2adaffbbb763ed79e4358ca70f878cea222d93b01c` |

The template/projection/assignment fixture is an independent end-to-end native
run using the binary64 FTR oracle. The surface and volume fixtures intentionally
start both implementations from identical intermediate coordinates. They isolate
arithmetic and state handling; they are not independent full-mesh parity claims.

The boundary-volume test compares original mesh vertices only. Upstream
explicit OpenMP team sizing allows auxiliary-centre races despite
`OMP_NUM_THREADS=1`; those centres are discarded after this single pass.
Forcing `OMP_THREAD_LIMIT=1` is unsafe in this native build.

The full capture context and remaining acceptance gates are documented in
[`REPORT.md`](../../../docs/verification/cartesian_mesher/progress_2026-09-06/REPORT.md).

## Volume decomposition details

`cfmesh_volume_first_pass.npz` records one cfMesh internal untangling pass on
the coarse-cylinder mesh after surface smoothing and five cell-centre Laplacian
iterations. It contains the exact round-trip double coordinates, original face
and cell-face order, the 2,124-node / 7,740-tetrahedron decomposition, and native
positions after Knupp, geometric untangling, and ten volume-smoothing iterations.
Load with `allow_pickle=False`.

Provenance (2026-09-06):

- cfMesh source: `3ff8555514827646c34cacfe5f0f691e49cdbc96`;
- OpenFOAM-v2412, `_8dbc61e11c-20241220`, double precision, 32-bit labels;
- `OMP_NUM_THREADS=1`;
- library SHA256: `1463df86026888f4c216a1910c1c7fd276fbe3f19fb720fa4f631d1d505f94d8`;
- fixture SHA256: `486c7f809a2c2b06502d2332ab435c28e752261f33cd19c68104cf32ee3fd484`.

The standalone diagnostic driver reads an OpenFOAM mesh, restores the saved
cell-face order, and calls `optimizeMeshFV(5, 0, 1, 0)`. Diagnostic-only output
statements record `partTetMesh` before/after each smoother. They do not modify
mesh coordinates. The native library itself is unchanged. This deliberately
uses identical optimiser inputs on both sides, independently of upstream
surface-mapping differences. These local regressions are **not** proof of full
mesh parity.
