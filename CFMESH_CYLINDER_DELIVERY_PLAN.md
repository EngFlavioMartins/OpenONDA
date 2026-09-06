# Deliver the reference-flow cylinder with conventional cfMesh meshing

Date: 6 September 2026

Status: implementation instructions, not a claim that the cylinder works.

## 1. Deliverable and change of direction

**Produce one complete, wall-fitted, hex-dominant cylinder mesh in
`tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow`, using the
built-in Python implementation of the conventional cfMesh Cartesian workflow.**

The immediate target is the current **coarse case, D/40**. It must have the same
geometry, refinement topology, wrapper construction and optimisation behaviour
as a pinned native cfMesh run of that case. It must be suitable for the existing
FVM solver and have inspected wall/transition images. A background template,
passing small fixtures, or an unfinished optimisation run is not delivery.

This is the cylinder-specific execution plan. It takes precedence over the
earlier broad plan's permission to accept deliberate algorithmic differences
for this milestone. The earlier plan's applicable safety/quality requirements
remain in force. Do not broaden into arbitrary-shape certification, a GUI, new
inflation controls, or a full flow/grid-independence campaign before this
cylinder is delivered.

The implementation remains built into OpenONDA. Native cfMesh is the independent
development oracle, **not a hidden runtime backend**. Do not substitute an O-grid,
an extruded 2D cylinder mesh, an analytic-circle projection, or an imported native
mesh for the Python-generated deliverable.

## 2. What is actually missing now

The code can be translated. The obstacle is not the language; it is that the
current path is not yet an independently verified, behaviour-preserving port of
the complete workflow on the actual reference-flow input.

| Item | Current evidence | Required correction |
| --- | --- | --- |
| Default dispatch | Ordinary `build()` now calls `_run_cfmesh_workflow("meshOptimisation")`. | Preserve this repair; do not maintain a second production algorithm. |
| Topology and wall validation | Three saved small cases now pass independent edge and unmasked owner-pyramid checks. | Keep these checks mandatory; this does not certify the large cylinder. |
| Surface topology preparation | The workflow assigns `surface_topology_changes = 0`; no corresponding native four-operation repair sequence is wired into this stage. | Translate the actual repair predicates and fixed-point loop, or demonstrate that each native operation is a no-op on the identical cylinder input. A hardcoded zero is not a demonstration. |
| Large-mesh surface optimisation | Above 200,000 faces, the workflow passes zero surface iterations. | Remove the size-dependent algorithm change. Accelerate the native operations instead of omitting them. |
| Large-mesh volume optimisation | The same threshold reduces the outer Laplacian pass count from five to one. | Use the pinned native call parameters and stopping rules at every size. |
| Final wall treatment | Public `build()` applies an additional nearest-STL projection after native-style optimisation; checkpoint output does not include it. | Measure and isolate this difference. Match native constraints in the workflow; do not present an extra post-snap as default-output parity. |
| Reference input | Historical small cylinder parity cases use a different domain that encloses the whole long cylinder. `reference_flow` cuts through the cylinder at its span planes. | Build one equivalent, frozen fluid-boundary representation for both implementations. |
| End-to-end evidence | The repaired coarse output directory is empty. A 443,714-cell template was reported, not a completed mesh. | Save and validate the full public-build output, with stage timings and images. |
| Reproducibility | Surviving fixtures cover useful small stages, but their documented `tools.mesh_parity` runner is absent from the working tree. | Provide a maintained cylinder oracle/comparison runner and keep its inputs and evidence in the repository, not only temporary directories. |

### Important correction about cfMesh's default layers

Native `cartesianMeshGenerator` always invokes `addLayerForAllPatches()` for its
default wrapper. This is not the same as user-requested multi-layer inflation.

Native also calls `optimizeBoundaryLayer()`, but the inspected implementation
returns without work unless `boundaryLayers.optimiseLayer` enables it. With no
`boundaryLayers` dictionary, skipping that specialised optimiser is correct.
Do not add it merely because the generator contains the call. Translate the
**executed branch**, including its guard. The active surface and finite-volume
optimisers must still run.

### What “castellation” means here

Use cfMesh's dyadic refinement and conforming split/polyhedral transitions.
Fine subfaces and adjacent coarse-face perimeters must share all required
edge-midpoint vertices. There is no requirement to invent an arithmetic
intermediate cell size between every pair of octree levels. The native reference
will determine the actual transition geometry and near-wall relaxation.

## 3. Freeze one cylinder case before further algorithm changes

Use the existing coarse configuration, with no silent domain/resolution changes:

| Parameter | Frozen target |
| --- | --- |
| Diameter | `D = 1` |
| Original STL | `tutorials/coupled_fvm_vpm/cylinder_shedding_flow/assets/cylinder_long.stl` |
| STL SHA256 | `ee241f6c06c1723eeebf44d6e5f3eaf5b4c7f372b23d357bc4a588b095b74cc3` |
| Domain | `(-8, 20, -8, 8, -0.6, 0.6)` |
| Outer patches | `inlet`, `outlet`, `ymin`, `ymax`, `zmin`, `zmax` |
| Object patch | `cylinder`, type `wall` |
| Requested cylinder size | `0.025` |
| Requested background/outer-boundary size | `0.2` |
| Near-body box | `(-2, 6, -2, 2, -0.6, 0.6)` |
| Near-body request | `0.05 * (1 + 1e-12)`, exactly as current setup |
| Wake box | `(-4, 12, -4, 4, -0.6, 0.6)` |
| Wake request | `0.1 * (1 + 1e-12)`, exactly as current setup |
| Global minimum-size override | None |
| Explicit inflation dictionary | None; retain the ordinary native wrapper |
| Anisotropic scaling | None |

The small epsilon in the existing object requests deals with cfMesh's strict
object-size conversion. It must be visible and identical in the native
dictionary and Python inputs, not applied on one side only. Record requested
sizes, resolved levels, root bounds and root size separately.

Generate both configurations from one canonical case description. Do not maintain
an independently hand-tuned native dictionary. A D/12 debug case may use the same
geometry/domain and proportionally scaled controls, but it cannot replace D/40
acceptance. Do not use the old enlarged-span cylinder as the acceptance oracle.

### The cylinder/box intersection is an input gate

For native cfMesh, construct the oriented, closed boundary of **box minus the
cylinder solid** from the supplied STL:

1. Clip the original side triangles at the two span planes without changing the
   source surface or rounding coordinates to binary32.
2. Keep the fluid portions of the span planes, including the cylinder holes.
   Weld their perimeter to the clipped cylinder-wall vertices.
3. Do not add a circular disk over either hole; that would close the flow-domain
   tunnel. Discard the original distant cylinder end caps from this fluid-boundary
   representation.
4. Retain the other four box faces, named patches, consistent fluid-outward
   orientation, feature edges and triangle ordering/provenance.
5. Check manifold edge incidence, patch coverage, bounding box and enclosed fluid
   volume before meshing.

Use a double-preserving native surface format; reuse the previous FTR precision
work after verifying the reader/writer round trip. Hash the original STL and
the derived fluid surface separately. Feed the same authoritative clipped
triangles/features to the Python stages, or prove the virtual-domain adapter
produces the identical predicates and root sizing. Exporting different surfaces
and hoping the resulting meshes agree is not a valid oracle.

## 4. Work package A — Produce the native target first

**First deliverable: a viewable, checked native D/40 cylinder mesh with frozen
inputs. No further speculative wall-algorithm changes before it exists.**

The installation is already located:

- Launcher: `/Applications/OpenFOAM-v2412.app/Contents/Resources/etc/openfoam`
- Executable: `/Users/flaviomartins/OpenFOAM/flaviomartins-v2412/platforms/darwin64ClangDPInt32Opt/bin/cartesianMesh`
- Executable SHA256: `6585db869dcc47920564d38d454ca286a44f9f95dcececbb7e10e58a421e18ce`
- Source revision: `3ff8555514827646c34cacfe5f0f691e49cdbc96`
- Source checkout currently: `/private/tmp/openonda-cfmesh.dwShoa/cfmesh`
- Upstream identity: `https://gitlab.com/openfoam/community/cfmesh.git`

Record linked-library hashes, compiler/build identity and launch environment too;
an executable hash alone does not identify a dynamically linked algorithm.
Copy required source provenance into durable evidence. Do not require the user
to rediscover or install cfMesh.

Run native `cartesianMesh` on the canonical fluid surface and dictionary. Save
the mesh, complete log, wall/refinement images, timings, peak RSS and
`checkMesh -allTopology -allGeometry` output. Parse the diagnostic result, not
only its exit code. Record and classify every warning.

Run twice to establish native repeatability. The old fixture notes report
explicit OpenMP team sizing and auxiliary-centre races despite
`OMP_NUM_THREADS=1`; do not assume that environment variable guarantees serial
execution, and do not use the previously unsafe `OMP_THREAD_LIMIT=1` workaround.
If necessary, build a separate, verified serial oracle from the pinned source
without changing its mesh operations. Preserve the stock executable and compare
stock/serial quality and outputs. Document any native nondeterminism before
choosing numerical tolerances.

**Exit A:** the actual native cylinder completes, its geometry and visual style
are acceptable for the request, and all input/provenance files can be replayed.
If native itself fails, fix the native case/preprocessing first. Do not tune the
Python port against a nonexistent or geometrically different target.

If native default geometry conflicts with an existing stricter product check,
record the conflict explicitly. Evaluate only documented native controls, such
as geometry constraints, in a separately labelled run. Any selected setting must
be mirrored in Python and the target re-frozen. Do not silently change the
meaning of “default” or weaken the product check.

## 5. Work package B — Locate the first different stage

Produce stage snapshots from the same canonical inputs on both sides. Native
`workflowControls { stopAfter <stage>; }` is supported by the inspected source;
use fresh case copies so restart metadata cannot accidentally skip work.
Use diagnostic-only native instrumentation where internal state is needed.

| Stage | Native operation to translate/verify | Required comparison |
| --- | --- | --- |
| Octree and template | Root sizing, refinement, automatic refinement, balancing, cell classification, `cartesianMeshExtractor`, octree addressing | Root and levels, complete selected-leaf set, classification, point/face/cell connectivity, edge-midpoint insertion |
| `surfaceTopology` | `checkIrregularSurfaceConnections`, `checkNonMappableCellConnections`, `checkCellConnectionsOverFaces`, fixed-point repetition, then `checkBoundaryFacesSharingTwoEdges` | Changed/removed cells, exposed faces, iteration trace, resulting topology |
| `surfaceProjection` | `preMapVertices`, `mapVerticesOntoSurface`, surface untangling | Per-point coordinates, selected triangle/feature and search/tie decisions, active/inverted point sets |
| `patchAssignment` | `edgeExtractor.extractEdges`, `updateMeshPatches` | Patch IDs and membership, topology changes, corner/edge/partition classifications |
| `edgeExtraction` | `meshSurfaceEdgeExtractorNonTopo`, `optimizeSurface` | Mapping/smoothing passes, feature constraints, point updates and resulting coordinates |
| `boundaryLayerGeneration` | `boundaryLayers.addLayerForAllPatches` | Wrapper columns, new points/cells, domain/body intersection treatment, normals and thickness computation |
| `meshOptimisation` | Surface optimisation, `optimizeMeshFV`, `optimizeLowQualityFaces`, conditional layer optimisation, `untangleMeshFV` | Native default parameters, active sets, decomposition, each update pass, refreshed auxiliary geometry and final mesh |
| Final output | Native renumbering and patch renaming | Full canonical face/cell incidence, patch names/types and coordinates; ordering differences separated from geometry differences |

For each stage record input/output hashes, counts, elapsed time, memory, branch
counts and the first mismatching entity. Maintain one comparison summary whose
answer is a stage name or PASS, not only a list of aggregate quality scores.

Compare topology modulo numbering and cyclic face starting vertices while
preserving orientation and owner/neighbour meaning. Preserve native iteration
and cell-face ordering during the algorithm: canonicalisation is for comparison,
not permission to reorder the arithmetic.

Use lossless intermediate coordinates. Start deterministic geometric comparisons
at an absolute tolerance of `1e-10 * L`, with `L` the domain diagonal and no
automatic relative-error slack. Establish the native repeatability envelope
first. If a discrepancy changes topology or a branch, investigate it even if
coordinates are close. Never widen a tolerance merely to make the port pass.
Exact byte identity of reordered files is not the target.

**Exit B:** the first divergence is reproduced and localised on the real cylinder.
Do not debug final wall appearance while an earlier topology or mapping stage
already differs. Retain existing small native fixtures as regression tests, not
as substitutes for this comparison.

## 6. Work package C — Translate the missing behaviour and make it fast

Repair the first different stage, rerun its comparison, then advance. Keep the
native formulas, predicates, tolerances, tie handling, update order, active-set
rules, constraint classification and iteration/termination logic together.
Do not approximate them with a generic smoother bearing the same function name.

For the current large-case bottleneck:

1. Remove the `n_faces > 200_000` zero-surface-pass and one-volume-pass policies
   from the conventional path. The algorithm must not change at a size threshold.
2. Translate the scalar surface-point objective, gradients, stabilisation,
   Newton/line-search operations and surrounding repeated loops into typed
   numerical kernels. Numba is already a project dependency and is already used
   by the volume port. Python plus compiled numerical kernels satisfies the
   built-in requirement; interpreted per-point Python overhead is not required.
3. Represent ragged incidence with contiguous indices/offsets where useful.
   Preserve the native traversal/update semantics, including Jacobi versus
   in-place updates, rather than sorting or parallelising indiscriminately.
4. Use binary64, `fastmath=False`, and a deterministic serial baseline. Avoid
   millions of tiny NumPy allocations, repeated incidence rebuilding and
   repeated global nearest-triangle searches. Cache only data whose invalidation
   is correct after coordinate/topology changes.
5. Broad-phase and batched searches must preserve the native candidate and
   tie-selection rules. Prove each accelerated kernel against identical native
   intermediate inputs, then rerun the whole cylinder stage.
6. Preserve all final wall and topology checks. Do not compensate for omitted
   relaxation with a post-projection or a lower-quality acceptance threshold.

Keep raw parity-stage output separate while resolving the current extra
`_constrain_cfmesh_wall_points` operation. If native constraints are needed for
the product, port those constraints in the corresponding native stages and use
the same documented settings on both sides. A final nearest-STL snap is not
equivalent to constrained optimisation of the neighbouring cells.

The specialised `optimizeBoundaryLayer` guard remains faithful to native;
implement its active body only if the frozen dictionary actually enables it.
Do not expand scope into optional inflation to repair the default wrapper.

**Exit C:** all applicable native cylinder stages match, including the full
optimisation sequence, and the ordinary public build produces a valid complete
mesh without size-based algorithm changes or native-executable delegation.

## 7. Work package D — Make reference_flow mesh-only and reproducible

Add a mesh-only entry point in `reference_flow`, sharing the case description
and public mesher configuration with `setup.py`. It must not start the flow
solver and must not rebuild an already-generated mesh just to export it.

Proposed command contract — these commands are deliverables, not currently
implemented commands:

```bash
cd tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow
python mesh.py --case coarse
```

It builds once, validates, then atomically publishes:

```text
solution/coarse/
  mesh.npz                 # lossless solver-native mesh
  mesh.vtu                 # genuine polyhedral cells for ParaView
  mesh_report.json         # complete validation and resolved settings
  mesh_manifest.json       # input/code identities, timings, output checksums
```

Allow an explicit output directory for isolated debugging. Refuse to overwrite
existing results by default. Preserve the old smoke/very-coarse outputs and
audit artifacts. The current `allrun.sh` deletes solution, samples and figures;
do not invoke it as a mesher test. Remove that destructive default as part of
the scoped workflow integration, without deleting any existing data.

Let `setup.py` explicitly load a saved mesh and check its manifest against the
requested case, so simulation does not silently remesh or use a stale grid.
Prove export/reload preserves every polygon loop, incidence and patch.

Keep native reference, stage comparisons and render evidence under a versioned
`reference_flow/mesh_evidence/<run-id>/` directory. Logs and metadata must survive
interruption. Checkpoint reuse must validate upstream mesh, configuration and
implementation identities; stale checkpoints are not permissible speedups.

## 8. Work package E — Objective finish line

All of the following are required for **cylinder delivery**:

- [ ] Native D/40 reference generated from the exact frozen fluid geometry and controls.
- [ ] Every applicable stage compared; no unexplained topology or coordinate difference.
- [ ] Full Python D/40 `build()` completes with native-equivalent optimisation semantics.
- [ ] Every cell is edge-closed; internal faces have opposite cell incidence; no unused/duplicate faces or disconnected fluid regions.
- [ ] Finite geometry, positive volumes/areas, and valid solver-relevant face decompositions; zero nonpositive owner/neighbour pyramids on **all** applicable faces.
- [ ] Normalized area-vector closure ≤ `1e-10`; source-wall and box-plane conformance checks remain active.
- [ ] Wall vertices within `1e-8 * L` of their assigned source surface, including the span intersection constraints; two-sided wall coverage gap ≤ `0.05 * h_local`.
- [ ] Maximum non-orthogonality ≤ 80° and p99 ≤ 65°; documented internal/boundary skewness checks meet the earlier plan. Do not mix different tools' metric definitions.
- [ ] Correct box/patch refinement levels and conforming 2:1 transitions, verified spatially, not inferred from total cell count.
- [ ] OpenFOAM checks executed on native and Python outputs; all failures/warnings retained and classified. Invalid topology or solver-relevant geometry is a failure, regardless of process exit code.
- [ ] Native and Python use identical cameras and actual mesh edges in domain, wake-transition, near-wall, and cylinder/span-intersection views. Also include a non-coplanar section such as `z/D = 0.013`. No centroid-only visual certification.
- [ ] Visual inspection confirms the native wrapper/transition structure and wall-cell relaxation, without folds, cracks, spurious caps, crushed rows or detached surface cells.
- [ ] Saved mesh reloads into FVM without regeneration and completes a bounded 20-step cylinder smoke run in an isolated output directory, with finite fields, converged configured solves and the earlier plan's mass-conservation checks. This is not a grid-independence claim.
- [ ] Warm coarse build ≤ 20 minutes and peak RSS < 8 GiB on the recorded Mac; cold/JIT overhead separately measured and ≤ 2 additional minutes, retaining the previous large-case budget.
- [ ] One rerun of the public mesh-only command reproduces the accepted result without cfMesh available to the runtime.

If native's conventional output itself fails a required product gate, report a
**native-default/product-contract conflict** and keep both results visible. Do
not certify an invalid mesh, pretend a post-snap is exact parity, or silently
weaken the gate. A plan cannot honestly guarantee numerical success in advance;
these conditions guarantee that incomplete work cannot be labelled success.

## 9. Stop wasting full-build cycles

Every run must expose the active stage, elapsed time and progress/iteration
counts. Flush a stage record at each boundary. A timed-out or interrupted run
is incomplete, not successful because it reached a later function.

On exceeding the 20-minute complete-build budget, preserve the checkpoint and
profile the active stage. Fix one measured bottleneck and test it from the
matched checkpoint before another full build. Do not repeatedly rerun all
upstream work with an unmeasured change, and do not fix performance by disabling
mesh operations. Two attempts without a measurable improvement require a
smaller same-stage reproducer and a revised explanation of the bottleneck.

Progress reports must state: last completed gate, first mismatching stage,
completed mesh files, measured quality, measured runtime, and next gate.
“Still optimising” is not enough. The first user-visible milestone is the
native reference image; the next is the matched complete Python cylinder image.

## 10. Grid-independence study comes after the first cylinder

Do not change the coarse target again to make the task easier. The current
D/80 and D/160 cases are later mesh-only gates, not permission to launch a long
simulation sweep now. Estimate their cell count and memory before building:
halving all three spatial scales can increase bulk cell count roughly eightfold,
so a successful coarse mesh alone does not establish that the full sequence is
practical. These are scaling estimates, not measured case counts.

Once the coarse deliverable is accepted, generate genuinely distinct remaining
grids, verify actual spacings and identities, and only then schedule the flow
study. Preserve duplicate-grid rejection in postprocessing. Do not claim grid
independence from the 20-step smoke test or from three requested `dx` values.

## 11. Source anchors for the implementing agent

Reviewed local sources, relative to the roots named above:

- Native `meshLibrary/cartesianMesh/cartesianMeshGenerator/cartesianMeshGenerator.C`:
  `surfacePreparation`, workflow order, wrapper insertion, final optimisation.
- Native `meshLibrary/utilities/octrees/meshOctree/meshOctreeAddressing/meshOctreeAddressingCreation.C`:
  `createOctreeFaces`, including edge-centre insertion into face loops.
- Native `meshLibrary/utilities/smoothers/geometry/meshOptimizer/meshOptimizer.H`:
  default FV iteration parameters, including five Laplacian passes.
- Native `meshLibrary/utilities/smoothers/geometry/meshOptimizer/optimizeMeshFV.C`:
  actual `optimizeBoundaryLayer` guard and executed branch.
- Native `meshLibrary/utilities/workflowControls/workflowControls.C`:
  stage saving and `stopAfter` behaviour.
- OpenONDA `source/solvers/fvm/mesh/cartesian/mesher.py`:
  unified dispatch, currently zeroed topology stage, size-dependent pass counts,
  extra post-optimisation wall projection and publication/validation boundary.
- OpenONDA `source/solvers/fvm/mesh/cartesian/cfmesh_surface_optimisation.py`:
  scalar surface kernels and current zero-iteration early return.
- OpenONDA `source/solvers/fvm/mesh/cartesian/cfmesh_mesh_optimisation.py` and
  `cfmesh_volume_optimisation.py`: default FV sequence and existing typed kernels.
- OpenONDA `tests/mesh_parity/fixtures/README.md`: useful surviving fixtures and
  important limits on their evidence, input precision and native concurrency.
- OpenONDA `artifacts/reference-flow-audit-2026-09-06/REPORT.md`: original failures
  and preserved evidence. Some small-case failures have since been repaired;
  do not misrepresent that older snapshot as the current result.

This plan used source review to distinguish missing work from a legitimate native
no-op, and documentation review to specify executable deliverables and pass/fail
gates. It does not change the meshing algorithm, run a new native cylinder, or
claim that any unchecked milestone has been completed.
