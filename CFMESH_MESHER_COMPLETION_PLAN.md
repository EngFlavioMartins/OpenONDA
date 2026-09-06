# Completion plan: a usable, built-in cfMesh-style Cartesian mesher

Date: 2026-09-06

Status: implementation plan; not a claim of completion

## 1. The result we are delivering

**Import STL objects, define a surrounding box, specify rectangular and patch
refinement, and build a wall-fitted Cartesian mesh directly usable by OpenONDA's
FVM solver.** This must be the ordinary mesher workflow, not an experimental mode.

The output must have:

- Cartesian/hexahedral cells away from surfaces and refinement transitions.
- Proper castellated, conforming polyhedral connections between different cell
  sizes. Fine cells and coarse cells share real faces, not disconnected grids.
- Boundary cells fitted to the imported walls, preserving resolved sharp edges,
  corners, curved surfaces, and passages.
- The conventional cfMesh Cartesian sequence, including its default boundary
  wrapper and final optimization. Wall fitting is not a staircase approximation
  or a geometry-specific O-grid.
- Correct named patches, a native FVM mesh, and inspection/export files.

The user must not configure octree balancing, topology repairs, feature mapping,
wrapper construction, optimizer iterations, or workflow checkpoints.

### What this plan can guarantee

No engineering plan can honestly guarantee a successful mesh for every possible
STL, including self-intersections, zero-thickness solids, and unresolved gaps.
The enforceable guarantee is a **release contract**: this task cannot be called
complete until the same public implementation passes every delivery gate below.
Passing a few primitives, matching cell counts, or producing a nice screenshot
does not satisfy that contract.

Valid acceptance cases must actually mesh successfully. Rejecting all difficult
geometries with a helpful error does not count as general-geometry support.

## 2. Scope and architecture decision

**Decision:** finish one native, surface-driven Cartesian pipeline, using the
existing cfMesh-derived modules where verified. Do not start another mesher or
continue maintaining two competing Cartesian algorithms.

| Option | Decision | Reason |
| --- | --- | --- |
| Complete the native cfMesh-style pipeline | Chosen | Meets the built-in FVM requirement and preserves verified work. |
| Invoke an installed cfMesh executable in production | Not chosen | Useful as a development reference, but not the requested built-in mesher. |
| Patch the old cut-cell/snapping pipeline into looking similar | Not chosen | Does not establish the required extraction, transition, and wrapper topology. |
| Introduce another meshing library or a new geometry-specific algorithm | Not chosen | Expands scope and does not finish the requested workflow. |

Production may use the project's Python, NumPy, and Numba dependencies. It must
not require OpenFOAM activation, native cfMesh, or an external meshing service.
Keep source provenance and applicable licence notices for translated routines.

### Supported input contract

- ASCII or binary STL representing one or more closed, manifold, non-intersecting
  solid objects, strictly inside the surrounding box.
- General shapes, including smooth, sharp, concave, thin-but-resolved, and
  multiple separated bodies. No shape recognition or filename-based branches.
- One connected exterior fluid region: the box minus the imported solids.
- Six domain limits, named outer patches, and named object wall patches.
- A background cell size, optional common boundary size, rectangular refinement
  boxes, and refinement of named object or outer-domain patches.
- Overlapping/nested refinement boxes and combined box-plus-patch refinement.
  The finest applicable request wins; the balancing region may extend beyond a
  requested refinement box.

STL does not reliably provide boundary-condition names. The minimal guaranteed
mapping is explicit: each imported `STLSurface(path, patch=...)` supplies a named
wall patch. Do not infer physics or patch names from filenames. Arbitrary
interactive subdivision of one object's wall is not a prerequisite for this
release; any imported region-label support must have an explicit tested mapping.

Open sheets, baffles, intersecting solids, objects crossing the domain, nested
material regions, CAD repair, anisotropic refinement, parallel mesh generation,
and user-designed multi-layer/y-plus controls are outside this first release.
The default cfMesh wrapper is **inside** scope; a configurable inflation-layer
product is not. Unsupported controls must be rejected, never accepted and ignored.

Input preflight may weld identical STL vertices and correct consistent shell
orientation without changing the surface. It must not silently close holes,
erase a body, merge a passage, or modify the requested geometry.

### Change from the earlier parity-first plan

This plan adopts the renewed objective: a useful **cfMesh-like mesher**, not a
bit-for-bit reimplementation as an end in itself.

- Keep exact topology/adjacency comparisons for the frozen reference cases and
  preserve existing passing regressions. Refinement-connectivity disagreements
  are algorithmic defects, not cosmetic differences.
- Use native checkpoints to localize defects. Do not spend additional work on
  harmless floating-point identity once topology, wall accuracy, and quality
  satisfy their fixed tests.
- Test practical mesh quality independently. Native cfMesh can itself produce
  poor or under-resolved meshes; reproducing one is not product acceptance.
- Do not hold delivery hostage to the historical 394,234-cell count: the exact
  508-triangle input behind that count has not been recovered. Freeze an available
  representative large case with its actual input and configuration instead.
- Any deliberate difference from the native default must be documented and
  tested. In particular, a quality/conformance correction must not be presented
  as exact default-output parity.

This is an explicit change in completion criteria, not permission to hide
connectivity failures or relax tolerances until a comparison passes.

## 3. The complete user workflow

Keep the existing public namespace and declarative configuration. The following
is the **target working example**, not a statement that today's default path
already implements it. Bounds use `(xmin, xmax, ymin, ymax, zmin, zmax)`.

```python
import openonda.fvm.mesher as msh
from openonda.fvm import create_fvm_solver

mesher = msh.CartesianMesher(
    surfaces=(msh.STLSurface("object.stl", patch="body"),),
    domain=msh.BoxDomain(
        bounds=(-3.0, 6.0, -3.0, 3.0, -3.0, 3.0),
        patches=msh.BoxPatches(
            xmin="inlet", xmax="outlet",
            ymin="sides", ymax="sides",
            zmin="span", zmax="span",
        ),
    ),
    max_cell_size=0.5,
    boundary_cell_size=0.25,
    refinements=(
        msh.BoxRefinement(
            name="wake",
            bounds=(-1.0, 4.0, -1.0, 1.0, -1.0, 1.0),
            cell_size=0.125,
        ),
    ),
    patch_refinements=(msh.PatchRefinement("body", cell_size=0.0625),),
)

mesh = mesher.build()  # Complete, validated mesh; no stop_after argument.
solver = create_fvm_solver(setup, mesh=mesh, case_dir="case")
```

The shipped tutorial must include a real, appropriately sized `object.stl` and a
complete `setup`, so users can run it unchanged. Solver boundary-condition
physics remain separate from mesh construction.

The public build must report requested versus effective dyadic sizes, cell
counts, timings, and quality. Normal octree quantization must follow the pinned
cfMesh size semantics; it is not permission to secretly choose another mesh size.
Optional minimum-size control must have explicit semantics for automatic
curvature/proximity refinement and must not override explicit patch/box requests
silently. Internal feature detection and the default wrapper are automatic.

## 4. Delivery sequence and mandatory gates

Each milestone ends with a runnable artifact and a pass/fail record. Until its
exit gate passes it is incomplete. After a core-stage change, rerun previously
passing small cases before using larger cases to assess progress.

### M0 — Freeze a reproducible starting point and acceptance contract

1. Inventory the current mesher changes without modifying unrelated work.
2. Locate the surviving oracle, reference arrays, reports, and native source.
   During preparation of this plan, older `docs/` content and `tools/mesh_parity/`
   were being removed by concurrent workspace changes. Do not restore or overwrite
   those changes blindly. Re-establish a small, durable test harness in the
   agreed repository layout before relying on any old command or report.
3. Pin the native reference, input checksums, complete effective configurations,
   and numerical gates. Feed both meshers identical decoded triangle coordinates;
   retain the FTR-based native-input precision fix.
4. Freeze the development matrix in section 5, including a combined-refinement
   nontrivial case. Freeze two additional holdout shapes before tuning the code.
5. Add the acceptance runner and result schema before further algorithm work.
   Initially it must honestly show failures and missing capabilities.

**Exit:** one command reproduces the current baseline and reports every required
case as pass, fail, or not implemented. Missing tests are not passes. No reference
input or essential evidence exists only under `/private/tmp`.

Target command to implement at M0, not an existing command to assume works:

```bash
python tests/mesh_parity/run_acceptance.py \
  --suite release --output artifacts/mesher-acceptance
```

Provide a small `--suite quick` for the repair loop. The release suite must fail
with a nonzero exit status for any failed, skipped, missing, or uninspected
mandatory case. Its manifest records the code revision and dirty diff hash,
input/configuration hashes, reference identity, all numerical results, image-review
status, elapsed time, and peak memory. Preserve a compact durable evidence bundle;
do not commit every intermediate mesh or create another collection of diaries.

### M1 — Complete refinement and conforming transition topology

1. Use one sparse octree path for all combinations of global, boundary, box,
   patch, and automatic refinement. Eliminate the divergent no-refinement path.
2. Apply the required cfMesh balancing after every refinement source, including
   multi-level boundary refinement. Support face/edge/corner neighbours without
   allocating a dense array at the finest resolution across the whole domain.
3. Extract genuine shared polygonal faces at coarse/fine interfaces. Support
   general face valence and general polyhedral cells throughout the pipeline.
4. Finish the first known refined-mesh failure: native cell decomposition checks
   face connections before making pyramids. Port and test that operation, including
   faces sharing multiple edges and polygonal boundary-face decomposition.
5. Exercise adjacent, overlapping, nested, and wall-intersecting refinement boxes,
   as well as simultaneous patch refinement and automatic refinement.

**Exit:** all refinement combinations produce closed, conforming templates with
correct requested/effective levels and zero face/adjacency mismatches against
the frozen native topology fixtures. The existing refined-cylinder edge-extraction
failure and the multi-level boundary-balancing failure are resolved by regression
tests, not bypassed or hidden by changing the inputs.

### M2 — Complete generic wall recovery and default wrapper

1. Implement the needed surface-topology corrections to convergence, following
   the native sequence. Add termination diagnostics for repair loops.
2. Preserve patch ownership and automatically detect/map geometric features,
   including sharp features inside a single named object patch.
3. Finish projection, edge/corner handling, default all-patch wrapper generation,
   and surface/volume optimization for arbitrary supported face/cell topology.
4. Validate wall distance, solid exclusion, feature preservation, and mesh
   validity after the final optimization—not just immediately after projection.
5. If the native default moves points off the wall beyond the product contract,
   address that explicitly with a documented, geometry-constrained operation.
   Test its validity transactionally; never append an unchecked nearest-point snap.
   Record raw native-parity results separately from this final quality policy.
6. Preserve resolved passages and objects. Diagnose genuinely insufficient
   resolution with the offending patch/location and a useful size recommendation;
   do not silently return a mesh that lost the feature.

**Exit:** every development shape produces a complete, wall-fitted mesh with both
refinement types enabled, passes the independent gates in section 6, and has
inspection images showing the wall, wrapper, and coarse/fine transitions.
No shape-specific source changes or per-case hidden optimizer settings are allowed.

### M3 — Make the normal public workflow work end to end

1. Route `CartesianMesher.build()` and its callable interface through the completed
   pipeline. Diagnostic checkpoint builds must be prefixes of that same pipeline,
   not a second implementation selected by a flag.
2. Keep one finalization path for geometry, patch types, validation, and reporting.
   Do not return an unvalidated mesh as a successful normal build.
3. Connect the result to the public `create_fvm_solver` factory. Test the installed
   package interface, not only internal imports or hand-assembled mesh arrays.
4. Provide native saved-mesh output, ASCII OpenFOAM `polyMesh` export, and a VTK
   inspection export through documented production interfaces. Test round trips,
   face orientation, owner/neighbour ordering, and named patch preservation.
5. Ship one copy-and-run tutorial containing the complete STL → box → refinements
   → mesh → solver workflow. Export and inspection require no development scripts.

**Exit:** the tutorial runs through the ordinary API in a fresh process with no
OpenFOAM environment and no cfMesh executable available. Mesh construction,
export/reload, boundary conditions, and a short flow run all succeed.

### M4 — Prove generality and inspect the actual meshes

1. Run the complete matrix and transformed variants without algorithm changes
   between cases. Include plain STL input with no precomputed feature file.
2. Run the frozen holdouts only after the development matrix passes. If a holdout
   exposes a defect, add a minimal regression, fix the general cause, and replace
   that holdout with a new one for the next independent check.
3. Save deterministic surface-edge views, three orthogonal interior sections,
   wall close-ups, and refinement-transition close-ups. Include patch labels,
   requested sizes, scale, and quality overlays; inspect all required images.
4. Compare native and OpenONDA results with identical effective inputs. Preserve
   exact topology gates on the frozen parity fixtures. Investigate unexplained
   topology or feature-loss differences on other cases rather than dismissing
   them because the picture looks acceptable.

**Exit:** 100% of supported development and holdout cases pass. Visual inspection
finds no staircase walls, cracks, lost bodies, closed-off resolved passages,
folded boundary cells, or incorrect patch assignment. Automated geometry and
topology checks corroborate those images.

### M5 — Certify solver usability and practical resource cost

1. Run constant/linear-field geometry checks and conservative internal-face flux
   cancellation on meshes containing coarse/fine transitions and wrapper cells.
2. Run at least 20 FVM time steps on a curved object and on a nontrivial concave
   or multi-body case, each with box-plus-patch refinement. Include no-slip object
   walls, an inlet, an outlet, and the configured outer patches.
3. Check finite fields/residuals, convergence against the configured solver
   tolerances, cellwise continuity, and global mass conservation at every step.
4. Benchmark one approximately 100k-cell case and one 400k–600k-cell case on the
   same recorded Mac/environment. Profile and fix demonstrated bottlenecks only.
5. Product performance targets: warm builds within 5 minutes for the 100k case
   and 20 minutes for the larger case, peak RSS below 8 GiB for the latter.
   Record cold-start/JIT time separately and require it below 2 additional minutes.
   These are proposed release targets, not measurements already achieved.

**Exit:** both flow cases meet section 6, the larger mesh passes the same geometry
and topology checks, and measured resource use meets the frozen targets. No dense
finest-grid allocation, uncontrolled growth, or silent quality bypass is permitted.

### M6 — Finish, simplify, and hand off

1. Remove obsolete Cartesian cut-cell/snapping/fallback paths only after the new
   default passes M0–M5. Migrate their callers; do not leave a hidden fallback.
2. Remove unsupported public controls or raise explicit configuration errors.
   Do not expand this cleanup to unrelated solver/mesher functionality.
3. Keep focused unit tests, public-API integration tests, saved native fixtures,
   the oracle runner, and the release acceptance command.
4. Run the relevant regression suite, Ruff, and the repository's required Pyrefly
   checks after final code/API changes. Verify the installed tutorial again.
5. Deliver one concise verification report, the reproducible inputs, generated
   meshes, inspection images, quality results, timings, and solver logs.

**Exit:** a clean rerun of the acceptance command succeeds for the exact code
revision being handed over. Only then call the mesher complete.

## 5. Acceptance matrix: generality is mandatory

For each development shape, run four configurations: **default**, **box only**,
**patch only**, and **box plus patch**. Default means the complete conventional
workflow with no custom topology, wrapper, or optimizer controls.

| Shape/test family | Required evidence |
| --- | --- |
| Aligned and translated/rotated sharp object | Corners, feature edges, outer boundaries, absence of alignment assumptions. |
| Closed curved cylinder | Wall fit and default wrapper, including the original refinement regressions. |
| Ellipsoid | Smooth arbitrary surface, unequal principal curvatures, multi-level boundary sizing. |
| Concave body or torus | Concave recovery and a preserved through-passage. |
| Closed finite wing-like body | Thin resolved regions, leading/trailing features, no lost solid. |
| Two disjoint unequal bodies | Separate named walls, per-patch sizing, connected exterior fluid. |
| Nested/overlapping boxes around a curved wall | Finest-request precedence and real transition topology across multiple levels. |
| Outer-patch refinement | Named box-boundary refinement and correct boundary/wrapper intersections. |

Use modest meshes for development, with adequately resolved geometric features.
Freeze geometry and sizes before fixing a failing case. If the reference proves
an input under-resolved, retain it as a rejection test and add a separately named,
properly resolved case; do not silently rewrite the original case into a pass.

Additional mandatory tests:

- Two frozen holdout STLs: one smooth/complex and one sharp/concave, not used to
  tune the implementation. Both run default and combined refinement.
- Translation and uniform scaling of geometry, domain, and all size controls;
  patch renaming; deterministic reruns; ASCII/binary encodings of the same
  representable surface coordinates.
- General rotations are tested for successful meshing, not identical cell counts:
  rotating a body relative to a Cartesian grid legitimately changes its topology.
- Seeded perturbations of refinement-box placement and size, including dyadic
  threshold boundaries. Compare like-for-like with the reference.
- Invalid STLs, zero/negative sizes, unknown patch names, colliding patch names,
  unsupported controls, and memory-budget exhaustion fail with useful diagnostics.

Invalid-input tests are reported separately and cannot inflate the success rate
of valid geometry cases.

Test responsibilities are explicit: unit tests cover size quantization, spatial
predicates, balancing, face splitting, and metric formulas; integration tests
cover complete builds, refinement combinations, patch ownership, and export
round trips; end-to-end tests cover the installed tutorial, holdouts, large mesh,
and both FVM runs. All listed user-facing behaviors require coverage; a line-
coverage percentage is not a substitute.

## 6. Fixed, independent numerical acceptance gates

M0 must encode these gates in versioned tests before further mesher tuning.
They are proposed product requirements, not claims about native cfMesh defaults.
Existing stricter applicable checks must not be weakened to obtain a pass.

Let `L` be the domain-box diagonal and `h_local` the effective target wall size.

| Area | Required gate |
| --- | --- |
| Topology | Closed cells; valid unique faces; consistent winding; one owner per face; exactly one neighbour on internal faces; no neighbour on boundary faces; no cracks or non-manifold connections. |
| Geometry | Finite coordinates/metrics, strictly positive cell volumes and face areas, valid solver-relevant face/cell decompositions. Normalized cell area-vector closure error ≤ `1e-10`. |
| Fluid region | One connected exterior component, every intended object present, no fluid cell centre inside a solid, no non-wall face crossing the solid away from the allowed wall-approximation band. |
| Wall vertices and features | Maximum vertex-to-assigned-STL distance ≤ `1e-8 * L`; feature vertices remain on their corresponding feature edges/corners. Preserve stricter existing conformance checks. |
| Wall coverage | Check both directions: generated wall to STL and STL to generated wall. Sample triangle interiors as well as vertices. Maximum surface gap ≤ `0.05 * h_local`; no missing component, passage, or feature larger than the prescribed resolution. |
| Domain boundary | Correct box planes within `1e-8 * L`; complete named patches and consistent outward normals. |
| Refinement | Correct effective octree levels and overlap precedence; required 2:1 regularity; conforming transition faces. Verify actual spatial refinement, not just increased total cell count. |
| Quality | Maximum non-orthogonality ≤ 80°, p99 ≤ 65°; maximum internal-face skewness ≤ 4 and boundary-face skewness ≤ 20 using explicitly documented OpenFOAM-style definitions. Cross-check metric implementations before applying limits. |
| Volume | Compare to box volume minus signed STL solid volumes; relative fluid-volume error ≤ 0.5% on the adequately resolved acceptance cases, alongside local solid/wall checks. |
| Conservation | Equal-and-opposite internal-face fluxes to roundoff. In flow tests, normalized global boundary imbalance ≤ `1e-6` and normalized summed absolute cell continuity defect ≤ `1e-5` after each completed pressure correction sequence. |
| Solver | At least 20 completed steps for both flow cases; finite velocity, pressure, fluxes, and residuals; configured linear-solver convergence achieved; no orientation/connectivity or boundary-condition errors. |

For conservation, normalize by a fixed nonzero prescribed inlet flow for each
test, not a denominator chosen from the observed error. Record absolute values
as well. Define skewness, closure, and continuity formulas once and unit-test
their implementations; different tools' similarly named metrics are not
automatically interchangeable.

Distance-to-STL is not the distance to a fitted analytic cylinder or other
substitute geometry. Wall-face interiors may chord a curved triangulated
surface, so do not demand zero distance everywhere on every polygon. Conversely,
testing only wall vertices cannot detect missing surface coverage.

Run native `checkMesh -allTopology -allGeometry` where available and parse its
diagnostics, not merely its exit code. Classify additional concavity/decomposition
warnings against the actual FVM operators; never ignore an invalid decomposition
used by OpenONDA. A native reference with poor quality is not an exemption from
these product gates.

If a gate proves incorrectly specified, document the mathematical or physical
reason and make an explicit contract revision before implementation continues.
Do not edit thresholds during a failing run or silently redefine success.

## 7. Rules that prevent another open-ended debugging cycle

1. Work on one reproducible failing behavior at a time. Save the input, first
   failing stage, minimal failing region, hypothesis, and expected measurable change.
2. Add the regression before the repair; change only the responsible operation.
   Keep a patch only when it fixes the stated defect without breaking prior gates.
3. After two unsuccessful hypotheses, stop speculative editing. Inspect the
   responsible native operation and produce a smaller reproducer before resuming.
   This is a diagnostic reset, not permission to start another mesher.
4. Do not chase bitwise smoother output at the expense of a failing required
   geometry, broken refinement transition, missing default API, or absent flow test.
5. No geometry names, special coordinates, oracle counts, hidden mesh-size
   substitutions, arbitrary displacement rings, or silent alternative algorithms.
6. Every progress report states: passing product cases, failing cases and their
   first failing stage, completed user-visible milestone, and the next concrete gate.
7. Do not broaden into a GUI, a full cfMesh clone, custom inflation controls, CAD
   repair, or unrelated repository cleanup. This task requests a plan now; the
   plan itself does not authorize a commit, push, or destructive cleanup.

## 8. Final completion checklist

- [ ] Ordinary `build()` produces the complete cfMesh-style pipeline.
- [ ] Default Cartesian cells, castellated transitions, and wall-fitting wrapper work.
- [ ] Rectangular and named-patch refinement work separately and together.
- [ ] All valid development cases and fresh holdouts generate usable meshes.
- [ ] Fixed topology, geometry, wall, quality, and conservation gates pass.
- [ ] Actual surface/section/refinement images have been inspected and saved.
- [ ] The same generated meshes pass both public FVM flow tests.
- [ ] Export/reload and a fresh installed-package tutorial work without cfMesh.
- [ ] The representative large case meets the measured resource limits.
- [ ] Obsolete competing Cartesian paths are removed after certification.
- [ ] Focused regressions, static checks, evidence, and reproduction instructions ship.
- [ ] One final acceptance run passes on the exact delivered revision.

**Completion means the user can replace the tutorial STL and adjust the box and
refinement sizes for another supported object, without editing mesher internals.**
Until every checkbox is satisfied, report the remaining work explicitly.

## 9. Evidence and reference anchors

Starting evidence from the prior work: aligned/oblique objects and a coarse
cylinder passed eight native checkpoints, and an explicit patch-refinement case
also passed. More demanding refined cases failed at edge extraction. The public
default and final solver certification remained unfinished. These are starting
observations to reproduce at M0, not pre-approved product acceptance.

The pinned native reference previously recorded was OpenFOAM-v2412 with cfMesh
source commit `3ff8555514827646c34cacfe5f0f691e49cdbc96`; executable SHA256
`6585db869dcc47920564d38d454ca286a44f9f95dcececbb7e10e58a421e18ce`.
Reverify that identity before generating new comparisons.

The inspected native `cartesianMeshGenerator.C` calls, in order, Cartesian
extraction, repeated surface preparation, mapping, patch/edge extraction,
`boundaryLayers::addLayerForAllPatches()`, final optimization, and optional
configured layer refinement. That source sequence—not a generic cut-cell
description—is the implementation map for M1/M2.

The cfMesh authors' [gap-handling examples](https://cfmesh.com/gap-handling-with-cfmesh/)
show that resolution and cell-retention choices can lose passages or obstacles.
This is why reference similarity alone is insufficient and why preservation of
the actual requested fluid/solid geometry is a separate release gate.
