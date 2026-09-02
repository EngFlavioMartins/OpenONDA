# OpenONDA general Cartesian mesher: implementation programme and agent contract

Status: engineering specification for `EngFlavioMartins/OpenONDA`, audited on 2026-09-02 against `development` commit `48d72e997337b4f6aaeeceb325d1367876e29514`.

## 1. Decision

OpenONDA must have one **surface-driven, geometry-agnostic Cartesian mesher**. A tutorial may describe *what* mesh is wanted, but it must never implement *how* that mesh is constructed.

The required public object is `fvm.CartesianMesher`. It accepts triangulated surfaces, an outer domain, physical cell-size requests, refinement regions, feature controls, and optional boundary-layer controls. It returns OpenONDA's native face-based FVM mesh. Geometry-specific meshers such as `ExplicitCylinderGridMesher` are forbidden and must be deleted once the replacement passes the acceptance matrix.

This is a cfMesh-inspired mesher, not a claim of API or output compatibility with cfMesh. Its pipeline follows the openly documented cfMesh Cartesian workflow, while its public interface and output are native to OpenONDA.

## 2. Why the current implementation is not the requested mesher

The present code contains useful foundations, but the public capability is not general-purpose.

| Evidence in `development` | Consequence |
| --- | --- |
| [`CFMESH_ATTRIBUTION.md`](https://github.com/EngFlavioMartins/OpenONDA/blob/development/source/solvers/fvm/mesh/CFMESH_ATTRIBUTION.md) describes an axis-aligned subset created for cube verification and explicitly excludes general surface mapping, feature/corner extraction, untangling, optimisation, general boundary layers, geometry repair, and parallel meshing. | The repository already admits that the implemented scope is only a subset of cfMesh. The attribution document is also stale because the Python code now attempts a general-surface projection path. |
| [`adaptive_cartesian.py`](https://github.com/EngFlavioMartins/OpenONDA/blob/development/source/solvers/fvm/mesh/adaptive_cartesian.py) is a 2,383-line module containing octree construction, topology extraction, surface projection, a cylinder O-grid, and a separate exact grid-study mesher. | Algorithmic layers are coupled in a monolith, making case-specific additions easier than general extensions. |
| `ExplicitCylinderGridMesher` hard-codes `1h/2h/4h/12h` regions, assumes a centred cylinder, and contains cylinder-specific coordinates and an O-grid interface. It is exported by both [`source.solvers.fvm`](https://github.com/EngFlavioMartins/OpenONDA/blob/development/source/solvers/fvm/__init__.py) and [`openonda.fvm`](https://github.com/EngFlavioMartins/OpenONDA/blob/development/openonda/fvm.py). | A scientific case has become a public meshing algorithm. |
| [`boundary_layer.py`](https://github.com/EngFlavioMartins/OpenONDA/blob/development/source/solvers/fvm/mesh/boundary_layer.py) recognises only a straight, z-aligned circular cylinder and exposes `interface_half_width` and `spanwise_cell_size` in the nominally general `BoundaryLayerSpec`. | Boundary-layer generation is geometry-specific rather than patch-normal and surface-driven. |
| General curved conformance projects wall vertices independently to their nearest triangle. A rejected projection silently leaves a local staircase corner. The acceptance check permits a wall-area error of 35%. | The mesher can call a partially snapped boundary conformal. This is not an adequate correctness gate for a general body-fitted FVM mesh. |
| [`test_curved_cylinder_mesh.py`](https://github.com/EngFlavioMartins/OpenONDA/blob/development/tests/fvm/test_curved_cylinder_mesh.py) is the only visible arbitrary-curvature regression and it tests the same cylinder used to develop the implementation. | The tests demonstrate cylinder support, not geometric generality. |
| The airfoil tutorial still builds its mesh through a case-local [`mesh_airfoil.py`](https://github.com/EngFlavioMartins/OpenONDA/blob/development/tutorials/fvm/airfoil_flow/assets/mesh_airfoil.py) using Gmsh. The cube and plate tutorials contain independent topology builders. | The native meshing API has not displaced case-local mesh implementations. |

The reusable parts should be retained: native face topology, dyadic refinement, coarse/fine subfaces, surface intersection and inside/outside queries, mesh geometry, native validation, provenance, and the callable mesh contract in `create_fvm_solver`.

## 3. Non-negotiable definition of “general-purpose”

The mesher is complete only when all of the following are true.

1. **Surface-driven:** the same implementation accepts any closed, watertight, orientable triangulated surface within its documented scale and topology limits. Rotating or translating a geometry cannot select a different algorithm.
2. **Intent-driven configuration:** the tutorial names surfaces, boundaries, sizes, refinement regions, features, and layers. It contains no point generation, cell tiling, face assembly, O-grid mathematics, solid classification, or stitching.
3. **No geometry identities in the engine:** no production class, function, branch, or validation rule asks whether a body is a cylinder, airfoil, cube, plate, sphere, or a named tutorial case.
4. **Native result:** the mesher constructs OpenONDA's native face-based mesh in memory. It does not launch OpenFOAM, cfMesh, Gmsh, or another external mesher behind the Python API.
5. **Patch-preserving:** configured surface and outer-domain patch names survive into `mesh_data["boundary"]` without case code renaming face ranges.
6. **Feature-aware:** sharp edges and corners are detected or supplied, captured topologically, and kept during surface optimisation.
7. **Region-refinable:** surface, feature, and volume sizing are independent. Volume refinement supports at least cfMesh's public primitive set: box, sphere, cone/cylinder, and line.
8. **Layer-capable:** boundary layers are extruded from arbitrary selected wall patches using local surface normals, with collision, concavity, termination, and quality handling. A cylinder-only O-grid does not satisfy this requirement.
9. **Strictly valid or explicit failure:** no negative-volume cells, open topology, orphan faces, fluid cells inside solids, silent unsnapped wall points, or quality-limit violations are returned. A failure identifies the stage, entities, and measured values.
10. **Reproducible:** identical input and configuration produce a canonically identical mesh and report on the same supported platform.
11. **Scientifically verified:** convergence and solver-operator tests accompany topology tests. A mesh that merely renders is not accepted.
12. **Credited honestly:** documentation distinguishes inspiration, reimplementation, and any translated code. It names Dr Franjo Juretić and Creative Fields, links the upstream source and user guide, preserves required GPL notices, and does not attribute an unverified behaviour to cfMesh.

## 4. Required Python interface

This is the target interface. Agents may improve names only through a design proposal; they may not replace it with dictionaries or case-specific constructor arguments.

```python
from pathlib import Path

import openonda.fvm as fvm

mesh = fvm.CartesianMesher(
    domain=fvm.BoxDomain(
        bounds=(-8.0, 20.0, -8.0, 8.0, -2.0, 2.0),
        patches=fvm.BoxPatches(
            xmin="inlet",
            xmax="outlet",
            ymin="farfield",
            ymax="farfield",
            zmin="front",
            zmax="back",
        ),
    ),
    surfaces=(
        fvm.STLSurface(
            Path("assets/cylinder_long.stl"),
            patch="cylinder",
        ),
    ),
    max_cell_size=0.50,
    boundary_cell_size=0.0625,
    min_cell_size=0.015625,
    refinements=(
        fvm.BoxRefinement(
            name="wake",
            bounds=(-1.0, 12.0, -2.0, 2.0, -2.0, 2.0),
            cell_size=0.125,
        ),
        fvm.SphereRefinement(
            name="near_body",
            centre=(0.0, 0.0, 0.0),
            radius=2.0,
            cell_size=0.0625,
        ),
    ),
    features=fvm.FeatureRefinement(
        angle=35.0,
        cell_size=0.03125,
    ),
    boundary_layers=(
        fvm.BoundaryLayers(
            patches=("cylinder",),
            layers=10,
            first_cell_height=0.004,
            growth_ratio=1.15,
        ),
    ),
)

setup = fvm.FVMSetup(
    case_name="cylinder_reference",
    mesh=fvm.MeshQualityConfig(
        max_non_orthogonality_deg=70.0,
        max_skewness=2.0,
        max_aspect_ratio=200.0,
    ),
    # physics, boundaries, numerics, output, and samplers follow
)

with fvm.create_fvm_solver(setup, case_dir=CASE_DIR, mesh=mesh) as solver:
    solver.run()
```

### Interface rules

- Constructors hold declarative physical intent. They are immutable after construction.
- `cell_size` means a requested upper size. Because Cartesian octree refinement is dyadic, the mesher records the effective size selected for every request. It never invents an exact 3:1 transition to satisfy one grid study.
- `min_cell_size` is a lower safety limit, not an alternative target.
- Multiple refinements combine by taking the smallest requested size at a location.
- Boundary-condition physics remain in `FVMSetup.boundaries`; meshing objects define only patch geometry and names.
- `mesh.build()` and `mesh()` are equivalent. `build()` returns native `mesh_data`; `mesh.report` exposes the immutable generation and quality report after a successful build.
- The public API contains no `preserve_body_geometry` switch. Preserving the supplied geometry within documented discretisation tolerance is mandatory.
- Unsupported surface topology or layer configuration fails at construction or the relevant build stage with a specific exception. It never selects a tutorial-specific fallback.

## 5. Internal architecture

Split the present monolith into a small pipeline whose stages correspond to physical meshing operations.

```text
source/solvers/fvm/mesh/cartesian/
    config.py              immutable public construction objects
    surface.py             STL ingestion, patches, manifold checks, spatial index
    features.py            sharp-edge and corner classification
    size_field.py          surface/feature/volume size requests
    octree.py              template creation, refinement, 2:1 balancing
    extraction.py          fluid-region selection and polyhedral topology
    surface_recovery.py    boundary preparation and conformal surface mapping
    boundary_layers.py     patch-normal extrusion and layer termination
    optimisation.py        surface and volume quality improvement
    native_mesh.py         OpenONDA mesh-data assembly and canonical numbering
    report.py              provenance, requested/effective sizes, quality evidence
    mesher.py              short orchestration class only
```

Existing generic `geometry.py`, `topology.py`, and `validation.py` remain authoritative. Do not duplicate their computations inside the mesher.

The pipeline follows the high-level stages visible in the upstream [`cartesianMeshGenerator`](https://github.com/wyldckat/cfMesh/blob/master/meshLibrary/cartesianMesh/cartesianMeshGenerator/cartesianMeshGenerator.C): octree/template creation, Cartesian extraction, surface-topology preparation, mapping to the surface, feature and corner recovery, surface optimisation, boundary-layer creation, final optimisation, renumbering, and patch naming. This is architectural lineage, not permission to collapse the stages into one translated file.

Every stage accepts a typed result and returns a typed result plus diagnostics. It must be independently testable. No stage reads tutorial constants or imports from `tutorials/`.

## 6. Implementation programme

Each phase is one reviewable pull request. The next phase cannot start until the previous phase's gates pass. Every PR includes code, tests, generated mesh reports, and a concise engineering note under `docs/verification/cartesian_mesher/`.

### Phase 0: freeze the contract and expose the gaps

Deliverables:

- Add this specification, an architecture decision record, and a public API test.
- Add failing geometry-independence and acceptance-matrix tests before changing algorithms.
- Record the current cylinder, cube, and curved-projection behaviour as baseline evidence, not as desired golden output.
- Audit licensing. The repository's current top-level `license` file contains only a short GPL heading, while mesher documentation claims `GPL-3.0-or-later`. Add the complete intended licence text, SPDX metadata, and a third-party notice after maintainer confirmation.
- Remove unsubstantiated claims such as calling silent partial snapping “cfMesh's real robustness fallback.”

Gate: reviewers can point from every requirement in Section 3 to a test name or a later explicitly scheduled test.

### Phase 1: typed, shape-agnostic configuration

Deliverables:

- Implement the public objects in Section 4 with complete docstrings, units, examples, validation, and exports through `openonda.fvm`.
- Replace hard-coded outer patch names with `BoxPatches` mapping.
- Introduce a general size-field interface and the box, sphere, cone/cylinder, and line refinement primitives.
- Make requested-to-effective dyadic resolution explicit in the build report.
- Keep the existing meshing engine temporarily behind `CartesianMesher`, but do not add compatibility wrappers for cylinder-specific arguments.

Gate: all configuration combinations can be constructed and validated without importing a tutorial. AST tests reject geometry-specific identifiers in the new package.

### Phase 2: robust surface model and features

Deliverables:

- Support one or more STL surfaces with stable patch IDs and configured patch names.
- Validate finite coordinates, non-degenerate triangles, orientation, watertightness, manifold edges, disconnected components, and surface/domain relationships.
- Replace brute-force nearest-triangle searches with an indexed query suitable for production geometry.
- Implement deterministic inside/outside classification with explicit handling of ambiguous rays and points on the surface.
- Detect feature edges from adjacent-face angle and feature corners from edge valence; accept optional user-supplied features.

Gate: rotated cube, sphere, torus/concave body, and two-body fixtures pass ingestion and classification tests at several translations and rotations. Invalid STL fixtures fail with stable, diagnostic errors.

### Phase 3: octree, size field, and native extraction

Deliverables:

- Build the octree from the combined background, surface, feature, and volume size field.
- Enforce 2:1 balance as a global invariant.
- Select the fluid region using the validated surfaces and an explicit region rule; do not use a geometry name or bounding-box approximation.
- Extract native polyhedra with correct owner/neighbour orientation and deterministic global numbering.
- Preserve patch identity through extraction.

Gate: every acceptance geometry passes topology, positive-volume, cell-closure, boundary-closure, 2:1 balance, patch-area, refinement-placement, and determinism tests before surface recovery is attempted.

### Phase 4: general surface recovery and quality optimisation

Deliverables:

- Prepare boundary topology so surface-adjacent cells are mappable.
- Map boundary vertices to the correct surface patch while constraining sharp edges and corners.
- Optimise surface and near-surface volume cells using objective quality measures.
- Replace point-by-point “snap or leave a staircase” behaviour with a transactional stage: it returns a conformal valid mesh or raises a diagnostic failure.
- Retain the input triangles unchanged; the surface is the geometric authority.

Gate: all acceptance geometries pass bidirectional surface-distance, patch-area, feature-distance, positive-volume, non-orthogonality, skewness, and solver-gradient tests. No rejected mapping is hidden inside a successful report.

### Phase 5: generic boundary layers

Deliverables:

- Extrude layers from selected patch faces along smoothed, feature-constrained surface normals.
- Honour `layers`, `first_cell_height`, and `growth_ratio` within documented tolerances.
- Detect collisions and opposing surfaces, terminate layers at selected concave/convex features, and allow explicit per-patch overrides.
- Stitch layers to the Cartesian core without a circular O-grid, square interface, fixed axis, or spanwise special case.
- Remove `interface_half_width` and `spanwise_cell_size` from the boundary-layer API.

Gate: layers pass on a sphere, a rotated sharp body, and a finite wing. Layer count, first height, monotonic growth, normal alignment, topology, collision handling, and final cell quality are measured. Passing on a cylinder alone is insufficient.

### Phase 6: solver integration and tutorial migration

Deliverables:

- Migrate the cylinder reference, cube, and airfoil/wing examples to the same `CartesianMesher` API.
- Delete `ExplicitCylinderGridMesher`, the cylinder O-grid engine, and their exports. Do not retain aliases or deprecation shims.
- Delete case-local mesh topology code made obsolete by the new mesher. Keep deliberately structured analytical meshes only where their structure is part of the verification problem.
- Update `capabilities.json`, the FVM README, API docs, attribution, and examples.
- Make every tutorial setup physics-first: it constructs meshing objects but contains no meshing algorithm.

Gate: the three migrated tutorials differ only in geometry files and declarative configuration. A one-step and short-run FVM smoke test succeeds on each generated mesh. Existing solver verification remains green.

### Phase 7: scale, parallel preparation, and release qualification

Deliverables:

- Profile surface queries, octree operations, extraction, recovery, and optimisation separately.
- Remove accidental quadratic work and cap peak-memory duplication.
- Add canonical mesh caching keyed by surface hashes, configuration, mesher version, and floating-point platform metadata.
- Record scaling at three cell counts. Establish timing and memory baselines before adding regression limits.
- Design parallel meshing only after the serial deterministic path is qualified; do not mix MPI complexity into earlier correctness phases.

Gate: documented scaling, no material solver-test regression, reproducible reports, and an updated support matrix that distinguishes qualified, experimental, and unsupported geometry classes.

## 7. Mandatory acceptance matrix

Small deterministic STL fixtures must be committed or generated by test utilities independent of production meshing code.

| Geometry | Purpose | Required configurations |
| --- | --- | --- |
| Rotated box | Planar patches, sharp edges, corners, orientation invariance | 0°, 17°, and 41° rotations; two translations |
| Sphere or ellipsoid | Smooth arbitrary curvature | three boundary sizes for convergence |
| Torus or another concave closed body | Concavity, genus, inside/outside classification | two orientations |
| Finite NACA wing | Mixed smooth curvature, leading/trailing features, finite tips | with and without layers |
| Two disjoint bodies | Multiple surfaces and patch identity | different surface cell sizes per body |
| Deliberately broken surfaces | Failure behaviour | open edge, non-manifold edge, inverted component, degenerate triangle, self-intersection if supported |

For every valid fixture, test:

- topology indices and owner/neighbour consistency;
- each internal face used by exactly two cells and each boundary face by one;
- positive face area and cell volume;
- cell closure, `||ΣSf|| / Σ||Sf||`, within a scale-aware tolerance;
- no fluid cell centre inside a solid and no positive-volume surface crossing missed by the classifier;
- correct outer and body patch names;
- requested/effective refinement sizes and monotone cell-count growth under refinement;
- 2:1 octree balance;
- surface and feature distance normalised by local cell size;
- patch-area convergence under refinement, not merely a single loose threshold;
- finite mesh-quality metrics and configured quality-limit enforcement;
- exact or tolerance-qualified reconstruction of constant and linear fields;
- one pressure/momentum assembly and one accepted FVM step;
- canonical equality of two repeated builds.

Additional layer tests measure layer count per patch region, first-cell height, growth sequence, normal alignment, termination consistency, collision outcome, and layer/core interface closure.

## 8. Anti-shortcut rules for coding agents

These rules are part of acceptance, not style suggestions.

- Do not add a production identifier containing a test geometry or tutorial name under `source/solvers/fvm/mesh/cartesian/`.
- Do not branch on geometric recognition, dimensions characteristic of a cylinder, alignment with a coordinate axis, or a file/case name.
- Do not implement mesh vertices, faces, or cells under `tutorials/**/assets/`.
- Do not call an external meshing executable and label the result native.
- Do not silently fall back to Gmsh, immersed boundaries, cut-cell staircases, partial snapping, fewer boundary layers, or a coarser size.
- Do not add legacy aliases, compatibility arguments, or deprecation adapters for `ExplicitCylinderGridMesher`.
- Do not weaken global validation to pass one fixture. Any tolerance change requires a dimensional argument and results across the full acceptance matrix.
- Do not use a rendered picture as proof of conformance. Numerical evidence is mandatory.
- Do not copy upstream source without recording file-level provenance and satisfying its licence requirements. Conversely, do not claim a behaviour comes from cfMesh without a precise upstream source.
- Do not declare completion while a required phase is represented only by a placeholder, `NotImplementedError`, skipped test, `xfail`, warning, or “experimental” label.

An automated architecture test should scan the production mesher for forbidden geometry terms. Its allowlist is restricted to documentation, test fixture names, migration notes, and third-party attribution.

## 9. Evidence required from every agent

Every implementation PR must finish with this evidence, in the PR description and in `docs/verification/cartesian_mesher/phase_<n>.md`:

1. Files added, changed, and deleted, with the reason for each.
2. Requirement-to-test mapping.
3. Exact test commands and unabridged pass/fail counts.
4. Mesh-report JSON for every applicable acceptance fixture.
5. A table of cell count, patch count/area, minimum volume, maximum non-orthogonality, maximum skewness, maximum aspect ratio, surface-distance error, and generation time.
6. Requested versus effective cell sizes for all size controls.
7. Any unsupported input, its explicit exception, and why support is deferred.
8. Confirmation that input surface hashes are unchanged.
9. `rg`/AST evidence that no geometry-specific production path or obsolete public export remains.
10. A frank list of incomplete items. If any mandatory item is incomplete, the phase status is **not complete**.

The reviewing agent must independently rerun the tests and inspect at least one mesh report and one generated VTK mesh. It may certify only requirements supported by reproduced evidence.

## 10. Master prompt to give the implementation agent

> Implement the OpenONDA general Cartesian mesher according to `OpenONDA_general_cartesian_mesher_program.md`. Work only on the next incomplete phase. Begin by auditing the current branch and mapping the phase requirements to existing code and tests. Do not write implementation code until you have stated which generic components will be retained, rewritten, or deleted.
>
> The outcome must be one surface-driven `fvm.CartesianMesher` with the typed Python interface in the specification. Tutorials declare geometry and sizing intent; they do not implement topology. Production code must never recognise a cylinder, airfoil, cube, plate, sphere, tutorial, filename, or coordinate-axis special case. Do not use external meshers, silent fallbacks, legacy wrappers, skipped tests, or fixture-specific tolerances.
>
> Follow the phase gates exactly. Add the tests and evidence required for the phase, run the relevant existing FVM suite, and write `docs/verification/cartesian_mesher/phase_<n>.md`. If a robust general algorithm cannot satisfy a gate, stop and report the precise technical blocker instead of adding a special case. Do not claim completion unless every mandatory gate for the phase passes.
>
> Preserve OpenONDA's native face-based mesh contract and use the existing authoritative topology, geometry, and validation routines. Keep the user-facing API physics/geometry-oriented and constructor-based, with complete docstrings, units, valid ranges, and a short example for every public class. Remove obsolete case-specific code rather than maintaining backward compatibility when the migration phase is reached.
>
> Credit cfMesh accurately: distinguish architectural inspiration from copied or translated code, retain all required GPL notices, name Dr Franjo Juretić and Creative Fields, and link the exact upstream sources used. Never attach the cfMesh name to behaviour that has not been traced to upstream documentation or source.

## 11. Independent completion-certification prompt

> Audit the claimed OpenONDA Cartesian-mesher phase independently. Treat the implementation agent's report as an assertion, not evidence. Read `OpenONDA_general_cartesian_mesher_program.md`, inspect the diff and current production paths, rerun every phase gate, and sample the generated meshes numerically and visually.
>
> Search for geometry-specific names, recognition logic, tutorial imports, external-mesher calls, silent fallbacks, compatibility shims, skipped/xfail tests, and weakened tolerances. Verify requested versus effective cell sizes, topology, geometry, conformance, feature capture, refinement placement, quality, determinism, and solver integration across the full applicable acceptance matrix. Verify attribution and licence provenance against exact upstream files.
>
> Return a requirement-by-requirement table with `PASS`, `FAIL`, or `NOT TESTED`, the reproduced command/output supporting each result, and file/line references for every failure. Certify the phase only if all mandatory rows are `PASS`. Otherwise provide the smallest ordered correction list and state plainly that completion is not certified.

## 12. Attribution baseline

At minimum, documentation should identify:

- cfMesh and its principal developer, Dr Franjo Juretić, and Creative Fields;
- the open-source [`wyldckat/cfMesh`](https://github.com/wyldckat/cfMesh) mirror and the exact commit used for study;
- the upstream [`cartesianMeshGenerator`](https://github.com/wyldckat/cfMesh/blob/master/meshLibrary/cartesianMesh/cartesianMeshGenerator/cartesianMeshGenerator.C) pipeline;
- upstream [`meshDict` examples](https://github.com/wyldckat/cfMesh/tree/master/tutorials/cartesianMesh), which demonstrate global surface/background sizes, patch-local refinement, primitive object refinements, and patch boundary layers;
- the official Creative Fields overview, [`cfMesh: A Novel Library for Automatic Mesh Generation`](https://cfmesh.com/cfmesh-a-novel-library-for-automatic-mesh-generation/);
- the upstream GPL notice and the provenance of any code more direct than independent reimplementation.

Suggested wording:

> OpenONDA's Cartesian mesher is an independent, solver-native implementation inspired by the open-source cfMesh Cartesian meshing workflow developed principally by Dr Franjo Juretić at Creative Fields. It uses an adaptive Cartesian template, conformal surface recovery, feature capture, mesh-quality optimisation, and optional boundary-layer construction, but exposes an OpenONDA-specific Python API and native FVM mesh representation. See the third-party notice for the exact upstream version and source files studied. Any source translated or adapted directly is identified at file level and distributed under the applicable GPL terms.

Do not use this independent-reimplementation wording for files that actually translate or copy upstream implementation details; those files need direct provenance notices.
