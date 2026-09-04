# ADR: one surface-driven Cartesian mesher

*Status: accepted direction for the mesher programme; Phase 0 contract freeze*

*Date: 2026-09-02*

## Context

OpenONDA currently exposes several meshing paths with different scopes. The
adaptive Cartesian implementation contains the octree, topology extraction,
surface projection, and a cylinder-specific O-grid in one module. The public
`ExplicitCylinderGridMesher` additionally encodes a grid-study layout. The
`GeneralBodyMesher` uses the Gmsh library for advancing layers and core fill.
These paths make a tutorial geometry part of the solver-facing meshing API and
make it difficult to prove that arbitrary closed surfaces receive the same
algorithm.

The engineering specification in [`mesher_project.md`](../../mesher_project.md)
requires a native, surface-driven Cartesian mesher with a typed declarative
configuration. The existing face-based mesh contract, geometry calculations,
topology routines, and validation routines remain useful and authoritative.

## Decision

OpenONDA will converge on one public meshing namespace and one public
Cartesian mesher. Solver configuration stays in `openonda.fvm`; mesh
construction is imported from `openonda.fvm.mesher`:

```python
import openonda.fvm.mesher as msh

msh.CartesianMesher(
    domain=msh.BoxDomain(...),
    surfaces=(msh.STLSurface(...),),
    max_cell_size=0.50,
    boundary_cell_size=0.0625,
    min_cell_size=0.015625,
    refinements=(...),
    features=msh.FeatureRefinement(...),
    boundary_layers=(...),
)
```

The flat paths `openonda.fvm.BoxRefinement`, `openonda.fvm.STLSurface`, and
the other mesher-object exports are deliberately absent. The namespace also
contains the mesher's refinement, surface, layer, report, and structured-mesh
helpers, so users can keep mesh construction under one `msh` import without
mixing mesh intent into the solver facade.

Configuration objects express physical intent and are immutable after
construction. `build()` and `__call__()` return OpenONDA's native face-based
mesh data. Requested sizes and the dyadic effective sizes are reported. Patch
names come from the surface/domain configuration and are preserved through
assembly. Boundary-condition physics remain in `FVMSetup`.

The implementation is split into independently testable stages under
`source/solvers/fvm/mesh/cartesian/`:

1. configuration and validation;
2. surface ingestion, spatial queries, and feature classification;
3. combined size field and balanced octree construction;
4. fluid extraction and native face topology;
5. transactional surface recovery;
6. generic patch-normal boundary layers;
7. quality optimisation, canonical numbering, and reporting.

Existing generic mesh geometry, topology, and validation functions are reused;
the new pipeline must not duplicate their calculations. No production branch
may identify a cylinder, airfoil, cube, plate, sphere, tutorial, filename, or
coordinate-axis special case. External meshing executables and external
meshing APIs are outside the native `CartesianMesher` contract.

## Alternatives rejected

* Retaining `ExplicitCylinderGridMesher` would preserve a case-specific public
  algorithm and would make the required geometry-independence claim false.
* Keeping the cylinder O-grid as a hidden fallback would make boundary-layer
  behavior depend on geometry recognition and would violate the generic layer
  contract.
* Routing `CartesianMesher` through Gmsh, cfMesh, OpenFOAM, or another
  executable would not produce a solver-native mesh from the requested Python
  API and would make provenance and determinism harder to audit.
* Treating nearest-point projection failures as successful staircase cells
  would hide a conformance failure. Surface recovery must either return a
  valid conformal result or raise a stage-specific diagnostic.

## Consequences

The current adaptive, general-body, and explicit-cylinder paths are baseline
evidence, not target behavior. Migration will require API changes in tutorials
and deletion of obsolete public exports. The acceptance matrix must cover
rotated, translated, smooth, concave, multi-body, and deliberately invalid
surfaces before the mesher can be certified.

This ADR records the architecture and contract only. It does not claim that
any later phase is complete.

## Provenance and licensing

The intended implementation is an independent OpenONDA implementation
inspired by the documented cfMesh Cartesian workflow. The initial architecture
phase copied or translated no cfMesh source. The exact upstream study commit
and any file-level provenance must be recorded before direct translation.
Current topology-parity limitations are recorded in
[`cfmesh_topology_parity.md`](../verification/cartesian_mesher/cfmesh_topology_parity.md).
