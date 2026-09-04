# cfMesh topology-parity audit

*Date: 2026-09-04*
*Status: **NOT COMPLETE — parity is disproved by reproduced evidence.***

This audit supersedes any visual or quality-only claim that OpenONDA's
`CartesianMesher` reproduces cfMesh topology.  The oracle is the locally built
OpenFOAM-v2412 cfMesh `cartesianMesh` executable and its corresponding GPL
source.  Production OpenONDA code still must not invoke that executable.

## Reproduced cfMesh reference

For the D/12 cylinder `meshDict` (`maxCellSize=2/3`,
`boundaryCellSize=minCellSize=1/12`, 2D/4D box refinements), cfMesh reports:

| Stage | Reproduced value |
| --- | ---: |
| Root octree box | `(-15.3333 -21.3333 -21.3333)` to `(27.3333 21.3333 21.3333)` |
| Background octree level | 6 |
| Boundary octree level | 9 |
| Extracted cells | 249,004 |
| Extracted faces | 875,792 |
| Extracted points | 377,797 |
| Cells after patch correction | 249,791 |
| Default-wrapper cells added | 144,443 |
| Final cells | 394,234 |
| Final faces | 1,312,951 |
| Final points | 524,508 |

The final mesh contains 356,819 hexahedra, 214 prisms, 295 pyramids, 574
tetrahedra, and 36,332 general polyhedra.  `checkMesh -allTopology
-allGeometry` passes topology, positive volume, face pyramids,
non-orthogonality, skewness, and interpolation-weight checks, but reports eight
low-quality/negative face-decomposition tets.  Therefore the OpenFOAM oracle
itself must not be described as passing every available `checkMesh` check.

## Source-confirmed algorithmic differences

1. cfMesh preserves the requested background size by resizing a cubic root
   octree; OpenONDA currently changes `2/3` to `0.4` so the background tiles
   the requested box.
2. cfMesh refines surface-intersected root-octree leaves and then extracts the
   fluid mesh.  OpenONDA starts from a box-fitted lattice and uses a different
   cut-cell/snap recovery path.
3. cfMesh always calls `boundaryLayers::addLayerForAllPatches()`.  Its default
   wrapper covers every non-empty boundary patch and creates face-, edge-, and
   corner-associated cells.  OpenONDA's current wrapper covers only selected
   smooth body patches and creates one cell per selected face.
4. OpenONDA's four-ring, factor-0.5 displacement propagation is not present in
   cfMesh.  cfMesh instead runs its surface optimizer and finite-volume mesh
   optimizer/untangler.
5. The checked-in long-cylinder STL has 1,280 triangles and crosses the
   spanwise domain boundaries.  The reproduced cfMesh oracle surface has 508
   cylinder triangles, closed end caps at `z=+/-0.5`, plus 12 outer-box
   triangles.  These are not normalized inputs and cannot support a
   cell-for-cell comparison.

Using the exact 508-triangle cfMesh cylinder as OpenONDA input exposes two
additional recovery failures: a rank-one tangent cut was incorrectly passed
to Delaunay triangulation, and after that case was handled explicitly, the
cut-cell face-orientation transaction still fails to converge.  Thus generic
same-input meshing does not currently complete.

## Completion gate

Parity may be claimed only after a normalized-input test proves all of the
following:

- identical root-box and leaf refinement coordinates/levels;
- identical cell, face, point, and per-patch face counts at every cfMesh stage;
- an isomorphism between final owner/neighbour/face connectivity and patch IDs;
- vertex coordinates equal within a declared floating-point tolerance;
- repeated deterministic builds;
- OpenONDA topology/geometry validation, ParaView/VTK export, and at least one
  accepted FVM step.

No current result satisfies this gate.
