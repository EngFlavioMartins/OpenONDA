<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# cfMesh attribution

OpenONDA's `CartesianMesher` follows the Cartesian workflow of **cfMesh**,
developed principally by Dr. Franjo Juretić; the original copyright holder is
Creative Fields, Ltd. cfMesh is free software distributed under the GNU
General Public License, version 3 or (at your option) any later version.

The `cfmesh_*` modules in `mesh/cartesian/` adapt cfMesh algorithms to Python
and Numba on OpenONDA's face-based mesh. Where they follow cfMesh's
implementation closely they are derivative works of cfMesh, and they are
distributed under GPL-3.0-or-later, the license of both projects.

Every module follows the same upstream source: the OpenFOAM.com cfMesh
integration, [`Community/integration-cfmesh`](https://develop.openfoam.com/Community/integration-cfmesh),
commit [`3ff8555514827646c34cacfe5f0f691e49cdbc96`](https://develop.openfoam.com/Community/integration-cfmesh/-/commit/3ff8555514827646c34cacfe5f0f691e49cdbc96)
(2024-12-18), the cfMesh of OpenFOAM-v2412 and the head of its `main` branch
when checked on 2026-10-13. The mesh-parity fixtures in `tests/mesh_parity`
were generated with that build. Paths are relative to `meshLibrary/`.

| Module | cfMesh counterpart | Source directory |
| --- | --- | --- |
| `cfmesh_octree.py` | serial `meshOctreeModifier` refinement and regularity | `utilities/octrees/meshOctree/meshOctreeModifier` |
| `cfmesh_automatic_refinement.py` | `meshOctreeAutomaticRefinement`, `triSurfaceCurvatureEstimator` | `utilities/octrees/meshOctree/meshOctreeAutomaticRefinement`, `utilities/triSurfaceTools/triSurfaceCurvatureEstimator` |
| `cfmesh_template.py` | Cartesian template extraction from the octree (`cartesianMeshGenerator`) | `cartesianMesh/cartesianMeshGenerator` |
| `cfmesh_boundary_layer.py` | `boundaryLayers::addLayerForAllPatches` | `utilities/boundaryLayers/boundaryLayers` |
| `cfmesh_edge_extraction.py` | `meshSurfaceEdgeExtractor` (`edgeExtraction` stage) | `utilities/surfaceTools/meshSurfaceEdgeExtractor` |
| `cfmesh_surface_optimisation.py` | `meshSurfaceOptimizer` (`edgeExtraction` stage) | `utilities/smoothers/geometry/meshSurfaceOptimizer` |
| `cfmesh_mesh_optimisation.py` | `meshOptimizer` (`meshOptimisation` stage) | `utilities/smoothers/geometry/meshOptimizer` |
| `cfmesh_volume_optimisation.py` | `volumeOptimizer` | `utilities/smoothers/geometry/meshOptimizer/tetMeshOptimisation/advancedSmoothers/volumeOptimizer` |

The other mesher modules (configuration, size fields, surface queries,
extrusion and reporting) are OpenONDA's own. OpenONDA has no runtime
dependency on cfMesh or OpenFOAM. Source links are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
