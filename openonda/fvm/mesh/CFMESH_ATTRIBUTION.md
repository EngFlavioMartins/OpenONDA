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

| Module | cfMesh counterpart |
| --- | --- |
| `cfmesh_octree.py` | serial `meshOctreeModifier` refinement and regularity |
| `cfmesh_automatic_refinement.py` | `meshOctreeAutomaticRefinement`, `triSurfaceCurvatureEstimator` |
| `cfmesh_template.py` | Cartesian template extraction from the octree (`cartesianMeshGenerator`) |
| `cfmesh_boundary_layer.py` | `boundaryLayers::addLayerForAllPatches` |
| `cfmesh_edge_extraction.py` | `meshSurfaceEdgeExtractor` (`edgeExtraction` stage) |
| `cfmesh_surface_optimisation.py` | `meshSurfaceOptimizer` (`edgeExtraction` stage) |
| `cfmesh_mesh_optimisation.py` | `meshOptimizer` (`meshOptimisation` stage) |
| `cfmesh_volume_optimisation.py` | `volumeOptimizer`, upstream commit `3ff855551482` |

The other mesher modules (configuration, size fields, surface queries,
extrusion and reporting) are OpenONDA's own. OpenONDA has no runtime
dependency on cfMesh or OpenFOAM. Source links are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
