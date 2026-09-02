# Third-party notices for the Cartesian mesher

## cfMesh architectural inspiration

OpenONDA's planned Cartesian mesher is an independent, solver-native
implementation inspired by the open-source cfMesh Cartesian meshing workflow
developed principally by Dr Franjo Juretić at Creative Fields. The planned
implementation has its own Python API and OpenONDA native face-based mesh
representation.

The sources consulted for the Phase 0 architecture contract are:

* [wyldckat/cfMesh](https://github.com/wyldckat/cfMesh), the public source
  mirror;
* [`cartesianMeshGenerator.C`](https://github.com/wyldckat/cfMesh/blob/master/meshLibrary/cartesianMesh/cartesianMeshGenerator/cartesianMeshGenerator.C),
  for the documented high-level stage lineage;
* [cfMesh Cartesian tutorials](https://github.com/wyldckat/cfMesh/tree/master/tutorials/cartesianMesh),
  for examples of global/background sizing, local refinement, primitive
  refinement objects, and patch boundary layers; and
* [Creative Fields' cfMesh overview](https://cfmesh.com/cfmesh-a-novel-library-for-automatic-mesh-generation/).

No cfMesh source file has been copied or translated into OpenONDA in Phase 0.
Consequently, there is no upstream commit or file-level translated-code
provenance to claim yet. The implementation phase must pin the exact upstream
commit studied and update this notice if any implementation detail is copied
or translated. It must also preserve the applicable GPL notices and distinguish
independent reimplementation from derivative code.
