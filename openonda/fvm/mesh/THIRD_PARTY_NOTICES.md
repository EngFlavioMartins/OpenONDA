# Third-party notices for the Cartesian mesher

## cfMesh

The `cfmesh_*` modules of the Cartesian mesher adapt algorithms of cfMesh,
developed principally by Dr Franjo Juretić; copyright Creative Fields, Ltd.,
licensed GPL-3.0-or-later. [`CFMESH_ATTRIBUTION.md`](CFMESH_ATTRIBUTION.md)
lists each adapted module and its cfMesh counterpart. Sources:

* [Community/integration-cfmesh](https://develop.openfoam.com/Community/integration-cfmesh)
  at commit `3ff8555514827646c34cacfe5f0f691e49cdbc96`, the source the
  modules adapt;
* [wyldckat/cfMesh](https://github.com/wyldckat/cfMesh), the public source
  mirror of the original release;
* [`cartesianMeshGenerator.C`](https://github.com/wyldckat/cfMesh/blob/master/meshLibrary/cartesianMesh/cartesianMeshGenerator/cartesianMeshGenerator.C),
  for the documented high-level stage algorithm origins;
* [cfMesh Cartesian tutorials](https://github.com/wyldckat/cfMesh/tree/master/tutorials/cartesianMesh),
  for examples of global/background sizing, local refinement, primitive
  refinement objects, and patch boundary layers; and
* [Creative Fields' cfMesh overview](https://cfmesh.com/cfmesh-a-novel-library-for-automatic-mesh-generation/).

