# cfMesh algorithmic attribution

OpenONDA's `adaptive_cartesian.py` implements a Python, axis-aligned subset of
the workflow popularised by the open-source **cfMesh** Cartesian mesher.

- Principal cfMesh developer: Dr. Franjo Juretić
- Original copyright holder: Creative Fields, Ltd.
- Official open-source project page: <https://cfmesh.com/cfmesh-open-source/>
- Upstream project: <https://sourceforge.net/projects/cfmesh/>
- Historical source mirror consulted for scope and workflow:
  <https://github.com/wyldckat/cfMesh>
- cfMesh license: GNU GPL version 3 or later
- OpenONDA license: GNU GPL version 3 or later

## Scope of the adaptation

The OpenONDA implementation reproduces the high-level octree-template ideas
needed by its cube verification cases:

1. reading and validating a closed triangulated STL surface;
2. dyadic Cartesian refinement;
3. explicit 2:1 transition bands;
4. polyhedral coarse/fine interfaces made of coplanar subfaces;
5. removal of an axis-aligned solid defined by the STL; and
6. direct construction of OpenONDA's face-based FVM mesh representation.

It is an independent Python implementation and is not a line-by-line
translation of cfMesh. It currently does **not** claim cfMesh compatibility or
implement its general surface mapping, feature/corner extraction, mesh
untangling and optimisation, arbitrary geometry repair, MPI meshing, or
boundary-layer extrusion. Those are substantial separate algorithms in the
upstream project. The current STL-driven implementation deliberately rejects
surfaces whose triangles do not describe a closed axis-aligned solid.
