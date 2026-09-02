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
6. direct construction of OpenONDA's face-based FVM mesh representation;
7. graded, conformal O-grid boundary layers for extruded cylinders; and
8. general closed-STL feature classification, surface-normal prismatic-layer
   advancement, size transition, and solver-native volume-core conversion.

Since the geometry-preserving policy was introduced, the body is the
authority: the STL body is never snapped, stretched or inflated.  On the exact
Cartesian-box path, the lattice is resolved so that every body face is an
exact lattice plane (the finest spacing is refined when the requested spacing
does not divide a body extent, and the outer domain is padded outward until the
lattice tiles it). On curved and general-body paths, wall vertices remain on
the triangulated surface while conformal cells advance into the fluid. The
mesh is validated at build time so that cells have positive volume and the
wall patch matches the STL surface. This mirrors cfMesh's core principle that
the input geometry is preserved while the volume mesh conforms to it.

It is an independent Python implementation and is not a line-by-line
translation of cfMesh. It does **not** claim file-format or implementation
compatibility with cfMesh. General-body layer advancement and core filling use
the in-process Gmsh library API already shipped as an OpenONDA dependency; no
external mesher executable or intermediate solver case is required. Geometry
repair, MPI-distributed meshing, and cfMesh's exact optimiser are outside the
implemented scope. General bodies must be closed, watertight STL surfaces and
must have adequate layer clearance inside the requested domain.
