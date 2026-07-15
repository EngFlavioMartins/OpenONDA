#!/usr/bin/env python3
"""Transfinite hex channel mesh for the stepProfile tutorial (gmsh).

A 1 x 0.1 channel extruded one cell in z.  Patches: ``inlet`` (x = 0),
``outlet`` (x = L), ``walls`` (y = 0, y = W), ``frontAndBack`` (z, empty).
"""

import gmsh


def generate_mesh(msh_path, nx, ny):
    L = 1.0
    W = 0.1
    D = 0.01

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("stepProfile")

    p1 = gmsh.model.occ.addPoint(0, 0, 0)
    p3 = gmsh.model.occ.addPoint(L, 0, 0)
    p4 = gmsh.model.occ.addPoint(L, W, 0)
    p5 = gmsh.model.occ.addPoint(0, W, 0)

    l1 = gmsh.model.occ.addLine(p1, p3)
    l2 = gmsh.model.occ.addLine(p3, p4)
    l3 = gmsh.model.occ.addLine(p4, p5)
    l4 = gmsh.model.occ.addLine(p5, p1)

    loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.extrude([(2, surf)], 0, 0, D, numElements=[1], heights=[1], recombine=True)
    gmsh.model.occ.synchronize()

    tol_e = 1e-6
    for c_dim, c_tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(c_dim, c_tag)
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        if dx > tol_e and dy < tol_e and dz < tol_e:
            gmsh.model.mesh.setTransfiniteCurve(c_tag, nx + 1)
        elif dy > tol_e and dx < tol_e and dz < tol_e:
            gmsh.model.mesh.setTransfiniteCurve(c_tag, ny + 1)
        else:
            gmsh.model.mesh.setTransfiniteCurve(c_tag, 2)

    for s_dim, s_tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s_tag)
        gmsh.model.mesh.setRecombine(s_dim, s_tag)

    for v_dim, v_tag in gmsh.model.getEntities(3):
        gmsh.model.mesh.setTransfiniteVolume(v_tag)

    tol = 1e-6
    boundary_surfaces = gmsh.model.getBoundary(gmsh.model.getEntities(3), oriented=False)
    inlet, outlet, walls, front_and_back = [], [], [], []

    for dim, tag in boundary_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

        if abs(zmin) < tol and abs(zmax) < tol:
            front_and_back.append(tag)
        elif abs(zmin - D) < tol and abs(zmax - D) < tol:
            front_and_back.append(tag)
        elif abs(xmin) < tol and abs(xmax) < tol:
            inlet.append(tag)
        elif abs(xmax - L) < tol and abs(xmin - L) < tol:
            outlet.append(tag)
        else:
            walls.append(tag)

    gmsh.model.addPhysicalGroup(2, inlet, 1, "inlet")
    gmsh.model.addPhysicalGroup(2, outlet, 2, "outlet")
    gmsh.model.addPhysicalGroup(2, walls, 3, "walls")
    gmsh.model.addPhysicalGroup(2, front_and_back, 4, "frontAndBack")
    gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], 1, "fluid")

    gmsh.model.mesh.generate(3)
    gmsh.write(msh_path)
    print(f"  Mesh written: {msh_path}")
